"""In-silico mixture simulation, simplex-constrained deconvolution, and metrics.

Mixture model (see docs/cfdna_deconvolution.md sec. 3):

    y  ~  R w ,     w >= 0,  sum(w) = 1

with `R` column-normalized so each cell type's signature is a distribution over
regions. A finite-depth cfDNA observation is modeled by drawing `N` fragments
from the clean mixture `p = R w*` (multinomial) -- this makes low depth show up
as shot noise, exactly the regime that limits real cfDNA. Deconvolution solves a
non-negative, sum-to-one constrained least squares (Houseman/MethAtlas recipe).
No cvxpy/quadprog dependency: projection-onto-the-simplex gradient descent.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import nnls


# ---------------------------------------------------------------------------
# simulation
# ---------------------------------------------------------------------------
def sample_fractions(m, n_active, rng, dominant=None, dom_frac=None,
                     alpha=1.0):
    """Draw a ground-truth fraction vector on the m-simplex.

    n_active cell types get non-zero mass (the rest are exactly 0). If
    `dominant` (index) and `dom_frac` are given, that type is pinned near
    `dom_frac` (a WBC-dominated plasma regime) and the remainder is Dirichlet
    over the other active types.
    """
    w = np.zeros(m)
    idx = rng.choice(m, size=n_active, replace=False)
    if dominant is not None and dom_frac is not None:
        if dominant not in idx:
            idx[0] = dominant
        rest = [i for i in idx if i != dominant]
        w[dominant] = dom_frac
        if rest:
            w[rest] = (1.0 - dom_frac) * rng.dirichlet(np.full(len(rest), alpha))
    else:
        w[idx] = rng.dirichlet(np.full(n_active, alpha))
    return w


def simulate_observation(R, w, n_frags, rng):
    """Finite-depth cfDNA-like observation: multinomial draw of `n_frags`.

    Returns the observed feature vector y (normalized counts, sums to 1).
    n_frags = inf returns the noiseless mixture R w.
    """
    p = R @ w
    p = np.clip(p, 0, None)
    p = p / p.sum()
    if not np.isfinite(n_frags):
        return p
    counts = rng.multinomial(int(n_frags), p)
    return counts / counts.sum()


# ---------------------------------------------------------------------------
# solver: min ||y - R w||^2  s.t.  w >= 0, sum(w) = 1
# ---------------------------------------------------------------------------
def _project_simplex(v):
    """Euclidean projection of vector v onto the probability simplex (Duchi 2008)."""
    n = len(v)
    u = np.sort(v)[::-1]
    css = np.cumsum(u) - 1.0
    rho = np.nonzero(u - css / (np.arange(n) + 1) > 0)[0][-1]
    theta = css[rho] / (rho + 1.0)
    return np.maximum(v - theta, 0.0)


def deconvolve(y, R, l2=0.0, iters=4000, tol=1e-9):
    """Simplex-constrained least squares via projected gradient descent.

    Warm-started from NNLS+renormalize. Accelerated (Nesterov-ish) PGD with a
    Lipschitz step size. `l2` adds ridge regularization on w (helps when R is
    ill-conditioned / columns near-collinear).
    """
    n, m = R.shape
    # warm start: NNLS then renormalize onto the simplex
    try:
        w0, _ = nnls(R, y, maxiter=10 * m)
        w = _project_simplex(w0) if w0.sum() > 0 else np.full(m, 1.0 / m)
    except Exception:
        w = np.full(m, 1.0 / m)

    RtR = R.T @ R + l2 * np.eye(m)
    Rty = R.T @ y
    # step size ~ 1/L, L = largest eigenvalue of RtR (power iteration, cheap)
    v = np.random.default_rng(0).standard_normal(m)
    for _ in range(50):
        v = RtR @ v
        v /= (np.linalg.norm(v) + 1e-12)
    L = float(v @ (RtR @ v)) + 1e-12
    step = 1.0 / L

    z = w.copy()
    t = 1.0
    prev = w.copy()
    for _ in range(iters):
        grad = RtR @ z - Rty
        w = _project_simplex(z - step * grad)
        t_next = 0.5 * (1 + np.sqrt(1 + 4 * t * t))
        z = w + ((t - 1) / t_next) * (w - prev)
        if np.linalg.norm(w - prev) < tol:
            prev = w
            break
        prev = w
        t = t_next
    return w


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def recovery_metrics(w_true, w_hat):
    """Per-mixture recovery quality of estimated vs true fractions."""
    wt, wh = np.asarray(w_true), np.asarray(w_hat)
    out = {
        "l1_error": float(np.abs(wt - wh).sum()),          # total variation * 2
        "rmse": float(np.sqrt(np.mean((wt - wh) ** 2))),
        "mae": float(np.mean(np.abs(wt - wh))),
    }
    if wt.std() > 0 and wh.std() > 0:
        out["pearson"] = float(np.corrcoef(wt, wh)[0, 1])
    else:
        out["pearson"] = float("nan")
    return out


def detection_stats(w_true, w_hat, present_thresh=1e-3, call_thresh=1e-2):
    """Present-vs-absent detection: is a truly-present cell type called?"""
    present = w_true >= present_thresh
    called = w_hat >= call_thresh
    tp = int(np.sum(present & called))
    fn = int(np.sum(present & ~called))
    fp = int(np.sum(~present & called))
    tn = int(np.sum(~present & ~called))
    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "sensitivity": tp / max(1, tp + fn),
            "precision": tp / max(1, tp + fp)}


def condition_diagnostics(R):
    """Collinearity diagnostics for the reference matrix."""
    s = np.linalg.svd(R, compute_uv=False)
    C = np.corrcoef(R.T)
    off = C[~np.eye(C.shape[0], dtype=bool)]
    return {
        "cond_number": float(s[0] / (s[-1] + 1e-12)),
        "min_singular": float(s[-1]),
        "mean_abs_col_corr": float(np.mean(np.abs(off))),
        "max_col_corr": float(np.max(off)),
    }
