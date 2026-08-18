"""Cell-type DEVIATION probe: score and train the cell-type-specific part of the
nucleosome profile, not the shared backbone.

Longer training was measured to improve absolute profile correlation while the cell-type
deviation collapsed (0.184 -> 0.048) -- the Role-b failure mode. Selecting on val_loss or
absolute profile therefore discards the signal we want. This module supplies:

  probe   : the same genomic windows evaluated across MANY cell types (the ordinary loader
            never yields that -- each sample there carries one cell type), with observed
            nucleosome profiles cached once.
  score() : mean per-(window, cell) Pearson between predicted and observed DEVIATION from
            the across-cell consensus -> a validation metric to select checkpoints on.
  loss()  : the differentiable form of the same quantity, to optimize directly.

Deviation is defined on occupancy-normalized profiles, so it measures repositioning rather
than coverage: dev(c,w) = norm(profile[c,w]) - mean_c' norm(profile[c',w]).
"""
from __future__ import annotations

import numpy as np
import pyBigWig
import pyfaidx
import torch
import torch.nn.functional as F

from chrombpnet.training.utils import one_hot


def _gauss_kernel(sigma: float, device, dtype):
    r = max(int(3 * sigma), 1)
    x = torch.arange(-r, r + 1, device=device, dtype=dtype)
    k = torch.exp(-(x ** 2) / (2 * sigma * sigma))
    return (k / k.sum()).view(1, 1, -1)


def _smooth(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Gaussian-smooth the last dim. The raw dyad target is far too sparse to compare
    per-base (raw profile r ~0.07 vs ~0.39 smoothed) -- without this the deviation is
    shot noise, not positioning. Differentiable, so score() and loss() share it."""
    if not sigma:
        return x
    shp = x.shape
    k = _gauss_kernel(sigma, x.device, x.dtype)
    y = F.conv1d(x.reshape(-1, 1, shp[-1]), k, padding=k.shape[-1] // 2)
    return y.reshape(*shp[:-1], y.shape[-1])


def _norm(x: torch.Tensor) -> torch.Tensor:
    """Normalize the last dim to a distribution (occupancy removed)."""
    s = x.sum(-1, keepdim=True)
    return x / torch.where(s > 0, s, torch.ones_like(s))


def _centered(x: torch.Tensor, sigma: float = 0.0) -> torch.Tensor:
    """Deviation from the across-cell consensus. x: (W, K, L) -> same shape."""
    n = _norm(_smooth(x, sigma))
    return n - n.mean(dim=1, keepdim=True)


class DeviationProbe:
    """Fixed (windows x cell types) probe with cached observed nucleosome profiles."""

    def __init__(self, regions, genome, nuc_bws, ct_idx, inputlen, outputlen,
                 n_windows=256, seed=0, min_counts=30, smooth_sigma=20.0):
        rng = np.random.default_rng(seed)
        g = pyfaidx.Fasta(genome)
        bws = [pyBigWig.open(p) for p in nuc_bws]
        half_in, half_out = inputlen // 2, outputlen // 2
        seqs, obs, kept = [], [], 0
        order = rng.permutation(len(regions))
        for i in order:
            if kept >= n_windows:
                break
            c, center = regions[i]
            prof = []
            ok = True
            for bw in bws:
                size = bw.chroms().get(c, 0)
                s = center - half_out
                if s < 0 or s + outputlen > size or center - half_in < 0:
                    ok = False
                    break
                v = np.nan_to_num(np.asarray(bw.values(c, s, s + outputlen), np.float32))
                if v.sum() < min_counts:
                    ok = False
                    break
                prof.append(v)
            if not ok:
                continue
            sq = str(g[c][center - half_in: center - half_in + inputlen]).upper()
            sq = (sq + "N" * inputlen)[:inputlen]
            seqs.append(one_hot.dna_to_one_hot([sq])[0].astype(np.float32))
            obs.append(np.stack(prof))
            kept += 1
        for bw in bws:
            bw.close()
        if not kept:
            raise ValueError(
                f"deviation probe is empty: no window had >={min_counts} dyad counts in ALL "
                f"{len(nuc_bws)} probe cell types. Use deeper probe cells or lower min_counts.")
        self.n = kept
        self.seq = torch.from_numpy(np.stack(seqs)) if kept else None          # (W, IL, 4)
        self.obs = torch.from_numpy(np.stack(obs)) if kept else None           # (W, K, OL)
        self.ct_idx = torch.tensor(ct_idx, dtype=torch.long)                   # (K,)
        self.sigma = smooth_sigma
        self.obs_dev = _centered(self.obs.double(), smooth_sigma).float()

    def _predict_dev(self, model, sel, device, amp=False):
        """Predicted deviation for probe windows `sel`. Returns (n, K, OL)."""
        K = len(self.ct_idx)
        seq = self.seq[sel].to(device)                                    # (n, IL, 4)
        n = seq.shape[0]
        x = seq.repeat_interleave(K, dim=0)                               # (n*K, IL, 4)
        ct = self.ct_idx.to(device).repeat(n)                             # (n*K,)
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=(amp and device.type == "cuda")):
            out = model(x, ct)
        prob = torch.softmax(out[2].float(), dim=-1).reshape(n, K, -1)
        return _centered(prob, self.sigma)

    @torch.no_grad()
    def score(self, model, device, batch_windows=16, amp=False) -> float:
        """Mean per-(window, cell) Pearson(pred_dev, obs_dev). Higher = more cell-type signal."""
        if not self.n:
            return float("nan")
        was_training = model.training
        model.eval()
        rs = []
        for i in range(0, self.n, batch_windows):
            sel = slice(i, min(i + batch_windows, self.n))
            pd_ = self._predict_dev(model, sel, device, amp)
            od = self.obs_dev[sel].to(device)
            a = pd_ - pd_.mean(-1, keepdim=True)
            b = od - od.mean(-1, keepdim=True)
            den = a.norm(dim=-1) * b.norm(dim=-1)
            r = (a * b).sum(-1) / torch.where(den > 0, den, torch.ones_like(den))
            rs.append(r.flatten().cpu())
        model.train(was_training)
        return float(torch.cat(rs).mean())

    def loss(self, model, device, batch_windows=8, amp=False, generator=None):
        """Differentiable 1 - mean Pearson(pred_dev, obs_dev) over a random probe batch."""
        if not self.n:
            return None
        idx = torch.randint(0, self.n, (min(batch_windows, self.n),), generator=generator)
        pd_ = self._predict_dev(model, idx, device, amp)
        od = self.obs_dev[idx].to(device)
        a = pd_ - pd_.mean(-1, keepdim=True)
        b = od - od.mean(-1, keepdim=True)
        den = a.norm(dim=-1) * b.norm(dim=-1) + 1e-8
        return 1.0 - ((a * b).sum(-1) / den).mean()
