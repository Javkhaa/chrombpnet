"""Shared evaluation metrics for the multi-task models (trainer + standalone eval).

Primitives (pearson/spearman/jsd/auroc) plus `heldout_metrics`, which runs inference
over selected dataset rows and returns a flat, wandb-friendly metric dict for both
heads. Used for in-training held-out logging and by evaluate_multitask.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch


def pearson(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 2 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    return pearson(pd.Series(x).rank().values, pd.Series(y).rank().values)


def jsd(p, q, eps=1e-12):
    """Jensen-Shannon divergence (base 2) between two non-negative vectors."""
    p = np.asarray(p, float) + eps; p /= p.sum()
    q = np.asarray(q, float) + eps; q /= q.sum()
    m = 0.5 * (p + q)
    kl = lambda a, b: float(np.sum(a * np.log2(a / b)))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def auroc(scores, labels):
    """AUROC via the rank (Mann-Whitney U) identity."""
    scores = np.asarray(scores, float); labels = np.asarray(labels, bool)
    n_pos = int(labels.sum()); n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(scores).rank().values
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _forward(model, seq, ct):
    return model(seq, ct) if ct is not None else model(seq)


@torch.no_grad()
def heldout_metrics(model, dataset, indices, is_peak_sub, device, outputlen,
                    batch_size=128, multicell=False, prefix="val"):
    """Inference over dataset[indices] -> per-head metric dict.

    dataset[i] yields (seq, acc, acc_lc, nuc, nuc_lc[, ct_idx]); is_peak_sub is the
    peak/nonpeak label aligned to `indices`. Returns keys like '<prefix>/acc/counts_pearson'.
    """
    model.eval()
    is_peak_sub = np.asarray(is_peak_sub, bool)
    acc_plc, acc_olc, nuc_plc, nuc_olc = [], [], [], []
    acc_jsd, acc_pear, nuc_jsd, nuc_pear = [], [], [], []
    for s in range(0, len(indices), batch_size):
        idx = indices[s:s + batch_size]
        items = [dataset[i] for i in idx]
        seq = torch.stack([it[0] for it in items]).to(device)
        acc = torch.stack([it[1] for it in items]).numpy()
        nuc = torch.stack([it[3] for it in items]).numpy()
        ct = torch.stack([it[5] for it in items]).to(device) if multicell else None
        acc_profile, acc_count, nuc_profile, nuc_count = _forward(model, seq, ct)
        ap = torch.softmax(acc_profile, -1).cpu().numpy()
        npf = torch.softmax(nuc_profile, -1).cpu().numpy()
        acc_plc.append(acc_count.squeeze(-1).cpu().numpy())
        nuc_plc.append(nuc_count.squeeze(-1).cpu().numpy())
        acc_olc.append(np.log1p(acc.sum(1)))
        nuc_olc.append(np.log1p(nuc.sum(1)))
        bpk = is_peak_sub[s:s + len(idx)]
        for i in range(len(idx)):
            for prob, obs, jl, pl in ((ap[i], acc[i], acc_jsd, acc_pear),
                                      (npf[i], nuc[i], nuc_jsd, nuc_pear)):
                if bpk[i] and obs.sum() > 0:
                    jl.append(jsd(prob, obs)); pl.append(pearson(prob, obs))
                else:
                    jl.append(np.nan); pl.append(np.nan)
    acc_plc = np.concatenate(acc_plc); acc_olc = np.concatenate(acc_olc)
    nuc_plc = np.concatenate(nuc_plc); nuc_olc = np.concatenate(nuc_olc)
    pk = is_peak_sub
    out = {
        f"{prefix}/acc/counts_pearson": pearson(acc_plc[pk], acc_olc[pk]),
        f"{prefix}/acc/counts_spearman": spearman(acc_plc[pk], acc_olc[pk]),
        f"{prefix}/acc/profile_jsd": float(np.nanmedian(acc_jsd)),
        f"{prefix}/acc/profile_pearson": float(np.nanmedian(acc_pear)),
        f"{prefix}/nuc/counts_pearson": pearson(nuc_plc[pk], nuc_olc[pk]),
        f"{prefix}/nuc/counts_spearman": spearman(nuc_plc[pk], nuc_olc[pk]),
        f"{prefix}/nuc/profile_jsd": float(np.nanmedian(nuc_jsd)),
        f"{prefix}/nuc/profile_pearson": float(np.nanmedian(nuc_pear)),
        f"{prefix}/peak_vs_nonpeak_auroc_acc": auroc(acc_plc, pk),
    }
    return out


@torch.no_grad()
def example_profile_figures(model, dataset, indices, device, outputlen,
                            multicell=False, n=3):
    """Return up to `n` matplotlib figures of predicted-vs-observed profiles (both heads)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    model.eval()
    figs = []
    xx = np.arange(outputlen) - outputlen // 2
    for i in indices[:n]:
        item = dataset[i]
        seq = item[0].unsqueeze(0).to(device)
        ct = item[5].unsqueeze(0).to(device) if multicell else None
        acc_profile, acc_count, nuc_profile, nuc_count = _forward(model, seq, ct)
        acc_pred = (torch.softmax(acc_profile, -1).squeeze(0).cpu().numpy()
                    * float(torch.expm1(acc_count)))
        nuc_pred = (torch.softmax(nuc_profile, -1).squeeze(0).cpu().numpy()
                    * float(torch.expm1(nuc_count)))
        acc_obs = item[1].numpy(); nuc_obs = item[3].numpy()
        fig, ax = plt.subplots(2, 1, figsize=(9, 4.5), sharex=True)
        ax[0].plot(xx, acc_pred, lw=1, label="pred"); ax[0].plot(xx, acc_obs, lw=1, alpha=.6, label="obs")
        ax[0].set_title(f"accessibility (region {i})"); ax[0].legend(fontsize=7)
        ax[1].plot(xx, nuc_pred, lw=1, label="pred"); ax[1].plot(xx, nuc_obs, lw=1, alpha=.6, label="obs")
        ax[1].set_title("nucleosome"); ax[1].legend(fontsize=7)
        fig.tight_layout()
        figs.append(fig)
    return figs
