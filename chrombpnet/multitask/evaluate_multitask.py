#!/usr/bin/env python3
"""Held-out evaluation for the multi-task accessibility + nucleosome model.

Computes metrics on the fold's TEST chromosomes (which training never sees;
valid is consumed by early stopping), for BOTH heads:

  Tier 1 (fit):
    - counts Pearson / Spearman: predicted log-count vs observed log1p(total), over peaks
    - profile JSD (median): softmax(profile) vs observed normalized profile, over peaks
                            (with a flat-profile baseline for reference)
    - profile Pearson (median): per-base predicted vs observed, over peaks
    - held-out multinomial-NLL + total loss
  Tier 2 (behaviour):
    - peak-vs-nonpeak AUROC from predicted accessibility log-count
    - reverse-complement count consistency (subsample)

Runs on CPU by default so it does not contend with GPU training.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from chrombpnet.multitask.torch_data import MultiTaskRegionDataset
from chrombpnet.multitask.train_multitask_torch import load_regions, NARROWPEAK_COLS
from chrombpnet.training.models.multitask_nucleosome_torch import (
    MultiTaskNucleosomeModel, multitask_loss, multinomial_nll,
)


# ---- metric helpers -------------------------------------------------------
def _pearson(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 2 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x, y):
    xr = pd.Series(x).rank().values
    yr = pd.Series(y).rank().values
    return _pearson(xr, yr)


def _jsd(p, q, eps=1e-12):
    """Jensen-Shannon divergence (base 2) between two prob vectors."""
    p = np.asarray(p, float) + eps; p /= p.sum()
    q = np.asarray(q, float) + eps; q /= q.sum()
    m = 0.5 * (p + q)
    kl = lambda a, b: np.sum(a * np.log2(a / b))
    return float(0.5 * kl(p, m) + 0.5 * kl(q, m))


def _auroc(scores, labels):
    """AUROC via the rank (Mann-Whitney U) identity."""
    scores = np.asarray(scores, float); labels = np.asarray(labels, bool)
    n_pos = int(labels.sum()); n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(scores).rank().values
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _subsample(df, n, rng):
    if df is None or len(df) == 0 or n is None or len(df) <= n:
        return df
    idx = rng.choice(len(df), size=n, replace=False)
    return df.iloc[idx].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('-c', '--checkpoint', required=True)
    ap.add_argument('-p', '--peaks', required=True)
    ap.add_argument('-n', '--nonpeaks', required=True)
    ap.add_argument('-g', '--genome', required=True)
    ap.add_argument('--acc-bw', required=True)
    ap.add_argument('--nuc-bw', required=True)
    ap.add_argument('-fl', '--fold-json', required=True)
    ap.add_argument('--split', default='test', choices=['test', 'valid', 'train'])
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--num-workers', type=int, default=12)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--max-peaks', type=int, default=8000, help='subsample peaks for speed (0=all)')
    ap.add_argument('--max-nonpeaks', type=int, default=8000, help='subsample nonpeaks for speed (0=all)')
    ap.add_argument('--revcomp-check', type=int, default=512, help='regions for revcomp consistency (0=skip)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('-o', '--out-json', default=None)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)

    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    a = ckpt['args']
    model = MultiTaskNucleosomeModel(a['inputlen'], a['outputlen'], a['filters'], a['n_dil_layers'])
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()
    INPUTLEN, OUTPUTLEN = a['inputlen'], a['outputlen']

    fold = json.load(open(args.fold_json))
    chroms = set(fold[args.split])

    peaks_df = load_regions(args.peaks, chroms)
    nonpeaks_df = load_regions(args.nonpeaks, chroms)
    peaks_df = _subsample(peaks_df, args.max_peaks or None, rng)
    nonpeaks_df = _subsample(nonpeaks_df, args.max_nonpeaks or None, rng)
    print(f"split={args.split} chroms={sorted(chroms)} | peaks={len(peaks_df)} nonpeaks={len(nonpeaks_df)}")

    # Deterministic eval: no jitter, no revcomp, no shuffle, keep all (ratio>=1).
    ds = MultiTaskRegionDataset(
        peaks_df, nonpeaks_df, args.genome, args.acc_bw, args.nuc_bw,
        INPUTLEN, OUTPUTLEN, max_jitter=0, negative_sampling_ratio=1.0,
        add_revcomp=False, shuffle=False, seed=args.seed,
    )
    is_peak = ds.regions['is_peak'].values.astype(bool)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=False)

    # accumulators
    rows = {h: {'pred_lc': [], 'obs_lc': [], 'jsd': [], 'jsd_flat': [], 'pear': [], 'is_peak': []}
            for h in ('acc', 'nuc')}
    loss_sum, n_seen = 0.0, 0
    ptr = 0
    flat = np.full(OUTPUTLEN, 1.0 / OUTPUTLEN)

    with torch.no_grad():
        for batch in loader:
            seq, acc, acc_lc, nuc, nuc_lc = [x.to(device) for x in batch]
            out = model(seq)
            loss = multitask_loss(out, (acc, acc_lc, nuc, nuc_lc))
            loss_sum += float(loss) * seq.shape[0]; n_seen += seq.shape[0]
            acc_profile, acc_count, nuc_profile, nuc_count = out
            bs = seq.shape[0]
            batch_is_peak = is_peak[ptr:ptr + bs]; ptr += bs

            for head, profile, count, obs in (
                ('acc', acc_profile, acc_count, acc),
                ('nuc', nuc_profile, nuc_count, nuc),
            ):
                prob = torch.softmax(profile, dim=-1).cpu().numpy()
                pred_lc = count.squeeze(-1).cpu().numpy()
                obs_np = obs.cpu().numpy()
                obs_lc = np.log1p(obs_np.sum(axis=1))
                R = rows[head]
                R['pred_lc'].append(pred_lc); R['obs_lc'].append(obs_lc)
                R['is_peak'].append(batch_is_peak)
                for i in range(bs):
                    if not batch_is_peak[i]:
                        R['jsd'].append(np.nan); R['jsd_flat'].append(np.nan); R['pear'].append(np.nan)
                        continue
                    o = obs_np[i]
                    if o.sum() <= 0:
                        R['jsd'].append(np.nan); R['jsd_flat'].append(np.nan); R['pear'].append(np.nan)
                        continue
                    R['jsd'].append(_jsd(prob[i], o))
                    R['jsd_flat'].append(_jsd(flat, o))
                    R['pear'].append(_pearson(prob[i], o))

    results = {
        'checkpoint': args.checkpoint, 'split': args.split,
        'test_chroms': sorted(chroms),
        'n_peaks': int(is_peak.sum()), 'n_nonpeaks': int((~is_peak).sum()),
        'heldout_loss': loss_sum / max(n_seen, 1),
    }
    for head in ('acc', 'nuc'):
        R = rows[head]
        pred_lc = np.concatenate(R['pred_lc']); obs_lc = np.concatenate(R['obs_lc'])
        pk = np.concatenate(R['is_peak'])
        jsd = np.array(R['jsd']); jsd_flat = np.array(R['jsd_flat']); pear = np.array(R['pear'])
        results[head] = {
            'counts_pearson_peaks': _pearson(pred_lc[pk], obs_lc[pk]),
            'counts_spearman_peaks': _spearman(pred_lc[pk], obs_lc[pk]),
            'profile_jsd_median': float(np.nanmedian(jsd)),
            'profile_jsd_flat_baseline': float(np.nanmedian(jsd_flat)),
            'profile_pearson_median': float(np.nanmedian(pear)),
        }

    # Tier 2: peak-vs-nonpeak AUROC from predicted accessibility log-count.
    acc_pred_lc_all = np.concatenate(rows['acc']['pred_lc'])
    results['peak_vs_nonpeak_auroc_acc'] = _auroc(acc_pred_lc_all, is_peak)

    # Tier 2: reverse-complement count consistency on a subsample.
    if args.revcomp_check:
        k = min(args.revcomp_check, len(ds))
        sel = rng.choice(len(ds), size=k, replace=False)
        a_fwd, a_rev, n_fwd, n_rev = [], [], [], []
        with torch.no_grad():
            for j in sel:
                seq = ds[j][0].unsqueeze(0).to(device)            # (1, L, 4)
                rc = torch.flip(seq, dims=[1, 2])                 # revcomp
                _, ac_f, _, nc_f = model(seq)
                _, ac_r, _, nc_r = model(rc)
                a_fwd.append(float(ac_f)); a_rev.append(float(ac_r))
                n_fwd.append(float(nc_f)); n_rev.append(float(nc_r))
        results['revcomp_count_pearson'] = {
            'acc': _pearson(a_fwd, a_rev), 'nuc': _pearson(n_fwd, n_rev),
            'acc_mean_abs_diff': float(np.mean(np.abs(np.array(a_fwd) - np.array(a_rev)))),
            'nuc_mean_abs_diff': float(np.mean(np.abs(np.array(n_fwd) - np.array(n_rev)))),
        }

    # ---- report ----
    print("\n================ EVAL ================")
    print(f"checkpoint : {args.checkpoint}")
    print(f"split={args.split}  chroms={results['test_chroms']}")
    print(f"peaks={results['n_peaks']}  nonpeaks={results['n_nonpeaks']}  heldout_loss={results['heldout_loss']:.2f}")
    for head in ('acc', 'nuc'):
        r = results[head]
        print(f"\n[{head}]")
        print(f"  counts Pearson (peaks)   : {r['counts_pearson_peaks']:.4f}")
        print(f"  counts Spearman (peaks)  : {r['counts_spearman_peaks']:.4f}")
        print(f"  profile JSD median       : {r['profile_jsd_median']:.4f}  (flat baseline {r['profile_jsd_flat_baseline']:.4f}; lower=better)")
        print(f"  profile Pearson median   : {r['profile_pearson_median']:.4f}")
    print(f"\npeak vs nonpeak AUROC (acc counts): {results['peak_vs_nonpeak_auroc_acc']:.4f}")
    if args.revcomp_check:
        rc = results['revcomp_count_pearson']
        print(f"revcomp count consistency: acc r={rc['acc']:.4f} (|d|={rc['acc_mean_abs_diff']:.3f}), "
              f"nuc r={rc['nuc']:.4f} (|d|={rc['nuc_mean_abs_diff']:.3f})")
    print("======================================")

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, 'w') as f:
            json.dump(results, f, indent=2)
        print("wrote", args.out_json)


if __name__ == '__main__':
    main()
