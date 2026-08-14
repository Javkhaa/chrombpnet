#!/usr/bin/env python3
"""Held-out evaluation for the multi-cell-type model (MultiCellMultiTaskModel).

Two levels, both on the fold's TEST chromosomes (never seen in training):

  (A) Per-cell-type accuracy: for each cell type, route inputs to ITS head and
      score counts Pearson/Spearman, profile JSD/Pearson, peak-vs-nonpeak AUROC
      (accessibility + nucleosome). Reports a per-cell table + means.

  (B) Cell-type SPECIFICITY: over a common set of regions, build observed vs
      predicted log-count matrices [regions x cell_types] and measure
      (i) overall Pearson and (ii) the CENTERED (per-region mean-subtracted)
      Pearson. The centered number isolates cell-type-specific variation from
      generic accessibility -- it is the metric that justifies a multi-cell model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from chrombpnet.multitask import metrics as M
from chrombpnet.multitask.torch_data import MultiTaskRegionDataset, _fetch_seq, _fetch_bw
from chrombpnet.multitask.train_multitask_torch import load_regions
from chrombpnet.multitask.train_multicell_torch import parse_manifest
from chrombpnet.training.models.multitask_nucleosome_torch import (
    MultiCellMultiTaskModel, ConditionedMultiCellModel)
import os
import pyBigWig
import pyfaidx


class _FixedCT(Dataset):
    """Wrap a single-cell dataset so every item carries a fixed cell-type index."""
    def __init__(self, ds, ci):
        self.ds = ds; self.ci = ci

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        return (*self.ds[i], torch.tensor(self.ci, dtype=torch.long))


def _subsample(df, n, rng):
    if df is None or len(df) == 0 or not n or len(df) <= n:
        return df
    return df.iloc[rng.choice(len(df), size=n, replace=False)].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('-c', '--checkpoint', required=True)
    ap.add_argument('-m', '--manifest', required=True)
    ap.add_argument('--manifest-root', default=None)
    ap.add_argument('-g', '--genome', required=True)
    ap.add_argument('-fl', '--fold-json', required=True)
    ap.add_argument('--split', default='test', choices=['test', 'valid', 'train'])
    ap.add_argument('--max-peaks', type=int, default=3000, help='per-cell peaks for part (A)')
    ap.add_argument('--max-nonpeaks', type=int, default=3000, help='per-cell nonpeaks for part (A)')
    ap.add_argument('--spec-regions', type=int, default=2000, help='regions for the specificity matrix (B)')
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--num-workers', type=int, default=8)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--smooth-sigma', type=float, default=20.0,
                    help='Gaussian sigma (bp) for smoothed-profile metrics + reproducibility ceiling')
    ap.add_argument('-o', '--out-json', default=None)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)

    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    a = ck['args']; cell_types = ck['cell_types']
    ct_index = {name: i for i, name in enumerate(cell_types)}
    if a.get('conditioned'):
        model = ConditionedMultiCellModel(len(cell_types), a['inputlen'], a['outputlen'],
                                          a['filters'], a['n_dil_layers'], a.get('embed_dim', 32),
                                          cond_mode=a.get('cond_mode', 'additive'))
    else:
        model = MultiCellMultiTaskModel(len(cell_types), a['inputlen'], a['outputlen'],
                                        a['filters'], a['n_dil_layers'])
    model.load_state_dict(ck['model_state_dict'])
    model.to(device).eval()
    IL, OL = a['inputlen'], a['outputlen']

    root = args.manifest_root or os.path.dirname(os.path.abspath(args.manifest))
    manifest = parse_manifest(args.manifest, root)
    fold = json.load(open(args.fold_json))
    chroms = set(fold[args.split])
    print(f"checkpoint cell types: {len(cell_types)} | split={args.split} chroms={sorted(chroms)}")

    # ---------- (A) per-cell-type accuracy ----------
    # One DataLoader over ALL cells (parallel IO, GPU inference) -> collect per-example
    # scalars -> group by cell type. Far faster than a single-threaded loop per cell.
    from torch.utils.data import ConcatDataset
    fixed, is_peak_parts, name_by_ci = [], [], {}
    for row in manifest:
        name = row['cell_type']
        if name not in ct_index:
            continue
        ci = ct_index[name]
        pdf = _subsample(load_regions(row['peaks'], chroms), args.max_peaks, rng)
        ndf = _subsample(load_regions(row['nonpeaks'], chroms), args.max_nonpeaks, rng)
        if pdf is None or len(pdf) == 0:
            continue
        ds = MultiTaskRegionDataset(pdf, ndf, args.genome, row['acc_bw'], row['nuc_bw'],
                                    IL, OL, max_jitter=0, negative_sampling_ratio=1.0,
                                    add_revcomp=False, shuffle=False, seed=args.seed)
        fixed.append(_FixedCT(ds, ci))
        is_peak_parts.append(ds.regions['is_peak'].values.astype(bool))
        name_by_ci[ci] = name
    cat = ConcatDataset(fixed)
    is_peak = np.concatenate(is_peak_parts)
    loader = DataLoader(cat, batch_size=args.batch_size, num_workers=args.num_workers,
                        shuffle=False, pin_memory=(device.type == 'cuda'))
    print(f"scoring {len(cat)} regions across {len(fixed)} cell types...")
    ct_arr = []
    from scipy.ndimage import gaussian_filter1d
    SIG = args.smooth_sigma
    thin = np.random.default_rng(args.seed)
    acc = {'plc': [], 'olc': [], 'jsd': [], 'pear': [], 'smpear': [], 'ceil': []}
    nuc = {'plc': [], 'olc': [], 'jsd': [], 'pear': [], 'smpear': [], 'ceil': []}
    ptr = 0
    with torch.no_grad():
        for batch in loader:
            seq, a_obs, _, n_obs, _, ct = batch
            bs = seq.shape[0]
            out = model(seq.to(device), ct.to(device))
            ct_arr.append(ct.numpy())
            bpk = is_peak[ptr:ptr + bs]; ptr += bs
            for store, prof, cnt, obs in ((acc, out[0], out[1], a_obs), (nuc, out[2], out[3], n_obs)):
                prob = torch.softmax(prof, -1).cpu().numpy()
                store['plc'].append(cnt.squeeze(-1).cpu().numpy())
                o = obs.numpy(); store['olc'].append(np.log1p(o.sum(1)))
                for i in range(bs):
                    if bpk[i] and o[i].sum() > 0:
                        store['jsd'].append(M.jsd(prob[i], o[i])); store['pear'].append(M.pearson(prob[i], o[i]))
                        # smoothed-occupancy profile metric + split-half reproducibility ceiling
                        os_ = gaussian_filter1d(o[i], SIG)
                        store['smpear'].append(M.pearson(gaussian_filter1d(prob[i], SIG), os_))
                        oi = o[i].astype(np.int64); A = thin.binomial(oi, 0.5)
                        store['ceil'].append(M.pearson(gaussian_filter1d(A.astype(float), SIG),
                                                       gaussian_filter1d((oi - A).astype(float), SIG)))
                    else:
                        for k in ('jsd', 'pear', 'smpear', 'ceil'):
                            store[k].append(np.nan)
    ct_arr = np.concatenate(ct_arr)
    for d in (acc, nuc):
        for k in ('plc', 'olc'):
            d[k] = np.concatenate(d[k])
        for k in ('jsd', 'pear', 'smpear', 'ceil'):
            d[k] = np.array(d[k])
    per_cell = {}
    for ci, name in sorted(name_by_ci.items()):
        m = ct_arr == ci
        pk = m & is_peak
        per_cell[name] = {
            'acc/counts_pearson': M.pearson(acc['plc'][pk], acc['olc'][pk]),
            'acc/counts_spearman': M.spearman(acc['plc'][pk], acc['olc'][pk]),
            'acc/profile_jsd': float(np.nanmedian(acc['jsd'][m])),
            'acc/profile_pearson': float(np.nanmedian(acc['pear'][m])),
            'nuc/counts_pearson': M.pearson(nuc['plc'][pk], nuc['olc'][pk]),
            'nuc/counts_spearman': M.spearman(nuc['plc'][pk], nuc['olc'][pk]),
            'nuc/profile_jsd': float(np.nanmedian(nuc['jsd'][m])),
            'nuc/profile_pearson': float(np.nanmedian(nuc['pear'][m])),
            'acc/profile_pearson_smooth': float(np.nanmedian(acc['smpear'][m])),
            'acc/profile_ceiling_smooth': float(np.nanmedian(acc['ceil'][m])),
            'nuc/profile_pearson_smooth': float(np.nanmedian(nuc['smpear'][m])),
            'nuc/profile_ceiling_smooth': float(np.nanmedian(nuc['ceil'][m])),
            'peak_vs_nonpeak_auroc_acc': M.auroc(acc['plc'][m], is_peak[m]),
        }

    def _avg(key):
        vals = [v[key] for v in per_cell.values() if v[key] == v[key]]
        return float(np.mean(vals)) if vals else float('nan')

    means = {k: _avg(k) for k in ['acc/counts_pearson', 'acc/profile_jsd', 'acc/profile_pearson',
                                  'acc/profile_pearson_smooth', 'acc/profile_ceiling_smooth',
                                  'nuc/counts_pearson', 'nuc/profile_jsd', 'nuc/profile_pearson',
                                  'nuc/profile_pearson_smooth', 'nuc/profile_ceiling_smooth',
                                  'peak_vs_nonpeak_auroc_acc']}

    # ---------- (B) cell-type specificity ----------
    # Common regions = pooled peaks across cells (test chroms), deduped, sampled.
    pooled = []
    for row in manifest:
        if row['cell_type'] not in ct_index:
            continue
        df = load_regions(row['peaks'], chroms)
        if df is None or len(df) == 0:
            continue
        c = (df['start'] + df['summit']).astype(int)
        pooled.append(pd.DataFrame({'chr': df['chr'], 'center': c}))
    spec = {}
    if pooled:
        pool = pd.concat(pooled, ignore_index=True)
        pool['key'] = pool['chr'] + ':' + (pool['center'] // 200 * 200).astype(str)
        pool = pool.drop_duplicates('key').reset_index(drop=True)
        if len(pool) > args.spec_regions:
            pool = pool.iloc[np.sort(rng.choice(len(pool), args.spec_regions, replace=False))].reset_index(drop=True)
        N = len(pool)
        NC = len(cell_types)
        genome = pyfaidx.Fasta(args.genome)

        # observed log-counts O[N x NC]
        acc_bw = {row['cell_type']: row['acc_bw'] for row in manifest}
        O = np.full((N, NC), np.nan, np.float32)
        for name, ci in ct_index.items():
            if name not in acc_bw:
                continue
            bw = pyBigWig.open(acc_bw[name])
            for r in range(N):
                v = _fetch_bw(bw, pool['chr'][r], int(pool['center'][r]), OL)
                O[r, ci] = np.log1p(v.sum())
            bw.close()

        # predicted log-counts P[N x NC] via trunk-once + all count heads
        P = np.zeros((N, NC), np.float32)
        with torch.no_grad():
            for s in range(0, N, args.batch_size):
                sub = range(s, min(s + args.batch_size, N))
                seq = np.stack([_fetch_seq(genome, pool['chr'][r], int(pool['center'][r]), IL) for r in sub])
                x = torch.from_numpy(seq).to(device)
                bs = x.shape[0]
                # predicted acc log-count for each cell type via the standard forward
                # (model-agnostic: works for vectorized-head and conditioned models)
                for ci in range(NC):
                    ct = torch.full((bs,), ci, dtype=torch.long, device=device)
                    _, ac, _, _ = model(x, ct)
                    P[list(sub), ci] = ac.squeeze(-1).cpu().numpy()

        valid = ~np.isnan(O).any(axis=1)
        O, P = O[valid], P[valid]
        Oc = O - O.mean(axis=1, keepdims=True)
        Pc = P - P.mean(axis=1, keepdims=True)
        # per-region across-cell correlation (does the model rank cell types right?)
        per_region = [M.pearson(P[r], O[r]) for r in range(len(O))]
        spec = {
            'n_regions': int(len(O)),
            'overall_pearson': M.pearson(P.ravel(), O.ravel()),
            'celltype_specific_pearson': M.pearson(Pc.ravel(), Oc.ravel()),
            'per_region_pearson_median': float(np.nanmedian(per_region)),
        }

    # ---------- report ----------
    print("\n================ MULTI-CELL EVAL ================")
    print(f"checkpoint: {args.checkpoint}")
    print(f"cell types scored: {len(per_cell)} | split={args.split}")
    def _dist(key):
        v = np.array([c[key] for c in per_cell.values() if c[key] == c[key]])
        return (np.min(v), np.median(v), np.max(v)) if len(v) else (float('nan'),) * 3
    aq = _dist('acc/counts_pearson'); nq = _dist('nuc/counts_pearson'); auq = _dist('peak_vs_nonpeak_auroc_acc')
    print("\n[A] per-cell-type accuracy (means over cell types):")
    print(f"  acc counts r  min/med/max = {aq[0]:.3f} / {aq[1]:.3f} / {aq[2]:.3f}")
    print(f"  nuc counts r  min/med/max = {nq[0]:.3f} / {nq[1]:.3f} / {nq[2]:.3f}")
    print(f"  acc AUROC     min/med/max = {auq[0]:.3f} / {auq[1]:.3f} / {auq[2]:.3f}")
    print(f"  acc: counts r={means['acc/counts_pearson']:.3f}  profile r(raw)={means['acc/profile_pearson']:.3f}  "
          f"profile r(smooth)={means['acc/profile_pearson_smooth']:.3f} / ceiling {means['acc/profile_ceiling_smooth']:.3f}")
    print(f"  nuc: counts r={means['nuc/counts_pearson']:.3f}  profile r(raw)={means['nuc/profile_pearson']:.3f}  "
          f"profile r(smooth)={means['nuc/profile_pearson_smooth']:.3f} / ceiling {means['nuc/profile_ceiling_smooth']:.3f}")
    print(f"       (smoothed sigma={args.smooth_sigma:.0f}bp; ceiling = split-half reproducibility, the achievable max)")
    print(f"  peak-vs-nonpeak AUROC (acc): {means['peak_vs_nonpeak_auroc_acc']:.3f}")
    if spec:
        print("\n[B] cell-type specificity (accessibility log-counts):")
        print(f"  regions: {spec['n_regions']}")
        print(f"  overall Pearson (pred vs obs)         : {spec['overall_pearson']:.3f}")
        print(f"  CELL-TYPE-SPECIFIC Pearson (centered) : {spec['celltype_specific_pearson']:.3f}  <- key multi-cell metric")
        print(f"  per-region across-cell Pearson median : {spec['per_region_pearson_median']:.3f}")
    print("=================================================")

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        json.dump({'means': means, 'per_cell': per_cell, 'specificity': spec,
                   'checkpoint': args.checkpoint, 'split': args.split}, open(args.out_json, 'w'), indent=2)
        print("wrote", args.out_json)


if __name__ == '__main__':
    main()
