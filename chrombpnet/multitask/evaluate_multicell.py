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
from chrombpnet.training.models.multitask_nucleosome_torch import MultiCellMultiTaskModel
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
    ap.add_argument('-o', '--out-json', default=None)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)

    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    a = ck['args']; cell_types = ck['cell_types']
    ct_index = {name: i for i, name in enumerate(cell_types)}
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
    per_cell = {}
    for row in manifest:
        name = row['cell_type']
        if name not in ct_index:
            print(f"  skip {name}: not in checkpoint heads"); continue
        ci = ct_index[name]
        pdf = _subsample(load_regions(row['peaks'], chroms), args.max_peaks, rng)
        ndf = _subsample(load_regions(row['nonpeaks'], chroms), args.max_nonpeaks, rng)
        if pdf is None or len(pdf) == 0:
            print(f"  skip {name}: no test peaks"); continue
        ds = MultiTaskRegionDataset(pdf, ndf, args.genome, row['acc_bw'], row['nuc_bw'],
                                    IL, OL, max_jitter=0, negative_sampling_ratio=1.0,
                                    add_revcomp=False, shuffle=False, seed=args.seed)
        is_peak = ds.regions['is_peak'].values.astype(bool)
        idx = np.arange(len(ds))
        m = M.heldout_metrics(model, _FixedCT(ds, ci), idx, is_peak, device, OL,
                              batch_size=args.batch_size, multicell=True, prefix='')
        per_cell[name] = {k.lstrip('/'): v for k, v in m.items()}
        print(f"  [{name[:42]:42s}] acc r={per_cell[name]['acc/counts_pearson']:.3f} "
              f"nuc r={per_cell[name]['nuc/counts_pearson']:.3f} "
              f"AUROC={per_cell[name]['peak_vs_nonpeak_auroc_acc']:.3f}")

    def _avg(key):
        vals = [v[key] for v in per_cell.values() if v[key] == v[key]]
        return float(np.mean(vals)) if vals else float('nan')

    means = {k: _avg(k) for k in ['acc/counts_pearson', 'acc/profile_jsd', 'acc/profile_pearson',
                                  'nuc/counts_pearson', 'nuc/profile_jsd', 'nuc/profile_pearson',
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
                feat = model.trunk(x)                       # (b, filters, L')
                pooled = feat.mean(dim=-1)                  # (b, filters)
                # predicted acc log-count for ALL cell types at once
                Pb = pooled @ model.acc_count_w.t() + model.acc_count_b  # (b, NC)
                P[list(sub), :] = Pb.cpu().numpy()

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
    print("\n[A] per-cell-type accuracy (means over cell types):")
    print(f"  acc: counts r={means['acc/counts_pearson']:.3f}  profile JSD={means['acc/profile_jsd']:.3f}  profile r={means['acc/profile_pearson']:.3f}")
    print(f"  nuc: counts r={means['nuc/counts_pearson']:.3f}  profile JSD={means['nuc/profile_jsd']:.3f}  profile r={means['nuc/profile_pearson']:.3f}")
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
