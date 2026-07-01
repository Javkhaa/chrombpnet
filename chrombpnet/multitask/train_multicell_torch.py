#!/usr/bin/env python3
"""Train the shared-trunk, per-cell-type multi-task ChromBPNet nucleosome model.

Scales the single-cell-type trainer to many cell types: one shared dilated-conv
trunk feeds a dedicated (accessibility, nucleosome) head pair per cell type. All
cell types share ONE chromosome fold (train/valid/test) so a held-out chromosome
is held out across every cell type — this is what prevents cross-cell-type label
leakage through the genome.

Cell types and their input files are listed in a tab-separated manifest:

    # cell_type   peaks                nonpeaks                acc_bw                       nuc_bw
    GM12878       GM12878/peaks.narrowPeak  GM12878/nonpeaks.narrowPeak  GM12878.acc.bw  GM12878.nuc.bw
    Kanemaru2023-cardiomyocyte  ...

Lines beginning with '#' and blank lines are ignored. Paths may be absolute or
relative to --manifest-root (default: the manifest's directory).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from chrombpnet.multitask import train_loop
from chrombpnet.multitask.torch_data import MultiTaskRegionDataset
from chrombpnet.multitask.train_multitask_torch import load_regions
from chrombpnet.training.models.multitask_nucleosome_torch import (
    MultiCellMultiTaskModel, ConditionedMultiCellModel, multitask_loss,
)


def parse_manifest(path, root):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('\t')
            if len(parts) != 5:
                raise ValueError(f"manifest line needs 5 tab-separated fields, got {len(parts)}: {line}")
            ct, peaks, nonpeaks, acc_bw, nuc_bw = parts
            resolve = lambda p: p if os.path.isabs(p) else os.path.join(root, p)
            rows.append({
                'cell_type': ct,
                'peaks': resolve(peaks), 'nonpeaks': resolve(nonpeaks),
                'acc_bw': resolve(acc_bw), 'nuc_bw': resolve(nuc_bw),
            })
    if not rows:
        raise ValueError(f"no cell types found in manifest {path}")
    return rows


def _drop_blacklist(df, blacklist_df, inputlen):
    """Remove regions whose input window overlaps any blacklist interval."""
    if df is None or len(df) == 0 or blacklist_df is None:
        return df
    keep = np.ones(len(df), dtype=bool)
    half = inputlen // 2
    for chrom, sub in df.groupby('chr'):
        bl = blacklist_df[blacklist_df['chr'] == chrom]
        if bl.empty:
            continue
        starts = bl['start'].values
        ends = bl['end'].values
        for i, r in sub.iterrows():
            c = int(r['start'] + r['summit'])
            ws, we = c - half, c + half
            if np.any((starts < we) & (ends > ws)):
                keep[i] = False
    return df[keep].reset_index(drop=True)


class TaggedConcat(Dataset):
    """Concatenate per-cell-type datasets, tagging each item with its cell-type index."""

    def __init__(self, datasets):
        self.datasets = datasets
        self._recompute()

    def _recompute(self):
        self.cum = np.cumsum([len(d) for d in self.datasets])

    def on_epoch_end(self):
        for d in self.datasets:
            d.on_epoch_end()
        self._recompute()

    def __len__(self):
        return int(self.cum[-1]) if len(self.cum) else 0

    def __getitem__(self, i):
        ct = int(np.searchsorted(self.cum, i, side='right'))
        prev = int(self.cum[ct - 1]) if ct > 0 else 0
        item = self.datasets[ct][i - prev]
        return (*item, torch.tensor(ct, dtype=torch.long))


def build_concat(manifest, chroms, args, revcomp, shuffle, seed, blacklist_df):
    datasets = []
    for k, m in enumerate(manifest):
        peaks_df = _drop_blacklist(load_regions(m['peaks'], chroms), blacklist_df, args.inputlen)
        nonpeaks_df = _drop_blacklist(load_regions(m['nonpeaks'], chroms), blacklist_df, args.inputlen)
        ds = MultiTaskRegionDataset(
            peaks_df, nonpeaks_df, args.genome, m['acc_bw'], m['nuc_bw'],
            args.inputlen, args.outputlen, args.max_jitter if revcomp else 0,
            negative_sampling_ratio=args.negative_sampling_ratio,
            add_revcomp=revcomp, shuffle=shuffle, seed=seed + k,
        )
        datasets.append(ds)
    return TaggedConcat(datasets)




def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('-m', '--manifest', required=True, help='TSV: cell_type, peaks, nonpeaks, acc_bw, nuc_bw')
    ap.add_argument('--manifest-root', default=None, help='base dir for relative paths (default: manifest dir)')
    ap.add_argument('-g', '--genome', required=True)
    ap.add_argument('-fl', '--fold-json', required=True)
    ap.add_argument('-o', '--output-prefix', required=True)
    ap.add_argument('--blacklist', default=None, help='optional BED of regions to drop')
    ap.add_argument('--inputlen', type=int, default=2114)
    ap.add_argument('--outputlen', type=int, default=1000)
    ap.add_argument('--max-jitter', type=int, default=500)
    ap.add_argument('--filters', type=int, default=512)
    ap.add_argument('--n-dil-layers', type=int, default=8)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--learning-rate', type=float, default=1e-3)
    ap.add_argument('--counts-loss-weight', type=float, default=1.0)
    ap.add_argument('--nucleosome-profile-weight', type=float, default=1.0)
    ap.add_argument('--nucleosome-counts-weight', type=float, default=1.0)
    ap.add_argument('--negative-sampling-ratio', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--early-stop-patience', type=int, default=5)
    ap.add_argument('--num-workers', type=int, default=16)
    ap.add_argument('--prefetch-factor', type=int, default=2)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--wandb', action='store_true')
    ap.add_argument('--wandb-entity', default='prima-mente')
    ap.add_argument('--wandb-project', default='jg_experiments')
    ap.add_argument('--wandb-run-name', default=None)
    ap.add_argument('--eval-every', type=int, default=5, help='epochs between held-out metric logging (0=off)')
    ap.add_argument('--eval-subset', type=int, default=2000, help='valid regions used for held-out metrics')
    ap.add_argument('--log-examples', type=int, default=3, help='example predicted-vs-observed plots to log (0=off)')
    ap.add_argument('--val-every-steps', type=int, default=0,
                    help='validate/checkpoint/early-stop every N optimizer steps (0=once per epoch)')
    ap.add_argument('--val-max-batches', type=int, default=0,
                    help='cap validation to N batches per check (0=full valid set); use with --val-every-steps')
    ap.add_argument('--log-loss-every', type=int, default=0,
                    help='log running train loss to stdout+wandb every N steps (0=off), independent of validation')
    ap.add_argument('--amp', action='store_true', help='bf16 mixed-precision autocast on CUDA (H100 tensor cores)')
    ap.add_argument('--compile', action='store_true', help='torch.compile the model')
    ap.add_argument('--conditioned', action='store_true',
                    help='cell-type-conditioned trunk (FiLM on embedding) + shared heads')
    ap.add_argument('--embed-dim', type=int, default=32, help='cell-type embedding dim (--conditioned)')
    ap.add_argument('--grad-clip', type=float, default=0.0, help='max grad norm (0=off); needed for FiLM stability')
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True  # stable-ish shapes -> faster conv algos
    device = torch.device(args.device)

    root = args.manifest_root or os.path.dirname(os.path.abspath(args.manifest))
    manifest = parse_manifest(args.manifest, root)
    cell_types = [m['cell_type'] for m in manifest]

    fold = json.load(open(args.fold_json))
    train_chroms, valid_chroms = set(fold['train']), set(fold['valid'])
    print(f"cell_types={len(cell_types)}: {cell_types}")
    print(f"train chroms={len(train_chroms)} valid chroms={len(valid_chroms)} device={device}")

    blacklist_df = None
    if args.blacklist:
        blacklist_df = pd.read_csv(args.blacklist, sep='\t', header=None,
                                   usecols=[0, 1, 2], names=['chr', 'start', 'end'])

    train_cat = build_concat(manifest, train_chroms, args, revcomp=True, shuffle=True,
                             seed=args.seed, blacklist_df=blacklist_df)
    valid_cat = build_concat(manifest, valid_chroms, args, revcomp=False, shuffle=False,
                             seed=args.seed + 1000, blacklist_df=blacklist_df)
    print(f"train regions={len(train_cat)} valid regions={len(valid_cat)}")

    # Fixed valid subset for held-out quality metrics (is_peak aligned to concat order).
    rng = np.random.default_rng(args.seed)
    valid_is_peak = np.concatenate([d.regions['is_peak'].values for d in valid_cat.datasets]).astype(bool)
    n_valid = len(valid_cat)
    ev_idx = np.arange(n_valid)
    if args.eval_subset and n_valid > args.eval_subset:
        ev_idx = np.sort(rng.choice(n_valid, size=args.eval_subset, replace=False))
    ev_is_peak = valid_is_peak[ev_idx]

    def make_loader(cat, shuffle):
        kw = {'batch_size': args.batch_size, 'shuffle': shuffle,
              'num_workers': args.num_workers, 'pin_memory': torch.cuda.is_available()}
        if args.num_workers > 0:
            kw.update(persistent_workers=True, prefetch_factor=args.prefetch_factor)
        return DataLoader(cat, **kw)

    train_loader = make_loader(train_cat, True)
    valid_loader = make_loader(valid_cat, False)

    if args.conditioned:
        raw_model = ConditionedMultiCellModel(len(cell_types), args.inputlen, args.outputlen,
                                              args.filters, args.n_dil_layers, args.embed_dim).to(device)
    else:
        raw_model = MultiCellMultiTaskModel(len(cell_types), args.inputlen, args.outputlen,
                                            args.filters, args.n_dil_layers).to(device)
    optimizer = torch.optim.Adam(raw_model.parameters(), lr=args.learning_rate)
    if args.compile:
        if args.conditioned:
            # Conditioned forward has no data-dependent control flow (FiLM is vectorized),
            # so the whole model compiles cleanly.
            model = torch.compile(raw_model)
        else:
            # Vectorized-head model: compile only the static trunk (the heads' grouped
            # conv uses a data-dependent groups=B that we keep eager).
            raw_model.trunk = torch.compile(raw_model.trunk)
            model = raw_model
    else:
        model = raw_model

    out = Path(args.output_prefix)
    out.parent.mkdir(parents=True, exist_ok=True)

    wandb_run = None
    if args.wandb:
        import wandb
        cfg = vars(args).copy(); cfg['cell_types'] = cell_types
        wandb_run = wandb.init(entity=args.wandb_entity, project=args.wandb_project,
                               name=args.wandb_run_name, config=cfg)

    def save_best(val_loss):
        torch.save({'model_state_dict': raw_model.state_dict(), 'args': vars(args),
                    'cell_types': cell_types, 'val_loss': val_loss}, str(out) + '.pt')

    try:
        best, best_marker = train_loop.fit(
            model, train_loader, valid_loader, optimizer, device, args,
            multicell=True, save_best=save_best, log_path=str(out.with_suffix('.log')),
            eval_ctx=(valid_cat, ev_idx, ev_is_peak), wandb_run=wandb_run,
            on_epoch_start=train_cat.on_epoch_end)
        print('saved', str(out) + '.pt')
        if wandb_run is not None:
            wandb_run.summary['best_val_loss'] = best
            wandb_run.summary['best_marker'] = best_marker
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == '__main__':
    main()
