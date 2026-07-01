#!/usr/bin/env python3
"""Train the PyTorch multi-task accessibility + nucleosome ChromBPNet model."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from chrombpnet.multitask import train_loop
from chrombpnet.multitask.torch_data import MultiTaskRegionDataset
from chrombpnet.training.models.multitask_nucleosome_torch import (
    MultiTaskNucleosomeModel,
    multitask_loss,
)

NARROWPEAK_COLS = ['chr', 'start', 'end', 'name', 'score', 'strand',
                   'signal', 'pval', 'qval', 'summit']


def load_regions(path, chroms):
    if path is None:
        return None
    df = pd.read_csv(path, sep='\t', header=None, names=NARROWPEAK_COLS, usecols=range(10))
    return df[df['chr'].isin(chroms)].reset_index(drop=True)


def make_loader(args, chroms, revcomp, shuffle, seed):
    ds = MultiTaskRegionDataset(
        load_regions(args.peaks, chroms), load_regions(args.nonpeaks, chroms),
        args.genome, args.acc_bw, args.nuc_bw,
        args.inputlen, args.outputlen, args.max_jitter,
        negative_sampling_ratio=args.negative_sampling_ratio,
        add_revcomp=revcomp, shuffle=shuffle, seed=seed,
    )
    loader_kwargs = {
        'batch_size': args.batch_size,
        'shuffle': False,
        'num_workers': args.num_workers,
        'pin_memory': torch.cuda.is_available(),
    }
    if args.num_workers > 0:
        loader_kwargs.update(
            persistent_workers=True,
            prefetch_factor=args.prefetch_factor,
        )
    loader = DataLoader(ds, **loader_kwargs)
    return ds, loader




def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('-p', '--peaks', required=True)
    ap.add_argument('-n', '--nonpeaks', required=True)
    ap.add_argument('-g', '--genome', required=True)
    ap.add_argument('--acc-bw', required=True)
    ap.add_argument('--nuc-bw', required=True)
    ap.add_argument('-fl', '--fold-json', required=True)
    ap.add_argument('-o', '--output-prefix', required=True)
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
    # Experiment tracking (Weights & Biases). Off by default so smoke tests and
    # offline runs need no wandb install or credentials.
    ap.add_argument('--wandb', action='store_true', help='Log metrics to Weights & Biases')
    ap.add_argument('--wandb-entity', default='prima-mente')
    ap.add_argument('--wandb-project', default='jg_experiments')
    ap.add_argument('--wandb-run-name', default=None, help='Optional run name (defaults to wandb auto-name)')
    ap.add_argument('--eval-every', type=int, default=5, help='epochs between held-out metric logging (0=off)')
    ap.add_argument('--eval-subset', type=int, default=2000, help='valid regions used for held-out metrics')
    ap.add_argument('--log-examples', type=int, default=3, help='example predicted-vs-observed plots to log (0=off)')
    ap.add_argument('--val-every-steps', type=int, default=0,
                    help='validate/checkpoint/early-stop every N optimizer steps (0=once per epoch)')
    ap.add_argument('--val-max-batches', type=int, default=0,
                    help='cap validation to N batches per check (0=full valid set); use with --val-every-steps')
    ap.add_argument('--log-loss-every', type=int, default=0,
                    help='log running train loss to stdout+wandb every N steps (0=off), independent of validation')
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    fold = json.load(open(args.fold_json))
    train_chroms, valid_chroms = set(fold['train']), set(fold['valid'])
    print(f"train chroms={len(train_chroms)} valid chroms={len(valid_chroms)} device={device}")

    train_ds, train_loader = make_loader(args, train_chroms, revcomp=True, shuffle=True, seed=args.seed)
    valid_ds, valid_loader = make_loader(args, valid_chroms, revcomp=False, shuffle=False, seed=args.seed + 1)

    # Fixed valid subset for held-out quality metrics (deterministic across epochs).
    rng = np.random.default_rng(args.seed)
    n_valid = len(valid_ds)
    ev_idx = np.arange(n_valid)
    if args.eval_subset and n_valid > args.eval_subset:
        ev_idx = np.sort(rng.choice(n_valid, size=args.eval_subset, replace=False))
    ev_is_peak = valid_ds.regions['is_peak'].values[ev_idx].astype(bool)

    model = MultiTaskNucleosomeModel(args.inputlen, args.outputlen, args.filters, args.n_dil_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    out = Path(args.output_prefix)
    out.parent.mkdir(parents=True, exist_ok=True)
    log_path = out.with_suffix('.log')

    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(entity=args.wandb_entity, project=args.wandb_project,
                               name=args.wandb_run_name, config=vars(args))

    def save_best(val_loss):
        torch.save({'model_state_dict': model.state_dict(), 'args': vars(args), 'val_loss': val_loss},
                   str(out) + '.pt')

    try:
        best, best_marker = train_loop.fit(
            model, train_loader, valid_loader, optimizer, device, args,
            multicell=False, save_best=save_best, log_path=str(log_path),
            eval_ctx=(valid_ds, ev_idx, ev_is_peak), wandb_run=wandb_run,
            on_epoch_start=train_ds.on_epoch_end)
        print('saved', str(out) + '.pt')
        if wandb_run is not None:
            wandb_run.summary['best_val_loss'] = best
            wandb_run.summary['best_marker'] = best_marker
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == '__main__':
    main()
