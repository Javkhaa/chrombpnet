#!/usr/bin/env python3
"""Train the PyTorch multi-task accessibility + nucleosome ChromBPNet model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

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


def run_epoch(model, loader, optimizer, device, args):
    training = optimizer is not None
    model.train(training)
    total = 0.0
    n = 0
    for batch in loader:
        seq, acc, acc_lc, nuc, nuc_lc = [x.to(device, non_blocking=True) for x in batch]
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            outputs = model(seq)
            loss = multitask_loss(outputs, (acc, acc_lc, nuc, nuc_lc),
                                  counts_loss_weight=args.counts_loss_weight,
                                  nucleosome_profile_weight=args.nucleosome_profile_weight,
                                  nucleosome_counts_weight=args.nucleosome_counts_weight)
            if training:
                loss.backward()
                optimizer.step()
        total += float(loss.detach()) * seq.shape[0]
        n += seq.shape[0]
    return total / max(n, 1)


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
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    fold = json.load(open(args.fold_json))
    train_chroms, valid_chroms = set(fold['train']), set(fold['valid'])
    print(f"train chroms={len(train_chroms)} valid chroms={len(valid_chroms)} device={device}")

    train_ds, train_loader = make_loader(args, train_chroms, revcomp=True, shuffle=True, seed=args.seed)
    valid_ds, valid_loader = make_loader(args, valid_chroms, revcomp=False, shuffle=False, seed=args.seed + 1)

    model = MultiTaskNucleosomeModel(args.inputlen, args.outputlen, args.filters, args.n_dil_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    best = float('inf')
    stale = 0
    out = Path(args.output_prefix)
    out.parent.mkdir(parents=True, exist_ok=True)
    log_path = out.with_suffix('.log')
    with open(log_path, 'w') as log:
        log.write('epoch,train_loss,val_loss\n')
        for epoch in range(1, args.epochs + 1):
            train_ds.on_epoch_end()
            train_loss = run_epoch(model, train_loader, optimizer, device, args)
            val_loss = run_epoch(model, valid_loader, None, device, args)
            print(f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
            log.write(f"{epoch},{train_loss:.8f},{val_loss:.8f}\n"); log.flush()
            if val_loss < best:
                best = val_loss
                stale = 0
                torch.save({'model_state_dict': model.state_dict(), 'args': vars(args), 'val_loss': best},
                           str(out) + '.pt')
            else:
                stale += 1
                if stale >= args.early_stop_patience:
                    break
    print('saved', str(out) + '.pt')


if __name__ == '__main__':
    main()
