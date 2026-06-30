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

from chrombpnet.multitask import metrics as M
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


COMPONENT_KEYS = ('acc_profile_nll', 'acc_count_mse', 'nuc_profile_nll', 'nuc_count_mse')


def run_epoch(model, loader, optimizer, device, args):
    training = optimizer is not None
    model.train(training)
    total = 0.0
    n = 0
    comp_sum = {k: 0.0 for k in COMPONENT_KEYS}
    grad_accum = 0.0
    for batch in loader:
        seq, acc, acc_lc, nuc, nuc_lc = [x.to(device, non_blocking=True) for x in batch]
        bs = seq.shape[0]
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            outputs = model(seq)
            loss, comps = multitask_loss(outputs, (acc, acc_lc, nuc, nuc_lc),
                                         counts_loss_weight=args.counts_loss_weight,
                                         nucleosome_profile_weight=args.nucleosome_profile_weight,
                                         nucleosome_counts_weight=args.nucleosome_counts_weight,
                                         return_components=True)
            if training:
                loss.backward()
                grad_accum += float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9)) * bs
                optimizer.step()
        total += float(loss.detach()) * bs
        for k in COMPONENT_KEYS:
            comp_sum[k] += float(comps[k]) * bs
        n += bs
    n = max(n, 1)
    result = {'loss': total / n, 'n': n}
    result.update({k: comp_sum[k] / n for k in COMPONENT_KEYS})
    if training:
        result['grad_norm'] = grad_accum / n
    return result


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

    best = float('inf')
    best_epoch = 0
    stale = 0
    out = Path(args.output_prefix)
    out.parent.mkdir(parents=True, exist_ok=True)
    log_path = out.with_suffix('.log')

    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
        )

    try:
        with open(log_path, 'w') as log:
            log.write('epoch,train_loss,val_loss\n')
            for epoch in range(1, args.epochs + 1):
                train_ds.on_epoch_end()
                t0 = time.perf_counter()
                tr = run_epoch(model, train_loader, optimizer, device, args)
                t1 = time.perf_counter()
                va = run_epoch(model, valid_loader, None, device, args)
                t2 = time.perf_counter()
                train_loss, val_loss = tr['loss'], va['loss']
                print(f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
                log.write(f"{epoch},{train_loss:.8f},{val_loss:.8f}\n"); log.flush()
                is_best = val_loss < best
                if is_best:
                    best = val_loss
                    best_epoch = epoch
                    stale = 0
                    torch.save({'model_state_dict': model.state_dict(), 'args': vars(args), 'val_loss': best},
                               str(out) + '.pt')
                else:
                    stale += 1
                if wandb_run is not None:
                    ld = {'epoch': epoch, 'train/loss': train_loss, 'val/loss': val_loss,
                          'val/best_loss': best, 'val/best_epoch': best_epoch,
                          'lr': optimizer.param_groups[0]['lr'],
                          'train/grad_norm': tr.get('grad_norm', float('nan')),
                          'time/epoch_sec': t2 - t0,
                          'time/train_samples_per_sec': tr['n'] / max(t1 - t0, 1e-9)}
                    for k in COMPONENT_KEYS:
                        ld[f'train/{k}'] = tr[k]; ld[f'val/{k}'] = va[k]
                    figs = None
                    if args.eval_every and (epoch % args.eval_every == 0 or epoch == args.epochs):
                        try:
                            ld.update(M.heldout_metrics(model, valid_ds, ev_idx, ev_is_peak,
                                                        device, args.outputlen, multicell=False, prefix='val'))
                        except Exception as e:  # noqa: BLE001
                            print(f"[warn] held-out metrics failed at epoch {epoch}: {e}")
                        if args.log_examples:
                            try:
                                figs = M.example_profile_figures(model, valid_ds, ev_idx, device,
                                                                 args.outputlen, multicell=False, n=args.log_examples)
                            except Exception as e:  # noqa: BLE001
                                print(f"[warn] example plots failed at epoch {epoch}: {e}")
                    M.log_epoch_to_wandb(wandb_run, ld, examples=figs)
                    if figs:
                        import matplotlib.pyplot as plt
                        for f in figs:
                            plt.close(f)
                if not is_best and stale >= args.early_stop_patience:
                    break
        print('saved', str(out) + '.pt')
        if wandb_run is not None:
            wandb_run.summary['best_val_loss'] = best
            wandb_run.summary['best_epoch'] = best_epoch
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == '__main__':
    main()
