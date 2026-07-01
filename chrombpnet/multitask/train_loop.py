"""Shared training loop for the multi-task trainers.

Supports both per-epoch validation (default) and **step-based** validation
(--val-every-steps N): validate / checkpoint / early-stop / log every N optimizer
steps instead of once per epoch. With N=0 the behaviour is identical to validating
at each epoch end.
"""
from __future__ import annotations

import contextlib
import time

import torch

from chrombpnet.multitask import metrics as M
from chrombpnet.training.models.multitask_nucleosome_torch import multitask_loss

COMPONENT_KEYS = ('acc_profile_nll', 'acc_count_mse', 'nuc_profile_nll', 'nuc_count_mse')


def _to_device(batch, device, multicell):
    if multicell:
        seq, acc, acc_lc, nuc, nuc_lc, ct = batch
        ct = ct.to(device, non_blocking=True)
    else:
        seq, acc, acc_lc, nuc, nuc_lc = batch
        ct = None
    seq = seq.to(device, non_blocking=True)
    acc = acc.to(device, non_blocking=True); acc_lc = acc_lc.to(device, non_blocking=True)
    nuc = nuc.to(device, non_blocking=True); nuc_lc = nuc_lc.to(device, non_blocking=True)
    return seq, (acc, acc_lc, nuc, nuc_lc), ct


def _forward(model, seq, ct):
    return model(seq, ct) if ct is not None else model(seq)


def _loss(outputs, targets, args):
    return multitask_loss(outputs, targets,
                          counts_loss_weight=args.counts_loss_weight,
                          nucleosome_profile_weight=args.nucleosome_profile_weight,
                          nucleosome_counts_weight=args.nucleosome_counts_weight,
                          return_components=True)


def _amp_ctx(args, device):
    """bf16 autocast on CUDA when --amp is set; no-op otherwise."""
    if getattr(args, 'amp', False) and str(device).startswith('cuda'):
        return torch.autocast('cuda', dtype=torch.bfloat16)
    return contextlib.nullcontext()


@torch.no_grad()
def validate(model, loader, device, args, multicell, max_batches=0):
    """Mean loss + components over the valid loader (optionally capped to max_batches)."""
    model.eval()
    total = 0.0; n = 0
    comp = {k: 0.0 for k in COMPONENT_KEYS}
    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        seq, targets, ct = _to_device(batch, device, multicell)
        with _amp_ctx(args, device):
            outputs = _forward(model, seq, ct)
            loss, comps = _loss(outputs, targets, args)
        bs = seq.shape[0]
        total += float(loss) * bs
        for k in COMPONENT_KEYS:
            comp[k] += float(comps[k]) * bs
        n += bs
    n = max(n, 1)
    out = {'loss': total / n}
    out.update({k: comp[k] / n for k in COMPONENT_KEYS})
    return out


def fit(model, train_loader, valid_loader, optimizer, device, args, *,
        multicell, save_best, log_path, eval_ctx=None, wandb_run=None,
        on_epoch_start=None):
    """Run training. `save_best(val_loss)` persists the best checkpoint; `eval_ctx` is
    (dataset, indices, is_peak) for held-out metric logging (or None). Returns
    (best_val_loss, best_marker) where best_marker is the step (step-based) or epoch."""
    best = float('inf'); best_marker = 0; stale = 0
    global_step = 0
    val_every = getattr(args, 'val_every_steps', 0) or 0
    val_max_batches = getattr(args, 'val_max_batches', 0) or 0
    log_loss_every = getattr(args, 'log_loss_every', 0) or 0

    win = {'loss': 0.0, 'n': 0, 'grad': 0.0}
    win.update({k: 0.0 for k in COMPONENT_KEYS})
    state = {'t0': time.perf_counter()}

    # separate fine-grained accumulator for frequent train-loss logging
    lw = {'loss': 0.0, 'n': 0, 'grad': 0.0}
    lw.update({k: 0.0 for k in COMPONENT_KEYS})
    lw_state = {'t0': time.perf_counter()}

    def reset_win():
        win['loss'] = win['grad'] = 0.0; win['n'] = 0
        for k in COMPONENT_KEYS:
            win[k] = 0.0
        state['t0'] = time.perf_counter()

    def log_trainstep():
        n = max(lw['n'], 1)
        tl = lw['loss'] / n
        dt = max(time.perf_counter() - lw_state['t0'], 1e-9)
        print(f"  step={global_step} train_loss={tl:.4f} ({lw['n'] / dt:.0f} samp/s)", flush=True)
        if wandb_run is not None:
            d = {'step': global_step, 'trainstep/loss': tl,
                 'trainstep/grad_norm': lw['grad'] / n,
                 'lr': optimizer.param_groups[0]['lr']}
            for k in COMPONENT_KEYS:
                d[f'trainstep/{k}'] = lw[k] / n
            try:
                wandb_run.log(d, step=global_step)
            except Exception as e:  # noqa: BLE001
                print(f"[warn] trainstep log failed at step {global_step}: {e}")
        for k in COMPONENT_KEYS:
            lw[k] = 0.0
        lw['loss'] = lw['grad'] = 0.0; lw['n'] = 0
        lw_state['t0'] = time.perf_counter()

    log = open(log_path, 'w')
    log.write('epoch,step,train_loss,val_loss\n')

    def do_validation(epoch):
        nonlocal best, best_marker, stale
        va = validate(model, valid_loader, device, args, multicell, val_max_batches)
        n = max(win['n'], 1)
        train_loss, val_loss = win['loss'] / n, va['loss']
        print(f"epoch={epoch} step={global_step} train_loss={train_loss:.6f} val_loss={val_loss:.6f}", flush=True)
        log.write(f"{epoch},{global_step},{train_loss:.8f},{val_loss:.8f}\n"); log.flush()
        is_best = val_loss < best
        if is_best:
            best = val_loss; best_marker = (global_step if val_every else epoch); stale = 0
            save_best(best)
        else:
            stale += 1
        if wandb_run is not None:
            dt = max(time.perf_counter() - state['t0'], 1e-9)
            ld = {'epoch': epoch, 'step': global_step,
                  'train/loss': train_loss, 'val/loss': val_loss, 'val/best_loss': best,
                  'lr': optimizer.param_groups[0]['lr'],
                  'train/grad_norm': win['grad'] / n,
                  'time/window_sec': dt, 'time/train_samples_per_sec': win['n'] / dt}
            for k in COMPONENT_KEYS:
                ld[f'train/{k}'] = win[k] / n; ld[f'val/{k}'] = va[k]
            figs = None
            if eval_ctx is not None:
                dataset, ev_idx, ev_is_peak = eval_ctx
                try:
                    ld.update(M.heldout_metrics(model, dataset, ev_idx, ev_is_peak, device,
                                                args.outputlen, multicell=multicell, prefix='val'))
                except Exception as e:  # noqa: BLE001
                    print(f"[warn] held-out metrics failed: {e}")
                if getattr(args, 'log_examples', 0):
                    try:
                        figs = M.example_profile_figures(model, dataset, ev_idx, device,
                                                         args.outputlen, multicell=multicell,
                                                         n=args.log_examples)
                    except Exception as e:  # noqa: BLE001
                        print(f"[warn] example plots failed: {e}")
            M.log_epoch_to_wandb(wandb_run, ld, examples=figs)
            if figs:
                import matplotlib.pyplot as plt
                for f in figs:
                    plt.close(f)
        model.train(True)
        return (not is_best) and stale >= args.early_stop_patience

    stop = False
    try:
        for epoch in range(1, args.epochs + 1):
            if on_epoch_start:
                on_epoch_start()
            model.train(True)
            for batch in train_loader:
                seq, targets, ct = _to_device(batch, device, multicell)
                optimizer.zero_grad(set_to_none=True)
                with _amp_ctx(args, device):
                    outputs = _forward(model, seq, ct)
                    loss, comps = _loss(outputs, targets, args)
                loss.backward()
                gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9))
                optimizer.step()
                global_step += 1
                bs = seq.shape[0]
                lval = float(loss.detach())
                cvals = {k: float(comps[k]) for k in COMPONENT_KEYS}
                win['loss'] += lval * bs; win['n'] += bs; win['grad'] += gn * bs
                lw['loss'] += lval * bs; lw['n'] += bs; lw['grad'] += gn * bs
                for k in COMPONENT_KEYS:
                    win[k] += cvals[k] * bs
                    lw[k] += cvals[k] * bs
                if log_loss_every and global_step % log_loss_every == 0:
                    log_trainstep()
                if val_every and global_step % val_every == 0:
                    stop = do_validation(epoch)
                    reset_win()
                    if stop:
                        break
            if stop:
                break
            if not val_every:
                stop = do_validation(epoch)
                reset_win()
                if stop:
                    break
    finally:
        log.close()
    return best, best_marker
