#!/usr/bin/env python3
"""Train the multi-task (accessibility + nucleosome) ChromBPNet model.

Reuses chrombpnet's region/peak conventions: peaks & nonpeaks are 10-col
narrowPeak (summit in col 10, region centered at start+summit). Two label
bigwigs are supplied: the accessibility cut-site track (built by chrombpnet's
own preprocessing) and the nucleosome dyad track (build_label_tracks.py).

Chromosome splits come from a chrombpnet folds JSON ({"train":[...], "valid":[...],
"test":[...]}). This is the bias-FREE multi-task model; for head-1 bias
correction, train the stock chrombpnet head-1 first and graft its frozen bias
model onto the accessibility head (future work — see model file docstring).
"""
import argparse, json, types
import numpy as np
import pandas as pd
import tensorflow as tf

from chrombpnet.training.models import multitask_nucleosome_model as mt
from chrombpnet.multitask.data_generator import MultiTaskBatchGenerator

NARROWPEAK_COLS = ['chr', 'start', 'end', 'name', 'score', 'strand',
                   'signal', 'pval', 'qval', 'summit']


def load_regions(path, chroms):
    if path is None:
        return None
    df = pd.read_csv(path, sep='\t', header=None, names=NARROWPEAK_COLS, usecols=range(10))
    df = df[df['chr'].isin(chroms)].reset_index(drop=True)
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('-p', '--peaks', required=True)
    ap.add_argument('-n', '--nonpeaks', required=True)
    ap.add_argument('-g', '--genome', required=True)
    ap.add_argument('--acc-bw', required=True, help='accessibility cut-site bigwig (head 1)')
    ap.add_argument('--nuc-bw', required=True, help='nucleosome dyad bigwig (head 2)')
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
    args = ap.parse_args()

    fold = json.load(open(args.fold_json))
    train_chroms, valid_chroms = set(fold['train']), set(fold['valid'])
    print(f"train chroms={len(train_chroms)} valid chroms={len(valid_chroms)}")

    def make_gen(chroms, revcomp, shuffle):
        return MultiTaskBatchGenerator(
            load_regions(args.peaks, chroms), load_regions(args.nonpeaks, chroms),
            args.genome, args.acc_bw, args.nuc_bw,
            args.inputlen, args.outputlen, args.max_jitter, args.batch_size,
            negative_sampling_ratio=args.negative_sampling_ratio,
            add_revcomp=revcomp, shuffle=shuffle)

    train_gen = make_gen(train_chroms, revcomp=True, shuffle=True)
    valid_gen = make_gen(valid_chroms, revcomp=False, shuffle=False)

    model = mt.getModelGivenModelOptionsAndWeightInits(
        types.SimpleNamespace(seed=args.seed, learning_rate=args.learning_rate),
        {'filters': args.filters, 'n_dil_layers': args.n_dil_layers,
         'inputlen': args.inputlen, 'outputlen': args.outputlen,
         'counts_loss_weight': args.counts_loss_weight,
         'nucleosome_profile_weight': args.nucleosome_profile_weight,
         'nucleosome_counts_weight': args.nucleosome_counts_weight})

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=args.early_stop_patience,
                                         restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(args.output_prefix + '.weights.h5',
                                           monitor='val_loss', save_best_only=True,
                                           save_weights_only=True),
        tf.keras.callbacks.CSVLogger(args.output_prefix + '.log'),
    ]
    model.fit(train_gen, validation_data=valid_gen, epochs=args.epochs, callbacks=callbacks)
    model.save(args.output_prefix + '.h5')
    print('saved', args.output_prefix + '.h5')


if __name__ == '__main__':
    main()
