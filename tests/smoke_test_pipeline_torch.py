#!/usr/bin/env python3
"""End-to-end PyTorch smoke test for fragments -> tracks -> dataset -> model."""
import gzip
import os
import tempfile
import types

import numpy as np
import pandas as pd
import pyfaidx
import torch

from chrombpnet.multitask import build_label_tracks as blt
from chrombpnet.multitask.torch_data import MultiTaskRegionDataset
from chrombpnet.training.models.multitask_nucleosome_torch import MultiTaskNucleosomeModel, multitask_loss

rng = np.random.default_rng(0)
chrom, chromlen = 'chr1', 60000
inputlen, outputlen, jitter = 2114, 1000, 100
work = tempfile.mkdtemp(prefix='mt_torch_smoke_')
print('workdir', work)

fa = os.path.join(work, 'genome.fa')
seq = ''.join(rng.choice(list('ACGT'), size=chromlen))
with open(fa, 'w') as f:
    f.write(f'>{chrom}\n')
    for i in range(0, chromlen, 80):
        f.write(seq[i:i+80] + '\n')
pyfaidx.Fasta(fa)
cs = os.path.join(work, 'chrom.sizes')
open(cs, 'w').write(f'{chrom}\t{chromlen}\n')

centers = [15000, 30000, 45000]
frag = os.path.join(work, 'frags.tsv.gz')
with gzip.open(frag, 'wt') as f:
    for _ in range(20000):
        c = rng.choice(centers) + int(rng.normal(0, 400))
        L = int(rng.choice([rng.integers(20, 100), rng.integers(150, 250), rng.integers(300, 470)]))
        s = max(0, c - L // 2); e = min(chromlen - 1, s + L)
        if e > s:
            f.write(f'{chrom}\t{s}\t{e}\tbc\t1\n')

sizes = blt.load_chrom_sizes(cs, main_only=True)
acc_counts, *_ = blt.accumulate_sparse([frag], sizes, 'cutsite', 150, 250)
acc_bw = os.path.join(work, 'acc.bw'); blt.write_sparse_bigwig(acc_counts, sizes, acc_bw)
nuc_counts, *_ = blt.accumulate_sparse([frag], sizes, 'dyad', 150, 250)
nuc_bw = os.path.join(work, 'nuc.bw'); blt.write_sparse_bigwig(nuc_counts, sizes, nuc_bw)

def mkdf(vals):
    return pd.DataFrame({'chr': chrom, 'start': [c - 500 for c in vals], 'summit': 500})

ds = MultiTaskRegionDataset(mkdf(centers), mkdf([8000, 22000, 38000, 52000]),
                            fa, acc_bw, nuc_bw, inputlen, outputlen, jitter,
                            negative_sampling_ratio=1.0, add_revcomp=True, seed=0)
seq_t, acc_t, acc_lc, nuc_t, nuc_lc = [x.unsqueeze(0) for x in ds[0]]
model = MultiTaskNucleosomeModel(inputlen=inputlen, outputlen=outputlen, filters=32, n_dil_layers=8)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
outs = model(seq_t.float())
loss = multitask_loss(outs, (acc_t.float(), acc_lc.float(), nuc_t.float(), nuc_lc.float()))
loss.backward(); opt.step()
assert torch.isfinite(loss)
print('TORCH_PIPELINE_SMOKE_OK', float(loss.detach()))
