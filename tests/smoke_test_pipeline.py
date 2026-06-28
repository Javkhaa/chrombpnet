#!/usr/bin/env python3
"""End-to-end CPU smoke test of the multi-task data path.

Synthesizes a tiny genome + fragment file, builds the accessibility (cut-site)
and nucleosome (dyad) label bigwigs with build_label_tracks, feeds them through
MultiTaskBatchGenerator into the multi-task model, and runs a couple of fit
steps. Proves the whole plumbing (fragments -> tracks -> dual-target batches ->
4-head model) works before any H100 time. Uses small dims so it runs on a laptop.
"""
import os, sys, gzip, tempfile, types, numpy as np
sys.path.insert(0, ".")

import pyfaidx, pandas as pd, tensorflow as tf
from chrombpnet.multitask import build_label_tracks as blt
from chrombpnet.multitask.data_generator import MultiTaskBatchGenerator
from chrombpnet.training.models import multitask_nucleosome_model as mt

rng = np.random.default_rng(0)
CHROM, CHROMLEN = "chr1", 60000
INPUTLEN, OUTPUTLEN, JITTER = 2114, 1000, 100
d = tempfile.mkdtemp(prefix="mt_smoke_")
print("workdir", d)

# --- tiny genome ---
fa = os.path.join(d, "genome.fa")
seq = "".join(rng.choice(list("ACGT"), size=CHROMLEN))
with open(fa, "w") as f:
    f.write(f">{CHROM}\n")
    for i in range(0, CHROMLEN, 80):
        f.write(seq[i:i+80] + "\n")
pyfaidx.Fasta(fa)  # build .fai
cs = os.path.join(d, "chrom.sizes")
open(cs, "w").write(f"{CHROM}\t{CHROMLEN}\n")

# --- tiny fragment file: realistic mixed insert sizes around peak centers ---
peak_centers = [15000, 30000, 45000]
frag = os.path.join(d, "frags.tsv.gz")
with gzip.open(frag, "wt") as f:
    for _ in range(200000):
        c = rng.choice(peak_centers) + int(rng.normal(0, 400))
        L = int(rng.choice([rng.integers(20, 100), rng.integers(150, 250),
                            rng.integers(300, 470)]))
        s = max(0, c - L // 2); e = min(CHROMLEN - 1, s + L)
        if e <= s:
            continue
        f.write(f"{CHROM}\t{s}\t{e}\tbc\t1\n")

# --- build both label tracks via the real builder ---
sizes = blt.load_chrom_sizes(cs, main_only=True)
acc_arr, *_ = blt.accumulate([frag], sizes, "cutsite", 150, 250)
acc_bw = os.path.join(d, "acc.bw"); blt.write_bigwig(acc_arr, sizes, acc_bw)
nuc_arr, *_ = blt.accumulate([frag], sizes, "dyad", 150, 250)
nuc_bw = os.path.join(d, "nuc.bw"); blt.write_bigwig(nuc_arr, sizes, nuc_bw)
print("built tracks:", os.path.exists(acc_bw), os.path.exists(nuc_bw))

# --- regions (peaks centered on the synthetic peak centers; nonpeaks elsewhere) ---
def mkdf(centers):
    return pd.DataFrame({"chr": CHROM, "start": [c - 500 for c in centers],
                         "summit": 500})
peaks = mkdf(peak_centers)
nonpeaks = mkdf([8000, 22000, 38000, 52000])

gen = MultiTaskBatchGenerator(peaks, nonpeaks, fa, acc_bw, nuc_bw,
                              INPUTLEN, OUTPUTLEN, JITTER, batch_size=4,
                              negative_sampling_ratio=1.0, add_revcomp=True)
seqb, targets = gen[0]
print("batch seq:", seqb.shape, "| targets:", [t.shape for t in targets])
assert seqb.shape[1:] == (INPUTLEN, 4)
assert targets[0].shape[1] == OUTPUTLEN and targets[2].shape[1] == OUTPUTLEN
assert targets[1].shape[1] == 1 and targets[3].shape[1] == 1

model = mt.getModelGivenModelOptionsAndWeightInits(
    types.SimpleNamespace(seed=0, learning_rate=1e-3),
    {"filters": 32, "n_dil_layers": 8, "inputlen": INPUTLEN, "outputlen": OUTPUTLEN,
     "counts_loss_weight": 1.0, "nucleosome_profile_weight": 1.0,
     "nucleosome_counts_weight": 1.0})

print("fit 2 steps:")
h = model.fit(gen, epochs=2, verbose=2)
assert np.isfinite(h.history["loss"][-1])
print("\nSMOKE_OK: fragments -> tracks -> dual-target batches -> 4-head model trains.")
