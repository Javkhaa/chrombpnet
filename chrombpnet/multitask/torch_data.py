"""Lazy PyTorch data path for multi-task accessibility+nucleosome training."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyBigWig
import pyfaidx
import torch
from torch.utils.data import Dataset

from chrombpnet.training.utils import one_hot


def _empty_regions():
    return pd.DataFrame(columns=["chr", "start", "summit", "is_peak"])


def _normalize_regions(df, is_peak):
    if df is None or len(df) == 0:
        return _empty_regions()
    regions = df[["chr", "start", "summit"]].copy()
    regions["is_peak"] = bool(is_peak)
    return regions.reset_index(drop=True)


def _fetch_seq(genome, chrom, center, width):
    start = int(center - width // 2)
    end = int(center + width // 2)
    seq = str(genome[chrom][start:end]).upper()
    if len(seq) != width:
        seq = (seq + "N" * width)[:width]
    return one_hot.dna_to_one_hot([seq])[0].astype(np.float32)


def _fetch_bw(bw, chrom, center, width):
    start = int(center - width // 2)
    end = int(center + width // 2)
    vals = bw.values(chrom, start, end)
    return np.nan_to_num(vals).astype(np.float32)


class MultiTaskRegionDataset(Dataset):
    def __init__(self, peaks_df, nonpeaks_df, genome_fasta, acc_bw_file, nuc_bw_file,
                 inputlen, outputlen, max_jitter, negative_sampling_ratio=1.0,
                 add_revcomp=True, shuffle=True, seed=0):
        self.genome_fasta = genome_fasta
        self.acc_bw_file = acc_bw_file
        self.nuc_bw_file = nuc_bw_file
        self.inputlen = inputlen
        self.outputlen = outputlen
        self.max_jitter = max_jitter
        self.negative_sampling_ratio = negative_sampling_ratio
        self.add_revcomp = add_revcomp
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self._genome = None
        self._acc_bw = None
        self._nuc_bw = None

        self.peaks = _normalize_regions(peaks_df, True)
        self.nonpeaks = _normalize_regions(nonpeaks_df, False)
        self.on_epoch_end()

    def _handles(self):
        if self._genome is None:
            self._genome = pyfaidx.Fasta(self.genome_fasta)
        if self._acc_bw is None:
            self._acc_bw = pyBigWig.open(self.acc_bw_file)
        if self._nuc_bw is None:
            self._nuc_bw = pyBigWig.open(self.nuc_bw_file)
        return self._genome, self._acc_bw, self._nuc_bw

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_genome"] = None
        state["_acc_bw"] = None
        state["_nuc_bw"] = None
        return state

    def on_epoch_end(self):
        parts = []
        if len(self.peaks):
            parts.append(self.peaks)
        if len(self.nonpeaks):
            nonpeaks = self.nonpeaks
            if self.negative_sampling_ratio < 1.0 and len(self.peaks):
                k = int(self.negative_sampling_ratio * len(self.peaks))
                idx = self.rng.choice(len(nonpeaks), size=min(k, len(nonpeaks)), replace=False)
                nonpeaks = nonpeaks.iloc[idx].reset_index(drop=True)
            parts.append(nonpeaks)
        if not parts:
            self.regions = _empty_regions()
        else:
            self.regions = pd.concat(parts, ignore_index=True)
        if self.shuffle and len(self.regions):
            order = self.rng.permutation(len(self.regions))
            self.regions = self.regions.iloc[order].reset_index(drop=True)

    def __len__(self):
        return len(self.regions)

    def __getitem__(self, idx):
        genome, acc_bw, nuc_bw = self._handles()
        r = self.regions.iloc[idx]
        center = int(r["start"] + r["summit"])
        if bool(r["is_peak"]):
            jitter = int(self.rng.integers(-self.max_jitter, self.max_jitter + 1)) if self.max_jitter else 0
            center += jitter
        seq = _fetch_seq(genome, r["chr"], center, self.inputlen)
        acc = _fetch_bw(acc_bw, r["chr"], center, self.outputlen)
        nuc = _fetch_bw(nuc_bw, r["chr"], center, self.outputlen)
        if self.add_revcomp and self.rng.random() < 0.5:
            seq = seq[::-1, ::-1].copy()
            acc = acc[::-1].copy()
            nuc = nuc[::-1].copy()
        acc_lc = np.log1p(acc.sum(keepdims=True)).astype(np.float32)
        nuc_lc = np.log1p(nuc.sum(keepdims=True)).astype(np.float32)
        return (
            torch.from_numpy(seq),
            torch.from_numpy(acc),
            torch.from_numpy(acc_lc),
            torch.from_numpy(nuc),
            torch.from_numpy(nuc_lc),
        )
