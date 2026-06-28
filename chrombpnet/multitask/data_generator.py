"""Multi-task batch generator: feeds a shared sequence input plus TWO base-
resolution targets (accessibility cut-site profile + nucleosome dyad profile).

Mirrors chrombpnet's ChromBPNetBatchGenerator but loads two bigwigs and applies
the SAME random crop offset and the SAME reverse-complement flip to both targets
so they stay aligned with the sequence. Yields:

    (seq,  [acc_profile, acc_logcount, nuc_profile, nuc_logcount])

matching multitask_nucleosome_model's 4 outputs.
"""
import numpy as np
from tensorflow import keras
import pyBigWig, pyfaidx
from chrombpnet.training.utils import one_hot, data_utils


def _take_per_row(A, starts, width):
    idx = starts[:, None] + np.arange(width)
    return A[np.arange(idx.shape[0])[:, None], idx]


def shared_random_crop(seqs, lab1, lab2, seq_w, lab_w):
    """Crop seq + both labels at one random offset per row (shared)."""
    max_start = seqs.shape[1] - seq_w
    assert seqs.shape[1] - seq_w == lab1.shape[1] - lab_w == lab2.shape[1] - lab_w
    starts = np.random.randint(0, max_start + 1, size=seqs.shape[0])
    return (_take_per_row(seqs, starts, seq_w),
            _take_per_row(lab1, starts, lab_w),
            _take_per_row(lab2, starts, lab_w))


def shared_rev_comp(seqs, lab1, lab2, frac=0.5):
    """Reverse-complement a fraction of rows; flip both label profiles identically."""
    k = int(seqs.shape[0] * frac)
    if k == 0:
        return seqs, lab1, lab2
    rc = np.random.choice(seqs.shape[0], size=k, replace=False)
    seqs = seqs.copy(); lab1 = lab1.copy(); lab2 = lab2.copy()
    seqs[rc] = seqs[rc, ::-1, ::-1]
    lab1[rc] = lab1[rc, ::-1]
    lab2[rc] = lab2[rc, ::-1]
    return seqs, lab1, lab2


def _cts(df, bw, width):
    vals = []
    for _, r in df.iterrows():
        vals.append(np.nan_to_num(
            bw.values(r['chr'], r['start'] + r['summit'] - width // 2,
                      r['start'] + r['summit'] + width // 2)))
    return np.array(vals, dtype=np.float32)


class MultiTaskBatchGenerator(keras.utils.Sequence):
    def __init__(self, peaks_df, nonpeaks_df, genome_fasta, acc_bw_file, nuc_bw_file,
                 inputlen, outputlen, max_jitter, batch_size,
                 negative_sampling_ratio=1.0, add_revcomp=True, shuffle=True):
        self.inputlen, self.outputlen = inputlen, outputlen
        self.batch_size = batch_size
        self.negative_sampling_ratio = negative_sampling_ratio
        self.add_revcomp = add_revcomp
        self.shuffle = shuffle

        genome = pyfaidx.Fasta(genome_fasta)
        acc_bw = pyBigWig.open(acc_bw_file)
        nuc_bw = pyBigWig.open(nuc_bw_file)

        # peaks loaded wide (for jitter), nonpeaks at exact width
        self.pk_seq = self.pk_acc = self.pk_nuc = None
        if peaks_df is not None and len(peaks_df):
            self.pk_seq = data_utils.get_seq(peaks_df, genome, inputlen + 2 * max_jitter)
            self.pk_acc = _cts(peaks_df, acc_bw, outputlen + 2 * max_jitter)
            self.pk_nuc = _cts(peaks_df, nuc_bw, outputlen + 2 * max_jitter)
        self.np_seq = self.np_acc = self.np_nuc = None
        if nonpeaks_df is not None and len(nonpeaks_df):
            self.np_seq = data_utils.get_seq(nonpeaks_df, genome, inputlen)
            self.np_acc = _cts(nonpeaks_df, acc_bw, outputlen)
            self.np_nuc = _cts(nonpeaks_df, nuc_bw, outputlen)
        genome.close(); acc_bw.close(); nuc_bw.close()
        self.on_epoch_end()

    def _assemble(self):
        parts_seq, parts_acc, parts_nuc = [], [], []
        if self.pk_seq is not None:
            s, a, n = shared_random_crop(self.pk_seq, self.pk_acc, self.pk_nuc,
                                         self.inputlen, self.outputlen)
            parts_seq.append(s); parts_acc.append(a); parts_nuc.append(n)
        if self.np_seq is not None:
            ns, na, nn = self.np_seq, self.np_acc, self.np_nuc
            if self.negative_sampling_ratio < 1.0 and self.pk_seq is not None:
                k = int(self.negative_sampling_ratio * len(self.pk_seq))
                idx = np.random.choice(len(ns), size=min(k, len(ns)), replace=False)
                ns, na, nn = ns[idx], na[idx], nn[idx]
            parts_seq.append(ns); parts_acc.append(na); parts_nuc.append(nn)
        seq = np.vstack(parts_seq); acc = np.vstack(parts_acc); nuc = np.vstack(parts_nuc)
        if self.add_revcomp:
            seq, acc, nuc = shared_rev_comp(seq, acc, nuc)
        if self.shuffle:
            perm = np.random.permutation(len(seq))
            seq, acc, nuc = seq[perm], acc[perm], nuc[perm]
        self.seq, self.acc, self.nuc = seq, acc, nuc

    def on_epoch_end(self):
        self._assemble()

    def __len__(self):
        return int(np.ceil(len(self.seq) / self.batch_size))

    def __getitem__(self, i):
        sl = slice(i * self.batch_size, (i + 1) * self.batch_size)
        seq, acc, nuc = self.seq[sl], self.acc[sl], self.nuc[sl]
        acc_lc = np.log(1 + acc.sum(-1, keepdims=True))
        nuc_lc = np.log(1 + nuc.sum(-1, keepdims=True))
        return seq, [acc, acc_lc, nuc, nuc_lc]
