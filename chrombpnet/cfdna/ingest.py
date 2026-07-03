"""Convert a cfDNA fragment table into feature vectors on the reference footing.

Input: a per-sample fragments parquet (columns chromosome, start_position,
end_position, ...) of merged proper-pair cfDNA fragments (e.g. Tao 2023 /
GSE186573 in-house processing). Output: per-region feature vectors over the
*same* marker panel used to build the reference matrix `R`, so `y` and `R`
are directly comparable for deconvolution.

Two channels, matching the model's two heads / R's two blocks:
  - **dyad / nucleosome** = fragment MIDPOINT density. A mono-nucleosomal
    fragment's midpoint approximates the protected nucleosome dyad (Snyder 2016).
  - **cut-site / accessibility** = fragment ENDPOINT density. cfDNA fragment ends
    cluster at nuclease-accessible boundaries -- the cfDNA analog of an ATAC
    cut site.

Both are reduced to per-region windowed counts (same reduction as `R`'s bigWig
sums), via sorted-position + searchsorted. Size-gating keeps mono-nucleosomal
fragments (default 100-250 bp) for the dyad channel.
"""
from __future__ import annotations

import numpy as np
import pyarrow.parquet as pq


def load_fragment_positions(parquet_path, size_gate=(100, 250)):
    """Read fragment coords, size-gate, return per-chrom sorted midpoints+endpoints.

    Returns dict: chrom -> {"mid": sorted int32 array of midpoints,
                            "end": sorted int32 array of both fragment ends}.
    """
    t = pq.read_table(parquet_path,
                      columns=["chromosome", "start_position", "end_position"])
    chrom = t.column("chromosome").to_numpy(zero_copy_only=False)
    s = t.column("start_position").to_numpy().astype(np.int64)
    e = t.column("end_position").to_numpy().astype(np.int64)
    L = e - s
    if size_gate is not None:
        keep = (L >= size_gate[0]) & (L <= size_gate[1])
        chrom, s, e = chrom[keep], s[keep], e[keep]
    mid = (s + e) // 2
    out = {}
    order = np.argsort(chrom, kind="stable")
    chrom_s = chrom[order]
    # group by chrom
    uniq, starts = np.unique(chrom_s, return_index=True)
    starts = list(starts) + [len(chrom_s)]
    inv = np.argsort(order)  # not needed; slice s/e by same order
    s_s, e_s, mid_s = s[order], e[order], mid[order]
    for i, c in enumerate(uniq):
        a, b = starts[i], starts[i + 1]
        ends = np.concatenate([s_s[a:b], e_s[a:b]])
        out[c] = {"mid": np.sort(mid_s[a:b]), "end": np.sort(ends)}
    return out


def _count_in_windows(sorted_pos, region_starts, region_ends):
    """Count sorted positions falling in each [start, end) window via searchsorted."""
    lo = np.searchsorted(sorted_pos, region_starts, side="left")
    hi = np.searchsorted(sorted_pos, region_ends, side="left")
    return (hi - lo).astype(np.float64)


def cfdna_feature_vectors(parquet_path, regions, size_gate=(100, 250)):
    """Per-region (cut-site endpoints, dyad midpoints) counts over the panel.

    `regions` is the panel DataFrame[chr, center, start, end] used for `R`.
    Returns (acc_vec, nuc_vec), each length len(regions), raw counts.
    """
    pos = load_fragment_positions(parquet_path, size_gate=size_gate)
    n = len(regions)
    acc = np.zeros(n, dtype=np.float64)   # endpoints  (cut-site channel)
    nuc = np.zeros(n, dtype=np.float64)   # midpoints  (dyad channel)
    rchr = regions["chr"].to_numpy()
    rs = regions["start"].to_numpy().astype(np.int64)
    re = regions["end"].to_numpy().astype(np.int64)
    for c in np.unique(rchr):
        m = rchr == c
        if c not in pos:
            continue
        acc[m] = _count_in_windows(pos[c]["end"], rs[m], re[m])
        nuc[m] = _count_in_windows(pos[c]["mid"], rs[m], re[m])
    return acc, nuc


def fragment_length_summary(parquet_path):
    """Quick QC: fragment-length stats (should peak at ~167 bp for real cfDNA)."""
    t = pq.read_table(parquet_path, columns=["start_position", "end_position"])
    L = (t.column("end_position").to_numpy().astype(np.int64)
         - t.column("start_position").to_numpy().astype(np.int64))
    h, edges = np.histogram(L, bins=np.arange(0, 401, 5))
    return {
        "n_fragments": int(len(L)),
        "median_len": float(np.median(L)),
        "modal_len_bin": float(edges[np.argmax(h)]),
        "frac_mono_nuc_100_250": float(((L >= 100) & (L <= 250)).mean()),
        "frac_sub_nuc_lt100": float((L < 100).mean()),
    }
