"""Build the deconvolution reference matrix `R` from per-cell-type tracks.

Given the training manifest (per-cell-type accessibility cut-site + nucleosome
dyad bigWigs and peak calls), this module:

  1. builds a panel of candidate marker regions (union of top peaks across cell
     types, de-duplicated onto a genomic grid), restricted to chosen chromosomes;
  2. reduces each cell type's tracks to a per-region feature vector (windowed
     signal sum) for the accessibility and/or nucleosome-dyad channel;
  3. assembles and normalizes the reference matrix `R` (columns = cell types),
     with optional marker selection to keep the most cell-type-discriminative
     regions (better-conditioned `R`).

Everything here is depth-free bookkeeping over the *observed* tracks (Role (a)
in the methodology doc). Model-imputed references (Role (b)) are a later phase.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyBigWig


# ---------------------------------------------------------------------------
# region panel
# ---------------------------------------------------------------------------
def _read_narrowpeak_summits(path, chroms):
    """Return DataFrame[chr, summit, signal] of peak summits on `chroms`.

    narrowPeak: col1 chrom, col2 start(0-based), col3 end, col7 signalValue,
    col10 peak offset from start (0-based). summit = start + offset.
    """
    cols = ["chr", "start", "end", "name", "score", "strand",
            "signal", "pval", "qval", "offset"]
    df = pd.read_csv(path, sep="\t", header=None, names=cols, usecols=range(10))
    df = df[df["chr"].isin(chroms)]
    if len(df) == 0:
        return pd.DataFrame(columns=["chr", "summit", "signal"])
    summit = df["start"].to_numpy() + df["offset"].to_numpy()
    return pd.DataFrame({"chr": df["chr"].to_numpy(),
                         "summit": summit.astype(int),
                         "signal": df["signal"].to_numpy(float)})


def build_region_panel(manifest, chroms, top_per_cell=200, grid=1000,
                       n_regions=4000, half_width=500, seed=0):
    """Union of each cell type's strongest peak summits, de-duplicated.

    Taking the *top* peaks per cell type (by signalValue) and unioning them
    ensures every cell type's characteristic regions are represented, so the
    reference columns are well separated. Summits are snapped to a `grid`-bp
    lattice and de-duplicated so near-identical peaks collapse to one region.

    Returns DataFrame[chr, center, start, end] of length <= n_regions.
    """
    chroms = set(chroms)
    parts = []
    for m in manifest:
        s = _read_narrowpeak_summits(m["peaks"], chroms)
        if len(s):
            s = s.nlargest(min(top_per_cell, len(s)), "signal")
            parts.append(s)
    if not parts:
        raise ValueError("no peak summits found on the requested chromosomes")
    alls = pd.concat(parts, ignore_index=True)
    # snap to grid and keep, per (chr, grid-cell), the strongest summit
    alls["key_bin"] = (alls["summit"] // grid).astype(int)
    alls = (alls.sort_values("signal", ascending=False)
                .drop_duplicates(subset=["chr", "key_bin"], keep="first"))
    if len(alls) > n_regions:
        alls = alls.sample(n=n_regions, random_state=seed)
    alls = alls.sort_values(["chr", "summit"]).reset_index(drop=True)
    center = alls["summit"].to_numpy().astype(int)
    return pd.DataFrame({
        "chr": alls["chr"].to_numpy(),
        "center": center,
        "start": center - half_width,
        "end": center + half_width,
    })


# ---------------------------------------------------------------------------
# feature extraction
# ---------------------------------------------------------------------------
def _region_sums(bw_path, regions):
    """Windowed signal sum per region from a bigWig. Missing -> 0."""
    bw = pyBigWig.open(bw_path)
    out = np.zeros(len(regions), dtype=np.float64)
    try:
        chrom_lens = bw.chroms()
        for i, r in enumerate(regions.itertuples(index=False)):
            start = max(0, int(r.start))
            end = int(r.end)
            clen = chrom_lens.get(r.chr)
            if clen is None:
                continue
            end = min(end, clen)
            if end <= start:
                continue
            v = bw.stats(r.chr, start, end, type="sum")[0]
            if v is not None:
                out[i] = float(v)
    finally:
        bw.close()
    return out


def build_reference_matrix(manifest, regions, tracks=("acc", "nuc")):
    """Assemble raw (un-normalized) per-track reference matrices.

    Returns dict: cell_types (list, len m), and for each requested track a
    matrix of shape [n_regions, m] of windowed signal sums.
    """
    cell_types = [m["cell_type"] for m in manifest]
    n, mm = len(regions), len(manifest)
    mats = {t: np.zeros((n, mm), dtype=np.float64) for t in tracks}
    key = {"acc": "acc_bw", "nuc": "nuc_bw"}
    for j, m in enumerate(manifest):
        for t in tracks:
            mats[t][:, j] = _region_sums(m[key[t]], regions)
    return {"cell_types": cell_types, **mats}


# ---------------------------------------------------------------------------
# marker selection + normalization
# ---------------------------------------------------------------------------
def select_markers(mat, n_markers, method="specificity"):
    """Pick the most cell-type-discriminative region indices from a raw matrix.

    - "specificity": rank regions by max one-vs-rest z-score across cell types
      (favors regions where some cell type is an outlier -- high specificity).
    - "variance": rank by across-cell-type coefficient of variation.
    Operates on per-region L1-normalized-per-column signal so depth differences
    between cell types don't dominate the ranking.
    """
    col = mat / (mat.sum(axis=0, keepdims=True) + 1e-12)      # each cell -> distribution
    row_mean = col.mean(axis=1, keepdims=True)
    row_std = col.std(axis=1, keepdims=True) + 1e-12
    if method == "specificity":
        score = ((col - row_mean) / row_std).max(axis=1)      # best outlier per region
    elif method == "variance":
        score = (col.std(axis=1) / (col.mean(axis=1) + 1e-12))
    else:
        raise ValueError(method)
    n_markers = min(n_markers, mat.shape[0])
    return np.sort(np.argsort(score)[::-1][:n_markers])


def normalize_reference(*blocks):
    """Stack per-track blocks into one reference `R` with equal block weight.

    Each block is L1-normalized per column (cell-type signature = distribution
    over regions), then blocks are stacked row-wise. With column-sum-1 blocks
    and `sum(w)=1`, the clean mixture `R w` is itself a distribution -- the
    well-posed footing the multinomial depth model (see deconvolve) assumes.
    """
    normed = []
    for b in blocks:
        normed.append(b / (b.sum(axis=0, keepdims=True) + 1e-12))
    R = np.vstack(normed)
    # renormalize the stacked columns so each cell-type column sums to 1 overall
    R = R / (R.sum(axis=0, keepdims=True) + 1e-12)
    return R
