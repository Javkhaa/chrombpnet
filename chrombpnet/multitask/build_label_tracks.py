#!/usr/bin/env python3
"""Build base-resolution label bigwigs from a standardized fragment file.

Two modes:
  dyad     (head 2): for mono-nucleosomal fragments (insert in [min_len,max_len)),
                     add `count` at the fragment CENTER (dyad estimate). Optional
                     Gaussian smoothing (NucleoATAC-style occupancy).
  cutsite  (head 1): add `count` at both ATAC cut sites after applying the
                     ChromBPNet convention (+4/-4 by default). Use
                     --plus-shift/--minus-shift to override.

Input fragment file = 5-col BED (chr,start,end,barcode,count), hg38, .tsv[.gz].
Restricts to the standard chromosomes by default (what the chrombpnet folds use).

The default path uses Polars to aggregate sparse positions, avoiding whole-genome
dense arrays. Gaussian smoothing still materializes dense arrays because the
kernel touches neighboring bases.
"""
import argparse
import sys

import numpy as np
import polars as pl
import pyBigWig

MAIN_CHROMS = {f"chr{c}" for c in list(range(1, 23)) + ["X", "Y"]}


def load_chrom_sizes(path, main_only):
    sizes = {}
    with open(path) as fh:
        for line in fh:
            c, s = line.split()[:2]
            if (not main_only) or (c in MAIN_CHROMS):
                sizes[c] = int(s)
    return sizes


def _scan_fragments(path):
    return pl.scan_csv(
        path,
        separator="\t",
        has_header=False,
        new_columns=["chrom", "start", "end", "barcode", "count"],
        schema_overrides={
            "chrom": pl.String,
            "start": pl.Int64,
            "end": pl.Int64,
            "barcode": pl.String,
            "count": pl.Int64,
        },
    )


def _position_counts_for_path(path, sizes, mode, min_len, max_len, plus_shift, minus_shift):
    chrom_size = pl.col("chrom").replace(sizes, return_dtype=pl.Int64)
    lf = _scan_fragments(path).filter(pl.col("chrom").is_in(list(sizes)))
    total = int(lf.select(pl.len().alias("n")).collect(engine="streaming").item())

    length = pl.col("end") - pl.col("start")
    if mode == "dyad":
        mono = lf.filter((length >= min_len) & (length < max_len))
        used = int(mono.select(pl.len().alias("n")).collect(engine="streaming").item())
        counts = (
            mono
            .with_columns(((pl.col("start") + pl.col("end")) // 2).alias("pos"))
            .filter((pl.col("pos") >= 0) & (pl.col("pos") < chrom_size))
            .group_by(["chrom", "pos"])
            .agg(pl.col("count").sum().alias("value"))
            .collect(engine="streaming")
        )
        return counts, used, total

    starts = lf.select(
        pl.col("chrom"),
        (pl.col("start") + plus_shift).alias("pos"),
        pl.col("count").alias("value"),
    )
    ends = lf.select(
        pl.col("chrom"),
        (pl.col("end") + minus_shift - 1).alias("pos"),
        pl.col("count").alias("value"),
    )
    counts = (
        pl.concat([starts, ends])
        .filter((pl.col("pos") >= 0) & (pl.col("pos") < chrom_size))
        .group_by(["chrom", "pos"])
        .agg(pl.col("value").sum().alias("value"))
        .collect(engine="streaming")
    )
    return counts, total, total


def accumulate_sparse(frag_paths, sizes, mode, min_len, max_len, plus_shift=4, minus_shift=-4):
    frames = []
    n_used = n_total = 0
    for path in frag_paths:
        counts, used, total = _position_counts_for_path(path, sizes, mode, min_len, max_len,
                                                        plus_shift, minus_shift)
        if counts.height:
            frames.append(counts)
        n_used += used
        n_total += total
    if not frames:
        return pl.DataFrame({"chrom": [], "pos": [], "value": []}), n_used, n_total
    counts = (
        pl.concat(frames)
        .group_by(["chrom", "pos"])
        .agg(pl.col("value").sum().alias("value"))
        .sort(["chrom", "pos"])
    )
    return counts, n_used, n_total


def accumulate(frag_paths, sizes, mode, min_len, max_len, plus_shift=4, minus_shift=-4):
    """Compatibility helper returning dense arrays for tests/smoothing."""
    counts, n_used, n_total = accumulate_sparse(frag_paths, sizes, mode, min_len, max_len,
                                                plus_shift, minus_shift)
    arrays = {c: np.zeros(n, dtype=np.float32) for c, n in sizes.items()}
    for chrom, group in counts.partition_by("chrom", as_dict=True).items():
        chrom = chrom[0] if isinstance(chrom, tuple) else chrom
        if chrom not in arrays:
            continue
        pos = group["pos"].to_numpy().astype(np.int64)
        val = group["value"].to_numpy().astype(np.float32)
        arrays[chrom][pos] = val
    return arrays, n_used, n_total


def smooth(arrays, sigma):
    from scipy.ndimage import gaussian_filter1d
    for c in arrays:
        arrays[c] = gaussian_filter1d(arrays[c], sigma=sigma).astype(np.float32)


def write_bigwig(arrays, sizes, out_path):
    bw = pyBigWig.open(out_path, "w")
    chroms = [(c, sizes[c]) for c in sizes if c in arrays]
    bw.addHeader(chroms)
    for c, _ in chroms:
        arr = arrays[c]
        nz = np.nonzero(arr)[0]
        if len(nz) == 0:
            continue
        bw.addEntries(c, nz.astype(int).tolist(),
                      values=arr[nz].astype(float).tolist(), span=1)
    bw.close()


def write_sparse_bigwig(counts, sizes, out_path):
    bw = pyBigWig.open(out_path, "w")
    chroms = [(c, sizes[c]) for c in sizes]
    bw.addHeader(chroms)
    if counts.height:
        groups = counts.partition_by("chrom", as_dict=True)
        for chrom, _ in chroms:
            group = groups.get(chrom) or groups.get((chrom,))
            if group is None or group.height == 0:
                continue
            group = group.sort("pos")
            bw.addEntries(
                chrom,
                group["pos"].to_list(),
                values=group["value"].cast(pl.Float64).to_list(),
                span=1,
            )
    bw.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frag", nargs="+", required=True, help="fragment .tsv[.gz] file(s)")
    ap.add_argument("--chrom-sizes", required=True)
    ap.add_argument("--out", required=True, help="output bigwig path")
    ap.add_argument("--mode", choices=["dyad", "cutsite"], default="dyad")
    ap.add_argument("--min-len", type=int, default=150)
    ap.add_argument("--max-len", type=int, default=250)
    ap.add_argument("--smooth-sigma", type=float, default=0.0,
                    help="Gaussian sigma (bp) for dyad smoothing; 0 = off")
    ap.add_argument("--plus-shift", type=int, default=4,
                    help="cutsite mode: shift applied to the fragment start (default: +4)")
    ap.add_argument("--minus-shift", type=int, default=-4,
                    help="cutsite mode: shift applied to the fragment end before taking the 5' base (default: -4)")
    ap.add_argument("--all-chroms", action="store_true",
                    help="keep all contigs (default: chr1-22,X,Y only)")
    args = ap.parse_args()

    sizes = load_chrom_sizes(args.chrom_sizes, main_only=not args.all_chroms)
    print(f"[tracks] {len(sizes)} chroms | mode={args.mode}", file=sys.stderr)
    if args.smooth_sigma > 0 and args.mode == "dyad":
        arrays, n_used, n_total = accumulate(args.frag, sizes, args.mode,
                                             args.min_len, args.max_len,
                                             args.plus_shift, args.minus_shift)
    else:
        counts, n_used, n_total = accumulate_sparse(args.frag, sizes, args.mode,
                                                    args.min_len, args.max_len,
                                                    args.plus_shift, args.minus_shift)
    print(f"[tracks] used {n_used:,}/{n_total:,} fragments "
          f"({100*n_used/max(n_total,1):.1f}%)", file=sys.stderr)
    if args.smooth_sigma > 0 and args.mode == "dyad":
        smooth(arrays, args.smooth_sigma)
        print(f"[tracks] smoothed sigma={args.smooth_sigma}", file=sys.stderr)
        write_bigwig(arrays, sizes, args.out)
    else:
        write_sparse_bigwig(counts, sizes, args.out)
    print(f"[tracks] wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
