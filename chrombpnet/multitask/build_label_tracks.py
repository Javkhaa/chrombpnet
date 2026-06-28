#!/usr/bin/env python3
"""Build base-resolution label bigwigs from a standardized fragment file.

Two modes:
  dyad     (head 2): for mono-nucleosomal fragments (insert in [min_len,max_len)),
                     add `count` at the fragment CENTER (dyad estimate). Optional
                     Gaussian smoothing (NucleoATAC-style occupancy).
  cutsite  (head 1): add `count` at both Tn5 cut sites (start and end-1). chrombpnet
                     builds this itself with auto shift-detection, so prefer its
                     pipeline for the accessibility target; this mode exists for
                     parity / standalone use.

Input fragment file = 5-col BED (chr,start,end,barcode,count), hg38, .tsv[.gz].
Restricts to the standard chromosomes by default (what the chrombpnet folds use).

Memory note: allocates one float32 array per chromosome it sees (~12 GB for the
whole genome). Fine on the H100 VM; for a laptop test use a small chrom.sizes.
"""
import argparse, gzip, sys
import numpy as np
import pyBigWig

MAIN_CHROMS = {f"chr{c}" for c in list(range(1, 23)) + ["X", "Y"]}


def open_maybe_gzip(path):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path, "r")


def load_chrom_sizes(path, main_only):
    sizes = {}
    with open(path) as fh:
        for line in fh:
            c, s = line.split()[:2]
            if (not main_only) or (c in MAIN_CHROMS):
                sizes[c] = int(s)
    return sizes


def accumulate(frag_paths, sizes, mode, min_len, max_len):
    arrays = {c: np.zeros(n, dtype=np.float32) for c, n in sizes.items()}
    n_used = n_total = 0
    for path in frag_paths:
        with open_maybe_gzip(path) as fh:
            for line in fh:
                p = line.rstrip("\n").split("\t")
                if len(p) < 3:
                    continue
                c = p[0]
                if c not in arrays:
                    continue
                try:
                    s, e = int(p[1]), int(p[2])
                except ValueError:
                    continue
                cnt = int(p[4]) if len(p) >= 5 and p[4].isdigit() else 1
                n_total += 1
                arr = arrays[c]; L = e - s
                if mode == "dyad":
                    if not (min_len <= L < max_len):
                        continue
                    center = (s + e) // 2
                    if 0 <= center < arr.shape[0]:
                        arr[center] += cnt; n_used += 1
                else:  # cutsite
                    if 0 <= s < arr.shape[0]:
                        arr[s] += cnt
                    if 0 <= e - 1 < arr.shape[0]:
                        arr[e - 1] += cnt
                    n_used += 1
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
    ap.add_argument("--all-chroms", action="store_true",
                    help="keep all contigs (default: chr1-22,X,Y only)")
    args = ap.parse_args()

    sizes = load_chrom_sizes(args.chrom_sizes, main_only=not args.all_chroms)
    print(f"[tracks] {len(sizes)} chroms | mode={args.mode}", file=sys.stderr)
    arrays, n_used, n_total = accumulate(args.frag, sizes, args.mode,
                                         args.min_len, args.max_len)
    print(f"[tracks] used {n_used:,}/{n_total:,} fragments "
          f"({100*n_used/max(n_total,1):.1f}%)", file=sys.stderr)
    if args.smooth_sigma > 0 and args.mode == "dyad":
        smooth(arrays, args.smooth_sigma)
        print(f"[tracks] smoothed sigma={args.smooth_sigma}", file=sys.stderr)
    write_bigwig(arrays, sizes, args.out)
    print(f"[tracks] wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
