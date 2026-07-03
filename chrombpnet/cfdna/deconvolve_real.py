"""Deconvolve a REAL cfDNA sample against the 146-cell-type reference.

Builds the reference panel + matrix `R` from the corpus tracks (same path as the
Phase 0 PoC), reduces a cfDNA fragments parquet to a feature vector `y` on the
identical panel, and solves the simplex-constrained deconvolution. Reports where
the mass lands (with a hematopoietic-vs-solid-tissue readout -- the Snyder
healthy-plasma sanity check) and the reconstruction quality (how well an
ATAC-derived reference explains cfDNA coverage -- the domain-gap indicator).

An "unknown" background column is optionally appended (CelFiE-style) so signal
from cell types absent in the corpus (e.g. adult peripheral-blood granulocytes,
which the fetal/tissue-heavy corpus lacks) is not force-attributed.

    chrombpnet-cfdna-deconvolve -m manifest.tsv --manifest-root ROOT \
        --cfdna sample.fragments.parquet --chroms chr1 chr3 chr6 -o out.json
"""
from __future__ import annotations

import argparse
import json
import os
import re

import numpy as np

from chrombpnet.multitask.train_multicell_torch import parse_manifest
from chrombpnet.cfdna import reference as ref
from chrombpnet.cfdna import deconvolve as dc
from chrombpnet.cfdna import ingest
from chrombpnet.cfdna.run_poc import build_R

# cell-type-name patterns for the hematopoietic (blood/immune) readout
HEME = re.compile(r"lymph|myeloid|macrophage|erythro|megakaryo|thymocyte|"
                  r"hematop|haematop|monocyt|granulocyt|neutrophil|"
                  r"\bt_cell|\bb_cell|microglia|\bmicro\b|dendritic|mast|"
                  r"stem_cell|hspc|\bhsc\b|kupffer", re.I)


def is_heme(name):
    return bool(HEME.search(name))


def normalize_like_R(acc_raw, nuc_raw, use_acc, use_nuc):
    blocks = []
    if use_acc:
        blocks.append(acc_raw)
    if use_nuc:
        blocks.append(nuc_raw)
    return ref.normalize_reference(*blocks)


def deconvolve_one(y, R, cell_types, add_unknown=True):
    Rw = R
    names = list(cell_types)
    if add_unknown:
        # flat background column (uniform over regions), normalized like the rest
        u = np.full((R.shape[0], 1), 1.0 / R.shape[0])
        Rw = np.hstack([R, u])
        names = names + ["__unknown__"]
    w = dc.deconvolve(y, Rw, l2=1e-4)
    recon = Rw @ w
    r = float(np.corrcoef(y, recon)[0, 1])
    return w, names, r


def report(w, names, recon_r, top=15):
    order = np.argsort(w)[::-1]
    heme_mass = float(sum(w[i] for i, n in enumerate(names)
                          if n != "__unknown__" and is_heme(n)))
    solid_mass = float(sum(w[i] for i, n in enumerate(names)
                           if n != "__unknown__" and not is_heme(n)))
    unk = float(sum(w[i] for i, n in enumerate(names) if n == "__unknown__"))
    lines = []
    lines.append(f"  reconstruction r(y, R w) = {recon_r:.3f}")
    lines.append(f"  hematopoietic mass = {heme_mass:.3f} | "
                 f"solid-tissue mass = {solid_mass:.3f} | unknown = {unk:.3f}")
    lines.append(f"  {'rank':>4} {'weight':>8}  {'heme?':>5}  cell type")
    for k in order[:top]:
        n = names[k]
        tag = "UNK" if n == "__unknown__" else ("HEME" if is_heme(n) else "")
        lines.append(f"  {list(order).index(k)+1:>4} {w[k]:>8.4f}  {tag:>5}  {n}")
    return "\n".join(lines), {"heme_mass": heme_mass, "solid_mass": solid_mass,
                              "unknown_mass": unk, "recon_r": recon_r,
                              "top": [(names[k], float(w[k])) for k in order[:top]]}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-m", "--manifest", required=True)
    ap.add_argument("--manifest-root", default=None)
    ap.add_argument("--cfdna", required=True, help="cfDNA fragments parquet")
    ap.add_argument("--chroms", nargs="*", default=["chr1", "chr3", "chr6"])
    ap.add_argument("--n-regions", type=int, default=5000)
    ap.add_argument("--top-per-cell", type=int, default=250)
    ap.add_argument("--half-width", type=int, default=500)
    ap.add_argument("--n-markers", type=int, default=2000)
    ap.add_argument("--feature-sets", nargs="*", default=["nuc", "acc", "acc+nuc"])
    ap.add_argument("--size-gate", nargs=2, type=int, default=[100, 250])
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-o", "--out-json", default=None)
    a = ap.parse_args()

    root = a.manifest_root or os.path.dirname(os.path.abspath(a.manifest))
    manifest = parse_manifest(a.manifest, root)
    cell_types = [m["cell_type"] for m in manifest]
    print(f"cell types: {len(cell_types)} | chroms {a.chroms}")

    print("building region panel + reference R ...")
    regions = ref.build_region_panel(manifest, a.chroms, top_per_cell=a.top_per_cell,
                                     n_regions=a.n_regions, half_width=a.half_width,
                                     seed=a.seed)
    acc_R, nuc_R = build_R(manifest, regions, a.jobs)
    marker_idx = ref.select_markers(acc_R + nuc_R, a.n_markers, method="specificity")
    print(f"  panel {len(regions)} regions -> {len(marker_idx)} markers")

    print(f"reducing cfDNA {os.path.basename(a.cfdna)} over the panel ...")
    qc = ingest.fragment_length_summary(a.cfdna)
    print(f"  cfDNA QC: n={qc['n_fragments']:,} median_len={qc['median_len']:.0f} "
          f"modal={qc['modal_len_bin']:.0f} mono-nuc%={qc['frac_mono_nuc_100_250']:.2f}")
    acc_y_full, nuc_y_full = ingest.cfdna_feature_vectors(
        a.cfdna, regions, size_gate=tuple(a.size_gate))
    acc_y, nuc_y = acc_y_full[marker_idx], nuc_y_full[marker_idx]
    acc_Rm, nuc_Rm = acc_R[marker_idx], nuc_R[marker_idx]

    report_json = {"cfdna": os.path.basename(a.cfdna), "qc": qc,
                   "n_markers": int(len(marker_idx)), "feature_sets": {}}
    for fs in a.feature_sets:
        use_acc, use_nuc = ("acc" in fs), ("nuc" in fs)
        R = normalize_like_R(acc_Rm, nuc_Rm, use_acc, use_nuc)
        # y through the identical normalization (single-column reuse)
        y2 = normalize_like_R(acc_y[:, None], nuc_y[:, None], use_acc, use_nuc)[:, 0]
        w, names, rr = deconvolve_one(y2, R, cell_types, add_unknown=True)
        txt, js = report(w, names, rr)
        print(f"\n=== feature set: {fs} ===\n{txt}")
        report_json["feature_sets"][fs] = js

    if a.out_json:
        json.dump(report_json, open(a.out_json, "w"), indent=2)
        print(f"\nwrote {a.out_json}")


if __name__ == "__main__":
    main()
