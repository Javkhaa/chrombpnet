"""Phase 0 in-silico cfDNA deconvolution proof-of-concept driver.

Builds a reference matrix `R` from the multi-task corpus's per-cell-type
accessibility + nucleosome-dyad tracks, then benchmarks reference-based
simplex-constrained deconvolution on synthetic mixtures across depths and
scenarios. Answers the Phase 0 decision-gate question: can we recover known
cell-type fractions from a cfDNA-like feature vector, and down to what depth /
minor fraction? Uses only existing tracks -- no plasma data, no model training.

    chrombpnet-cfdna-poc -m manifest.tsv --manifest-root ROOT -fl fold.json \
        --chroms chr1 chr3 chr6 --n-regions 4000 --n-markers 1500 -o poc.json
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from chrombpnet.multitask.train_multicell_torch import parse_manifest
from chrombpnet.cfdna import reference as ref
from chrombpnet.cfdna import deconvolve as dc


# ---- parallel per-cell reference column builder --------------------------
_PANEL = {}


def _init_worker(chrom, start, end):
    import pandas as pd
    _PANEL["regions"] = pd.DataFrame({"chr": chrom, "start": start, "end": end})


def _one_cell(args):
    acc_path, nuc_path = args
    regions = _PANEL["regions"]
    return (ref._region_sums(acc_path, regions),
            ref._region_sums(nuc_path, regions))


def build_R(manifest, regions, jobs):
    chrom = regions["chr"].to_numpy()
    start = regions["start"].to_numpy()
    end = regions["end"].to_numpy()
    tasks = [(m["acc_bw"], m["nuc_bw"]) for m in manifest]
    n, mm = len(regions), len(manifest)
    acc = np.zeros((n, mm)); nuc = np.zeros((n, mm))
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs, initializer=_init_worker,
                                 initargs=(chrom, start, end)) as ex:
            for j, (a, nu) in enumerate(ex.map(_one_cell, tasks, chunksize=2)):
                acc[:, j] = a; nuc[:, j] = nu
    else:
        _init_worker(chrom, start, end)
        for j, t in enumerate(tasks):
            a, nu = _one_cell(t)
            acc[:, j] = a; nuc[:, j] = nu
    return acc, nuc


# ---- benchmark -----------------------------------------------------------
def run_scenarios(R, cell_types, depths, n_active_list, n_trials, seed,
                  wbc_dominant=True):
    m = R.shape[1]
    rng = np.random.default_rng(seed)
    results = []
    scenarios = [("dirichlet", k) for k in n_active_list]
    if wbc_dominant:
        scenarios.append(("wbc_dominant", 5))
    for scen, n_active in scenarios:
        for depth in depths:
            trials = []
            for _ in range(n_trials):
                if scen == "wbc_dominant":
                    dom = int(rng.integers(m))
                    w = dc.sample_fractions(m, n_active, rng, dominant=dom,
                                            dom_frac=0.85, alpha=1.0)
                else:
                    w = dc.sample_fractions(m, n_active, rng, alpha=1.0)
                y = dc.simulate_observation(R, w, depth, rng)
                wh = dc.deconvolve(y, R, l2=0.0)
                rec = dc.recovery_metrics(w, wh)
                det = dc.detection_stats(w, wh)
                trials.append({**rec, **det})
            agg = {k: float(np.nanmean([t[k] for t in trials]))
                   for k in ["pearson", "rmse", "mae", "l1_error",
                             "sensitivity", "precision"]}
            agg.update(scenario=scen, n_active=n_active,
                       depth=(None if not np.isfinite(depth) else int(depth)),
                       n_trials=n_trials)
            results.append(agg)
    return results


def _fmt_depth(d):
    return "inf" if d is None else f"{d:g}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-m", "--manifest", required=True)
    ap.add_argument("--manifest-root", default=None)
    ap.add_argument("-fl", "--fold-json", default=None,
                    help="if given, restrict region panel to the fold's TEST chroms")
    ap.add_argument("--chroms", nargs="*", default=None,
                    help="explicit chromosomes for the region panel (overrides fold)")
    ap.add_argument("--n-regions", type=int, default=4000)
    ap.add_argument("--top-per-cell", type=int, default=200)
    ap.add_argument("--half-width", type=int, default=500)
    ap.add_argument("--n-markers", type=int, default=1500)
    ap.add_argument("--marker-method", default="specificity",
                    choices=["specificity", "variance"])
    ap.add_argument("--feature-sets", nargs="*", default=["acc", "nuc", "acc+nuc"])
    ap.add_argument("--depths", nargs="*", type=float,
                    default=[1e3, 1e4, 1e5, 1e6, float("inf")])
    ap.add_argument("--n-active", nargs="*", type=int, default=[2, 3, 5, 10])
    ap.add_argument("--n-trials", type=int, default=30)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-o", "--out-json", default=None)
    a = ap.parse_args()

    import os
    root = a.manifest_root or os.path.dirname(os.path.abspath(a.manifest))
    manifest = parse_manifest(a.manifest, root)

    if a.chroms:
        chroms = a.chroms
    elif a.fold_json:
        fold = json.load(open(a.fold_json))
        chroms = fold.get("test", fold.get("valid"))
    else:
        chroms = [f"chr{i}" for i in range(1, 23)]
    print(f"cell types: {len(manifest)} | panel chroms: {chroms}")

    print("building region panel ...")
    regions = ref.build_region_panel(manifest, chroms, top_per_cell=a.top_per_cell,
                                     n_regions=a.n_regions, half_width=a.half_width,
                                     seed=a.seed)
    print(f"  panel regions: {len(regions)}")

    print(f"building reference matrix (jobs={a.jobs}) ...")
    acc, nuc = build_R(manifest, regions, a.jobs)
    cell_types = [m["cell_type"] for m in manifest]

    # marker selection on the combined (acc+nuc) signal
    marker_idx = ref.select_markers(acc + nuc, a.n_markers, method=a.marker_method)
    blocks_raw = {"acc": acc[marker_idx], "nuc": nuc[marker_idx]}
    print(f"  markers kept: {len(marker_idx)} / {len(regions)}")

    report = {"n_cell_types": len(cell_types), "chroms": list(chroms),
              "n_regions": int(len(regions)), "n_markers": int(len(marker_idx)),
              "half_width": a.half_width, "feature_sets": {}}

    for fs in a.feature_sets:
        if fs == "acc":
            R = ref.normalize_reference(blocks_raw["acc"])
        elif fs == "nuc":
            R = ref.normalize_reference(blocks_raw["nuc"])
        elif fs == "acc+nuc":
            R = ref.normalize_reference(blocks_raw["acc"], blocks_raw["nuc"])
        else:
            raise ValueError(fs)
        cond = dc.condition_diagnostics(R)
        res = run_scenarios(R, cell_types, a.depths, a.n_active, a.n_trials, a.seed)
        report["feature_sets"][fs] = {"conditioning": cond, "results": res}

        print(f"\n=== feature set: {fs} ===")
        print(f"  R shape {R.shape} | cond(R)={cond['cond_number']:.1f} "
              f"| mean|col corr|={cond['mean_abs_col_corr']:.3f}")
        print(f"  {'scenario':<14}{'n_act':>6}{'depth':>9}"
              f"{'pearson':>9}{'rmse':>8}{'mae':>8}{'sens':>7}{'prec':>7}")
        for r in res:
            print(f"  {r['scenario']:<14}{r['n_active']:>6}{_fmt_depth(r['depth']):>9}"
                  f"{r['pearson']:>9.3f}{r['rmse']:>8.3f}{r['mae']:>8.3f}"
                  f"{r['sensitivity']:>7.2f}{r['precision']:>7.2f}")

    if a.out_json:
        with open(a.out_json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nwrote {a.out_json}")


if __name__ == "__main__":
    main()
