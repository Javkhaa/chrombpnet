"""Streaming cfDNA composition deconvolution (dip-based), local or gs://.

Same method as griffin_grouped (cached accessibility reference, merge-corr grouping,
GC-corrected coverage-dip at cell-type marker sites, l2=0 simplex solver + unknown
component, hematopoietic rollup) but reads ONLY chr/start/end and computes per-fragment
GC from the reference genome — so it works on gs:// URIs without downloading the large
`sequence` column, and on any parquet with chromosome/start/end columns.

Validated on Tao-2023 (docs/cfdna_deconv_report.md): reproduces the full-parquet
pipeline (NC-10 -> 96% heme, recon r~0.89).

Input fragments: parquet with columns chromosome / start_position / end_position (override
names with --cols), hg38, chr-prefixed. Runs one file, many files, or a gs:// "directory".
Per-sample JSON + a printed summary table. recon_r < ~0.6 = unreliable fit (flag/exclude).
"""
import argparse, glob, json, os, time
import numpy as np
import pyarrow.parquet as pq
import pyfaidx
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from chrombpnet.cfdna import deconvolve as dc
from chrombpnet.cfdna.griffin_lineage import lineage_of, region_profiles, dip_of
from chrombpnet.cfdna.griffin_grouped import is_heme, group_label
import pandas as pd


def build_reference(cache, merge_corr, markers_per_group):
    z = np.load(cache, allow_pickle=True)
    accA = z["accA"]; cts = list(z["cts"])
    regions = pd.DataFrame({"chr": z["rchr"], "center": z["rc"].astype(np.int64)})
    col = accA / (accA.sum(0, keepdims=True) + 1e-12)
    C = np.clip(np.corrcoef(np.log1p(col * 1e4).T), -1, 1)
    D = 1.0 - C; np.fill_diagonal(D, 0.0)
    clusters = fcluster(linkage(squareform(D, checks=False), method="average"),
                        t=1.0 - merge_corr, criterion="distance")
    groups = {}
    for i, g in enumerate(clusters): groups.setdefault(g, []).append(i)
    gids = sorted(groups); G = len(gids)
    Agrp = np.zeros((len(regions), G)); gmembers = []
    for gi, g in enumerate(gids):
        idx = groups[g]; Agrp[:, gi] = accA[:, idx].mean(1); gmembers.append([cts[i] for i in idx])
    cg = Agrp / (Agrp.sum(0, keepdims=True) + 1e-12)
    rm = cg.mean(1, keepdims=True); rs = cg.std(1, keepdims=True) + 1e-12
    Zm = (cg - rm) / rs
    markers = {gi: np.argsort(Zm[:, gi])[::-1][:markers_per_group] for gi in range(G)}
    R = np.zeros((G, G))
    for k in range(G): R[k] = Agrp[markers[k]].mean(0)
    Rn = R / (R.sum(0, keepdims=True) + 1e-12)
    heme_g = [any(is_heme(m) for m in gmembers[gi]) for gi in range(G)]
    labels = [group_label(m) for m in gmembers]
    return dict(regions=regions, Rn=Rn, markers=markers, heme_g=heme_g, labels=labels, G=G)


def build_gc(hg38, chroms):
    fa = pyfaidx.Fasta(hg38); cumgc = {}; sizes = {}
    for c in chroms:
        s = np.frombuffer(str(fa[c][:]).upper().encode(), dtype=np.uint8)
        cumgc[c] = np.concatenate([[0], np.cumsum(((s == 71) | (s == 67)).astype(np.int64))])
        sizes[c] = len(s)
    return cumgc, sizes


def _open(path):
    if path.startswith("gs://"):
        import pyarrow.fs as fs
        return fs.GcsFileSystem().open_input_file(path[len("gs://"):])
    return path


def load_stream(path, cols, chroms, cumgc, sizes, rng):
    tb = pq.read_table(_open(path), columns=list(cols))
    ch = tb.column(cols[0]).to_numpy(zero_copy_only=False)
    s = tb.column(cols[1]).to_numpy().astype(np.int64)
    e = tb.column(cols[2]).to_numpy().astype(np.int64)
    L = e - s
    keep = (L >= 100) & (L <= 250) & np.isin(ch, chroms)
    ch, s, e = ch[keep], s[keep], e[keep]
    if len(s) == 0:
        return {}, 0
    def fgc(c, ss, ee):
        cg = cumgc[c]; LL = ee - ss
        return np.where(LL > 0, (cg[np.clip(ee, 0, sizes[c])] - cg[np.clip(ss, 0, sizes[c])]) / np.maximum(LL, 1), 0.5)
    parts = [(c, s[ch == c], e[ch == c]) for c in chroms if (ch == c).any()]
    gcf = np.concatenate([fgc(c, ss, ee) for c, ss, ee in parts])
    chc = np.concatenate([np.full(len(ss), c) for c, ss, ee in parts])
    sc = np.concatenate([ss for _, ss, _ in parts]); ec = np.concatenate([ee for _, _, ee in parts])
    Lc = ec - sc
    N = 120000; Ls = Lc[rng.integers(0, len(Lc), N)]; egc = np.empty(N)
    for i in range(N):
        c = chroms[rng.integers(0, len(chroms))]; ln = int(Ls[i])
        p = int(rng.integers(3_000_000, sizes[c] - 3_000_000)); egc[i] = (cumgc[c][p + ln] - cumgc[c][p]) / ln
    bins = np.linspace(0, 1, 51)
    oh, _ = np.histogram(gcf, bins=bins, density=True); eh, _ = np.histogram(egc, bins=bins, density=True)
    wc = np.where(oh > 1e-6, eh / oh, 0.0)
    fw = wc[np.clip(np.digitize(gcf, bins) - 1, 0, len(wc) - 1)]; fw /= (fw.mean() + 1e-9)
    frby = {}
    for c in chroms:
        m = chc == c
        if m.any():
            o = np.argsort(sc[m]); frby[c] = (sc[m][o], ec[m][o], fw[m][o])
    return frby, len(sc)


def deconvolve_one(path, ref, cols, chroms, cumgc, sizes, W, BIN, rng):
    frby, nfr = load_stream(path, cols, chroms, cumgc, sizes, rng)
    if nfr == 0:
        raise ValueError("no fragments on target chroms — wrong genome build or chrom naming?")
    prof = region_profiles(frby, ref["regions"], W, BIN)
    G = ref["G"]
    y = np.array([dip_of(prof[ref["markers"][k]].sum(0)) for k in range(G)])
    yn = np.clip(y, 0, None); yn = yn / (yn.sum() + 1e-12)
    u = ref["Rn"].mean(1, keepdims=True); Rw = np.hstack([ref["Rn"], u])
    w = dc.deconvolve(yn, Rw, l2=0.0, iters=8000)
    recon = float(np.corrcoef(yn, Rw @ w)[0, 1])
    heme = float(sum(w[gi] for gi in range(G) if ref["heme_g"][gi]))
    top = [(ref["labels"][int(k)], float(w[int(k)])) for k in np.argsort(w[:G])[::-1][:8]]
    return dict(nfrag=int(nfr), recon_r=recon, heme_mass=heme, unknown=float(w[G]), top=top)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cfdna", nargs="+", required=True,
                    help="fragment parquet path(s): local, gs://, glob, or a gs:// prefix/dir")
    ap.add_argument("--cache", default="/mnt/data/jganbat/scatac_corpus/run/griffin_AN_adult346.npz",
                    help="cached accessibility reference (accA, rchr, rc, cts)")
    ap.add_argument("-g", "--genome", default="/mnt/data/jganbat/scatac_corpus/refs/hg38.fa")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--chroms", nargs="*", default=["chr1", "chr3", "chr6"])
    ap.add_argument("--cols", nargs=3, default=["chromosome", "start_position", "end_position"],
                    metavar=("CHROM", "START", "END"), help="fragment column names")
    ap.add_argument("--merge-corr", type=float, default=0.85)
    ap.add_argument("--markers-per-group", type=int, default=400)
    ap.add_argument("--W", type=int, default=1000)
    ap.add_argument("--bin", type=int, default=10)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    # expand inputs (local globs + gs:// dirs)
    paths = []
    for c in a.cfdna:
        if c.startswith("gs://") and not c.endswith(".parquet"):
            import pyarrow.fs as fs
            gcs = fs.GcsFileSystem()
            fi = gcs.get_file_info(fs.FileSelector(c[len("gs://"):].rstrip("/")))
            paths += ["gs://" + f.path for f in fi if f.path.endswith(".parquet")]
        elif "*" in c:
            paths += sorted(glob.glob(c))
        else:
            paths.append(c)
    print(f"samples: {len(paths)}", flush=True)

    print("building reference ...", flush=True)
    ref = build_reference(a.cache, a.merge_corr, a.markers_per_group)
    print(f"  {ref['G']} groups; {sum(ref['heme_g'])} hematopoietic", flush=True)
    cumgc, sizes = build_gc(a.genome, a.chroms)
    rng = np.random.default_rng(0)

    rows = []
    for i, p in enumerate(paths):
        tag = os.path.basename(p).split(".parquet")[0].split("-wgs")[0]
        outp = os.path.join(a.out_dir, f"{tag}.json")
        if os.path.exists(outp):
            r = json.load(open(outp))
        else:
            t0 = time.time()
            try:
                r = deconvolve_one(p, ref, a.cols, a.chroms, cumgc, sizes, a.W, a.bin, rng)
            except Exception as ex:
                print(f"[{i+1}/{len(paths)}] {tag}: ERROR {type(ex).__name__} {ex}", flush=True); continue
            r["cfdna"] = tag
            json.dump(r, open(outp, "w"), indent=2)
            r["_t"] = time.time() - t0
        rows.append(r)
        flag = "  <-- low recon, unreliable" if r["recon_r"] < 0.6 else ""
        print(f"[{i+1}/{len(paths)}] {tag}: heme={r['heme_mass']:.3f} recon={r['recon_r']:.3f} "
              f"nfr={r['nfrag']/1e6:.1f}M{flag}", flush=True)

    if rows:
        h = np.array([r["heme_mass"] for r in rows]); rc = np.array([r["recon_r"] for r in rows])
        ok = rc >= 0.6
        print(f"\n=== summary: {len(rows)} samples ({ok.sum()} with recon>=0.6) ===")
        print(f"  hematopoietic mass: median {np.median(h[ok]) if ok.any() else float('nan'):.3f} "
              f"(well-fit only), overall mean {h.mean():.3f}")
        print("  lowest-heme well-fit samples (candidate non-blood signal):")
        for r in sorted([r for r in rows if r['recon_r'] >= 0.6], key=lambda r: r['heme_mass'])[:6]:
            print(f"    {r['heme_mass']:.3f} heme  recon={r['recon_r']:.3f}  {r['cfdna']}")


if __name__ == "__main__":
    main()
