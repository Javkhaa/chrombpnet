"""Block B: gene-body WPS-FFT nucleosome-phasing feature from cfDNA WGS.

Rationale (see docs/DESIGN_cfdna_tool.md): cfDNA coverage-dip at ATAC peaks is
depth-fragile and compartment-limited. The nucleosome-spacing *periodicity* in gene
bodies (Snyder 2016; Nat Commun 2024) is genome-wide and depth-robust: transcription
disrupts nucleosome arrays, so active genes show weak gene-body phasing and silent
genes strong, well-phased arrays. We read that out as the power of the WPS spectrum
in the nucleosome-repeat band.

Per protein-coding gene: per-base Windowed Protection Score (Snyder K=120) over a
TSS+downstream window (strand-oriented), detrend, Hanning-taper, rFFT. Features:
  - nratio: nucleosome-band power (period 150-250bp) / broadband (50-500bp) power
            -> depth-robust per-gene phasing index
  - aggregate power spectrum (QC: should peak at the ~185-200bp nucleosome repeat)

Validated on Tao-2023 cfDNA (0.28x-3x): clean 190bp spectral peak; nratio tracks
expression in the expected direction and is NOT coverage-confounded (Spearman
cov~nratio = +0.11). See docs/cfdna_deconvolution.md.

Sources accepted: local .parquet path or gs:// URI (streams chr/start/end only,
skipping the large sequence column).
"""
import argparse, os, time
import numpy as np

MAIN = set(f"chr{c}" for c in list(range(1, 23)) + ["X", "Y"])


def load_genes(path, win):
    """genebodies TSV: chrom, start, end, strand, name. Keep genes >= win, main chroms."""
    genes = []
    for line in open(path):
        c, gs, ge, strand, name = line.rstrip("\n").split("\t")
        gs, ge = int(gs), int(ge)
        if c in MAIN and (ge - gs) >= win:
            genes.append((c, gs, ge, strand, name.strip()))
    return genes


def load_fragments(src, minlen, maxlen, trim=0):
    """Return (chrom, start, end) numpy arrays for fragments in [minlen, maxlen].
    src is a local .parquet path or a gs:// URI (column-pruned streaming read).
    trim = bp hard-trimmed from EACH fragment end upstream; compensated here by
    extending both ends (start-=trim, end+=trim) before the length filter."""
    import pyarrow.parquet as pq
    if src.startswith("gs://"):
        import pyarrow.fs as fs
        gcs = fs.GcsFileSystem()
        handle = gcs.open_input_file(src[len("gs://"):])
    else:
        handle = src
    tb = pq.read_table(handle, columns=["chromosome", "start_position", "end_position"])
    ch = tb.column("chromosome").to_numpy(zero_copy_only=False)
    s = tb.column("start_position").to_numpy().astype(np.int64)
    e = tb.column("end_position").to_numpy().astype(np.int64)
    if trim:
        s = s - trim
        e = e + trim
    L = e - s
    k = (L >= minlen) & (L <= maxlen)
    return ch[k], s[k], e[k]


def index_by_chrom(ch, s, e):
    frby = {}
    for c in MAIN:
        m = ch == c
        if m.any():
            o = np.argsort(s[m])
            frby[c] = (s[m][o], e[m][o])
    return frby


def gene_wps(st, en, tss, win, half_k, maxlen):
    """Per-base WPS over [tss, tss+win) (absolute orientation)."""
    lo, hi = tss, tss + win
    j0 = np.searchsorted(st, lo - maxlen)
    j1 = np.searchsorted(st, hi)
    cs, ce = st[j0:j1], en[j0:j1]
    sel = ce > lo
    cs, ce = cs[sel], ce[sel]
    if len(cs) == 0:
        return None
    n = win
    dspan = np.zeros(n + 1)
    dend = np.zeros(n + 1)
    sa = np.clip((cs + half_k - lo).astype(int), 0, n)
    sb = np.clip((ce - half_k - lo).astype(int), 0, n)
    g = sb > sa
    np.add.at(dspan, sa[g], 1.0)
    np.add.at(dspan, sb[g], -1.0)
    for ep in (cs, ce):
        ea = np.clip((ep - half_k - lo).astype(int), 0, n)
        eb = np.clip((ep + half_k - lo).astype(int), 0, n)
        np.add.at(dend, ea, 1.0)
        np.add.at(dend, eb, -1.0)
    return np.cumsum(dspan[:n]) - np.cumsum(dend[:n])


def extract(src, genes, win=4000, wps_k=120, minlen=120, maxlen=220,
            nuc_lo=150, nuc_hi=250, broad_lo=50, broad_hi=500, detrend=601, trim=0, verbose=True):
    """Returns dict: nratio, namp, gnames, Pmean, freqs, peak, used, nfrag."""
    t0 = time.time()
    ch, s, e = load_fragments(src, minlen, maxlen, trim=trim)
    nfrag = len(s)
    if verbose:
        print(f"  fragments {minlen}-{maxlen}bp: {nfrag:,} ({time.time()-t0:.0f}s)", flush=True)
    frby = index_by_chrom(ch, s, e)
    del ch, s, e

    win_taper = np.hanning(win)
    freqs = np.fft.rfftfreq(win, d=1.0)
    period = np.where(freqs > 0, 1.0 / np.maximum(freqs, 1e-9), np.inf)
    nucband = (period >= nuc_lo) & (period <= nuc_hi)
    broadband = (period >= broad_lo) & (period <= broad_hi)
    ker = np.ones(detrend) / detrend
    half_k = wps_k // 2

    Psum = np.zeros(len(freqs))
    nratio, namp, gnames = [], [], []
    used = 0
    for c, gs, ge, strand, name in genes:
        if c not in frby:
            continue
        st, en = frby[c]
        tss = gs if strand == '+' else ge - win
        w = gene_wps(st, en, tss, win, half_k, maxlen)
        if w is None:
            continue
        if strand == '-':
            w = w[::-1]
        w = w - w.mean()
        w = w - np.convolve(w, ker, mode='same')
        W = np.abs(np.fft.rfft(w * win_taper)) ** 2
        Psum += W
        nb = W[nucband].mean()
        bb = W[broadband].mean() + 1e-9
        nratio.append(nb / bb)
        namp.append(nb)
        gnames.append(name)
        used += 1
    Pmean = Psum / max(used, 1)
    pk = period[np.argmax(Pmean * ((period >= 100) & (period <= 300)))]
    if verbose:
        print(f"  genes used: {used}  spectral peak: {pk:.1f}bp  ({time.time()-t0:.0f}s)", flush=True)
    return dict(nratio=np.array(nratio), namp=np.array(namp), gnames=np.array(gnames),
                Pmean=Pmean, freqs=freqs, peak=pk, used=used, nfrag=nfrag)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cfdna", required=True, help="local .parquet path or gs:// URI")
    ap.add_argument("--genes", required=True, help="genebodies TSV (chrom,start,end,strand,name)")
    ap.add_argument("--out", required=True, help="output .npz")
    ap.add_argument("--win", type=int, default=4000)
    ap.add_argument("--wps-k", type=int, default=120)
    ap.add_argument("--minlen", type=int, default=120)
    ap.add_argument("--maxlen", type=int, default=220)
    ap.add_argument("--trim", type=int, default=0,
                    help="bp hard-trimmed from each fragment end upstream; compensated by extending both ends")
    args = ap.parse_args()
    genes = load_genes(args.genes, args.win)
    print(f"genes (>= {args.win}bp): {len(genes)}", flush=True)
    r = extract(args.cfdna, genes, win=args.win, wps_k=args.wps_k,
                minlen=args.minlen, maxlen=args.maxlen, trim=args.trim)
    np.savez(args.out, **r)
    print(f"saved {args.out}  peak={r['peak']:.1f}bp  median_nratio={np.median(r['nratio']):.3f}", flush=True)


if __name__ == "__main__":
    main()
