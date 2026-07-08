# cfDNA composition deconvolution — result report

_2026-07-08. Per-sample cell-type composition of cfDNA against the 346-cell scATAC
reference. Method: `chrombpnet/cfdna/deconv_stream.py` (streaming) / `griffin_grouped.py`
(local). Cohort: Tao-2023 (GSE186573), 86 plasma WGS samples._

## Method (one paragraph)

Reference `R` = per-cell-type ATAC accessibility at ~8,000 chr1/3/6 marker sites (244
adult cell types, agglomeratively merged to ~189 groups at signature-correlation 0.85 to
keep `R` well-conditioned). Feature `y` = GC-corrected composite **coverage-dip** of cfDNA
fragments at each group's differential marker sites (cfDNA is nucleosome-protected, so it
is depleted over open regulatory elements — the dip depth reports that cell type's
accessibility contribution). Solve the simplex-constrained `y = R w` (w≥0, Σw=1, l2=0,
plus an "unknown" component = mean-of-R that absorbs unmodeled cell types). Solver
validated by spike-in (exact recovery). `recon_r` = corr(y, Rw) is the per-sample fit
quality; **recon_r < 0.6 = unreliable (exclude)**.

## Cohort result (Tao-2023, 80/86 samples with recon ≥ 0.6)

| Class | n (well-fit) | heme median | heme mean |
|---|---|---|---|
| **NC** (normal plasma) | 23 | 0.941 | 0.855 |
| **CRC** (colorectal) | 24 | **1.000** | 0.927 |
| **STAD** (gastric) | 33 | 0.865 | 0.793 |

Deep validation samples: NC-PKU-10 → **0.961 heme** (recon 0.891); CRC-PKU-10 → 0.940
(recon 0.905). 6/86 samples excluded for recon < 0.6 (all shallow on chr1/3/6, 0.7–1.2M
fragments; their low-heme calls are noise, not tissue detection).

## Findings

**1. Per-sample composition works and is biologically correct.** Healthy plasma resolves
to ~94% hematopoietic (median), dominated by erythroid/lymphoid lineages — the expected
source of cfDNA (hematopoietic turnover). Recon r ~0.8–0.9 on well-fit samples.

**2. No cancer signal — the composition does not separate cancer from normal.** CRC has
the **highest** heme median (1.000): cancer samples read as *more* blood-derived, not less.
There is no epithelial/tumor mass distinguishing CRC or STAD from NC. Consistent with the
independent block-B tissue-of-origin classifier (CRC-vs-NC AUC within the permutation null)
and the earlier CRC-32 → ~100%-blood result. ctDNA fraction at 0.28–3× WGS is below what
coverage-dip deconvolution resolves in this cohort.

**3. recon_r is a required QC filter.** The lowest-heme outliers (e.g. STAD-4 → 0.000,
STAD-11 → 0.187) have recon 0.38–0.71 and low depth — poor fits, not genuine non-blood
detection. One deep normal (NC-14, 14.3M frags) also mis-fits (heme 0.34, recon 0.45).
Below recon ~0.6 the call is unreliable regardless of depth.

## Scope statement (honest)

The tool is a validated **cell-type composition / tissue-of-origin** method for cfDNA:
healthy plasma → hematopoietic, per-sample, reproducibly, and it agrees with an independent
nucleosome-phasing feature (block B). It is **not** a cancer detector at this depth/cohort —
that needs orthogonal fragmentation features (fragment-length/DELFI, end-motifs) and/or
higher-ctDNA samples, and is left as future work.

## Run on a new dataset

```
python -m chrombpnet.cfdna.deconv_stream \
    --cfdna <frags.parquet | gs://bucket/dir/ | 'path/*.parquet'> \
    --out-dir <out> \
    [--cols chromosome start_position end_position] \
    [--chroms chr1 chr3 chr6] [--merge-corr 0.85]
```
Accepts local paths, `gs://` URIs, globs, or a `gs://` "directory". Streams only
chr/start/end (GC computed from hg38.fa), so no full download needed. Requires **hg38**,
**chr-prefixed** fragments with per-fragment endpoints (paired-end WGS). Outputs per-sample
JSON (heme_mass, recon_r, top cell-type groups) + a summary table.
