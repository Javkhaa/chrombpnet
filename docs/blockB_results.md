# Block B results — gene-body WPS-FFT nucleosome phasing (2026-07-07)

First build + validation of the DESIGN doc's block B (gene-body nucleosome-spacing
periodicity) on Tao-2023 cfDNA. Code: `chrombpnet/cfdna/genebody_wps.py`.
Design rationale: `docs/DESIGN_cfdna_tool.md`.

## What block B is

Per protein-coding gene, compute per-base WPS (Snyder K=120, long fragments 120-220bp)
over a 4kb TSS+downstream window (strand-oriented), detrend, Hanning-taper, rFFT. The
**nratio** feature = nucleosome-band power (period 150-250bp) / broadband (50-500bp).
Biology: transcription disrupts nucleosome arrays, so active genes show weak gene-body
phasing, silent genes strong, well-phased arrays. Unlike the ATAC coverage-dip feature,
this is **genome-wide (gene bodies, not peaks)** and **depth-robust** (a periodicity
integrated over 4kb, not a per-region count).

## Validation on local Tao samples (NC-PKU-10 normal, CRC-PKU-32 cancer)

**1. Feature is extractable at this depth.** Aggregate WPS power spectrum peaks at
**190.5 bp** — the human nucleosome repeat length — clean bump (185bp=0.86, 200bp=0.84,
falling to 0.15/0.44 at 147/220bp). Reproduces across NC and CRC and across all 86
samples processed so far (every one peaks at 190bp).

**2. nratio behaves biologically (curated gene sets, NC-10).** median nratio: housekeeping
3.60 < blood 3.84 < silent(blood-off) 4.24. Active genes phase less, silent genes more —
the expected transcription-disruption direction.

**3. Not a coverage artifact.** Spearman(coverage, nratio) = **+0.11** across 17.5k genes
(weakly *positive*). A coverage confound would be negative; the silent>active phasing
signal is real biology (nucleosome-protected closed chromatin has both more long fragments
and stronger phasing), not depth.

**4. Tissue-of-origin: healthy plasma -> hematopoietic (the key result).**
Expression proxy = per-cell-type promoter accessibility from the local 346-cell scATAC
corpus (egress-free; Tabula Sapiens unreachable from this box). For the contributing cell
type, corr(cfDNA gene-body phasing, promoter accessibility) is negative (active->low
phasing). Ranking cell types by that correlation, using **per-gene z-scored (cell-type-
specific)** accessibility:

| Top-N | immune cell types | baseline |
|---|---|---|
| top-10 | **10/10 (100%)** | 38% |
| top-15 | **15/15 (100%)** | 38% |
| top-20 | 17/20 (85%) | 38% |
| top-30 | 22/30 (73%) | 38% |

Hypergeometric enrichment top-30: **p = 3.3e-5**. The top-15 is entirely immune
(bone-marrow NK/CD8/CD4/preB/Mono, PBMC subsets, tissue-resident macrophages + T cells).
Only 3 non-immune in the top-20 (cardiac, renal tumor). Raw (non-z-scored) accessibility
gives a muddier ranking (16/30) — z-scoring removes the shared housekeeping axis and
isolates cell-type specificity.

**Significance:** block B recovers the known ~hematopoietic composition of healthy plasma
via a feature **fully independent** of the ATAC coverage-dip that produced the earlier
~90%-hematopoietic result. Two orthogonal features now agree on the ground-truth biology.
This is (to our knowledge) the first scATAC-referenced gene-body-nucleosome-phasing
deconvolution of cfDNA.

## Limitations / open

- Promoter-accessibility is a noisy expression proxy; a real RNA atlas (Tabula Sapiens)
  would likely sharpen further but is unreachable from this box.
- Per-gene nratio correlations are individually small (~0.05); power comes from ranking
  over ~17.5k genes.
- Cancer detection not yet tested — that is T1 (supervised classifier over all 86 samples).

## Status of the scaled run (T1 prep)

`blockB_all/` : per-sample block-B npz for all 86 Tao samples (26 CRC / 25 NC / 35 STAD),
streamed column-pruned from GCS (chr/start/end only). Running in background (~2h).
Next: assemble sample x gene feature matrix, train T1 classifier (normal vs CRC vs STAD)
with patient-disjoint CV, benchmark vs Nat Commun 2024 (CRC 84.7%).

## T1 cancer classifier — NEGATIVE at this cohort/depth (2026-07-08)

All 86 samples extracted (25 NC / 26 CRC / 35 STAD). Patient-disjoint 5-fold CV,
three feature representations of block B:

| Representation | CRC vs NC | STAD vs NC |
|---|---|---|
| 13-lineage cell-type affinity | 0.680 | 0.511 |
| full 244-cell affinity (L1/L2) | 0.623 | 0.457 |
| PCA-20 of raw per-gene nratio | 0.560 | 0.581 |
| **permutation null (CRC-NC)** | **0.515 ± 0.121 (max 0.806)** | — |

**Block B alone does not detect cancer here.** Richer representations are no better
(lower), and the permutation null is so wide (±0.121; max shuffled AUC 0.806) that
CRC-vs-NC 0.68 is ~1.4σ above chance — not significant. STAD is flat chance. Class-mean
lineage affinities are near-identical across NC/CRC/STAD. Consistent with the earlier
finding (CRC-32 -> ~100% blood): low ctDNA fraction at 0.28-3x WGS + n=86 is
underpowered for fragmentation-only cancer detection. Nat Commun 2024's CRC 84.7% used
744 subjects and stacked features — power and cohort size we don't have with block B alone.

**Takeaway:** block B is a validated *composition / tissue-of-origin* feature (its real
win), not a standalone cancer detector. Cancer detection needs the orthogonal global
fragmentation blocks (C: fragment-length/DELFI, D: end-motifs) stacked on top, and/or
higher-ctDNA samples. Diagnostic: `scratchpad/blockB_T1_diagnostic.py`.

## Reproduce

```
# gene bodies (protein-coding, chr-prefixed, >=4kb): built from gencode.gtf.gz -> genebodies_pc.tsv
python -m chrombpnet.cfdna.genebody_wps --cfdna <sample.parquet|gs://...> \
    --genes genebodies_pc.tsv --out <sample>.npz
# tissue-of-origin + z-scored ranking: scratchpad/blockB_too_v2.py
```
