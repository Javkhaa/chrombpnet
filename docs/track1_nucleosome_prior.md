# Track 1 — global nucleosome-occupancy prior for block B (NEGATIVE)

_2026-07-09. Tests whether residualizing block-B gene-body phasing (nratio) against a
global/cell-type-invariant nucleosome-occupancy prior sharpens the healthy-plasma
tissue-of-origin. Motivated by the "use existing bulk MNase for global/preserved
positioning" idea. Code: `/mnt/data/jganbat/nucprior/rebuild/prior_rank.py`._

## Setup

- **Priors** (from NucPosDB / Teif lab, hg38 stable-nucleosome tracks, 100bp bins):
  - Gaffney 2012 bulk B-cell **MNase** (`GSE36979`, 21.4M bins ≈ 70% genome, dense).
  - Song healthy **cfDNA** (`GSE81314`, 406k bins, sparse — in-domain but thin).
  - Per-gene prior = mean occupancy over the same TSS+4kb window block B uses.
- **Feature:** block-B nratio for NC-PKU-10 (regenerated: 190.5bp peak, median 3.857 — exact repro).
- **Tissue-of-origin:** Spearman(nratio, per-gene z-scored promoter accessibility) per cell
  type (244-cell corpus), rank ascending, hematopoietic enrichment by cell-type-name regex
  (71/244 = 29% heme baseline). Residualize nratio ~ prior (per-sample OLS) and re-rank.

## Result

| Variant | heme top-15 | top-30 | hypergeom p | mean r top-15 |
|---|---|---|---|---|
| Baseline (no prior) | 11/15 | 18/30 | 1.6e-4 | −0.040 |
| Resid vs Gaffney MNase | 10/15 | 19/30 | 3.2e-5 | −0.034 |
| Resid vs Song cfDNA | 6/15 | 18/30 | 1.6e-4 | −0.041 |
| Resid vs both | 6/15 | 18/30 | 1.6e-4 | −0.036 |

**Decisive:** `corr(nratio, Gaffney occ) = +0.037` (n=17,364); `corr(nratio, Song occ) = −0.018`.

## Verdict

**No help.** Bulk nucleosome **occupancy level** is nearly orthogonal to block-B **phasing**
(nratio) — r=0.037 — so there is no shared baseline to residualize out. The Gaffney p
"improvement" (1.6e-4→3.2e-5) is one cell type crossing the top-30 boundary, uncorroborated
(top-15 *drops*, mean |r| *weakens*) = noise. Song is too sparse (6,335 genes) and hurts.

## What this does / doesn't rule out

- **Rules out** occupancy-*level* residualization of block B (cheap, done).
- **Does NOT test** periodicity residualization — needs a *per-base* positioning/WPS prior;
  the 100bp stable-nuc tracks can't resolve the 190bp band (Nyquist ≈ 200bp). That would
  require **reprocessing raw bulk MNase BAMs → per-base WPS** (a real build).

## Decision

Bank the negative; **do not** invest in raw-MNase reprocessing now. Block B already recovers
the immune tissue-of-origin at p~1e-4 with NO prior, and it is a *composition* feature, not the
sensitivity engine. The r=0.037 orthogonality suggests even a better prior has modest upside.
Redirect to **methylation (FACS WGBS reference)** for rare-fraction sensitivity. Keep the
raw-MNase per-base periodicity prior as a documented future item if block B is productionized.
