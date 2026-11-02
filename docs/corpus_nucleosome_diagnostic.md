# scATAC corpus — nucleosome positioning: diagnosis, differential calling, deviation eval

_2026-07-31. Why the multi-cell nucleosome head plateaus at ~0.37, whether cell-type-specific
nucleosome modeling is viable, and a Role-b-proof way to measure it. Model
`run/model_346_film_512long.pt`, test chroms chr1/chr3/chr6, on the pseudobulk per-cell-type
`nucleosome_dyad.bw` tracks (integer dyad counts). Code: `corpus_nuc_diagnostic.py`,
`call_diff_nucleosomes.py`, `eval_deviation.py`; existing eval `chrombpnet-eval-multicell`._

> **Status (2026-11-02).** This file records the 2026-07-31 baseline. The cfDNA-trunk transfer
> that it proposes was built, and it led to the hybrid per-cell nucleosome head. The 841 calls
> come from a panel with fetal cells and cell lines. `adult_differential_nucleosomes.md`
> re-scores them in adult cells only. For the current results, see
> `corpus_nucleosome_report.html`.

## 1. The 0.37 plateau is model-limited, not target-limited

Ran the existing multicell eval on all 346 cells (`run/eval_346_test.json`):

| nucleosome head (346 cells, test) | value |
|---|---|
| profile r, smoothed (= the "0.37") | 0.369 |
| profile r, raw (sparse dyad) | 0.063 |
| **split-half ceiling** (achievable max) | **0.770** |
| **model / ceiling** | **0.47** |
| count r | 0.636 |

The target is real (dyad periodicity 200 bp; independent estimate of the ceiling 0.80). The
model reaches only **47% of the achievable ceiling** — large headroom. For contrast, the
accessibility head on the same model reaches 0.54 / 0.82 = **66%** of its ceiling, so the trunk
can learn positioning-scale structure; the nucleosome head specifically lags.

## 2. Cell-type-specific positioning genuinely exists (depth-controlled)

New measurement (`corpus_nuc_diagnostic.py`) the codebase never had: cross-cell-type
nucleosome correlation, controlled for depth. If two cell types had identical positioning,
their cross-correlation would be ≈ √(ceiling_a·ceiling_b) (attenuated only by sampling noise).

- deep, well-measured pairs: **cross_r / √(ceil·ceil) = 0.68** — meaningfully below 1.0.
- So **~⅓ of the reproducible nucleosome positioning is genuinely cell-type-specific**; ~⅔ is
  shared backbone. This is not the Role-b "all invariant" collapse — there is real contrast.

## 3. Differential-nucleosome calling (the localized signal)

`call_diff_nucleosomes.py` — one-vs-rest per cell type on the pseudobulk tracks. Profiles are
normalized to distributions (occupancy removed → tests POSITIONING). A window is called
differential for cell c when c's departure from the leave-one-out consensus is a robust-z
outlier (z ≥ 3) among the panel AND exceeds c's own split-half noise floor. 20-cell panel
(fetal neuron, cardiac, retina, hepatic, renal, PBMC B/plasma, GM12878, MCF7, smooth muscle…):

- **841 cell-type-specific nucleosome sites** (5.4% of well-measured cell-windows), median 30
  per cell.
- **repositioning shift: median 30 bp** (90th pct ~78 bp) — nucleosomes moving tens of bp
  between cell types (e.g. a GM12878 site shifted −54 bp, z = 9.4).
- The ~5% strong-outlier rate reconciles with the diffuse ⅓ aggregate contrast: most windows
  differ a little, few are strong single-cell outliers.
- Output: per-cell BEDs `run/diffnuc/*.diffnuc.bed` (chrom, window, cell, z-effect, strand,
  shift_bp, z, r_self) + cached masks for the eval.

## 4. Deviation eval — the Role-b-proof metric

Absolute profile corr scores mostly the shared backbone. `eval_deviation.py` scores the model
on the **deviation**: `dev(c,w) = norm(profile_c) − norm(consensus_-c)`, observed vs predicted,
consensus built identically on each side. On the 20-cell panel:

| metric (positioning shape, occupancy removed) | value |
|---|---|
| absolute-profile r (comparable windows) | 0.108 |
| **deviation r — all comparable windows** | **0.030** |
| **deviation r — 841 called differential sites** | **0.217** |
| **shift-direction r — called sites** | **0.067** |

Reading:
- **Overall the model is ~backbone-collapsed on positioning** (deviation r 0.03) — it barely
  captures the cell-type-specific component, confirming the design's Role-b warning that the
  absolute number is inflated by the invariant backbone.
- **A weak but real cell-type signal exists** at the strongest differential sites (0.217) — the
  FiLM conditioning does something, it's not a total collapse.
- **The model does not get the repositioning direction** (shift-dir 0.067 ≈ 0) — it senses
  "something differs here" but not which way the nucleosome moves.

## Bottom line + next

- **Cell-type-specific nucleosome modeling is viable**: the data supports 0.77, carries a real
  ~⅓ cell-type-specific component, and 841 concrete differential sites with ~30 bp shifts exist.
- **The current model captures almost none of it** (deviation 0.03 overall, 0.22 at strong
  sites, shift-direction ~0). This is the baseline.
- **Lever:** transfer the validated cfDNA genome-wide positioning trunk
  (`nucprior/models/nuc_full.pt`) into the corpus nucleosome head (and/or up-weight the nuc
  profile loss, add head capacity). Success is measured here — **deviation r and shift-direction
  r at the called sites must rise**, not the backbone-inflated absolute 0.37.

## Assets

- `corpus_nuc_diagnostic.py` · `run/nuc_contrast_diagnostic.json`
- `call_diff_nucleosomes.py` · `run/diffnuc/*.diffnuc.bed` · `run/diffnuc/summary.json`
- `eval_deviation.py` · `run/diffnuc/deviation_eval.json`
- existing eval: `chrombpnet-eval-multicell` · `run/eval_346_test.json`
