# Project summary — nucleosome model + cfDNA deconvolution

_Last updated 2026-07-07. Companion to `cfdna_deconvolution.md` (full log), `multicell_findings.md`
(model), `HANDOFF_2026-07-03.md` (restart guide)._

## One-paragraph version

We built a ChromBPNet-style multitask model that predicts per-cell-type ATAC accessibility and
nucleosome-dyad tracks across a **346-cell-type** scATAC corpus, then used those per-cell tracks as
a reference basis to deconvolve cell-free DNA (cfDNA) into cell-type fractions. **Healthy-plasma
composition works and is validated**: NC-PKU plasma resolves to ~90% hematopoietic, stable across
granularity, and the solver recovers known spike-in compositions exactly. **Minor-fraction / cancer
detection does not work** at the depth and marker set we have. Two attempts to inject the sequence
model into the deconvolution — an observed nucleosome channel and model-predicted references
("Role-b") — both hurt and were dropped. Observed accessibility tracks remain the best reference.

## What we built

- **Corpus:** 346 cell types staged (`/mnt/data/jganbat/scatac_corpus/staged/`), adult-only
  reference = 244 cells (drops fetal Domcke2020 + Pierce cell lines, which added noise since cfDNA
  is adult-tissue-derived).
- **Model:** `model_346_film_512long` — 512-filter fp32 trunk, FiLM per-cell-type conditioning,
  warmup→cosine LR. Converged (val 1327). The 512-fp32 path matters: bf16 on the 512-ch dilated
  conv hits a pathological cuDNN kernel (9× slower).
- **Deconvolution pipeline** (`chrombpnet/cfdna/`): Griffin/Snyder-style — GC-corrected composite
  coverage-dip over cell-type-specific marker site SETS, similarity-grouped references, simplex-
  constrained NNLS solver with an "unknown" component for missing cell types (e.g. neutrophils).

## What works (validated)

1. **TSS positive control** — composite cfDNA coverage/dyad/WPS around 20,033 protein-coding TSS
   shows the canonical Ulz/Snyder dip, 15× over matched random. The method + 0.28× data are alive.
2. **Healthy composition** — NC-PKU plasma → ~90% hematopoietic, biologically sensible, stable
   across merge granularity. Deep NC-PKU-10 → 0.937.
3. **Solver correctness** — spike-in recovery is exact after the l2-ridge bug fix (`l2=0`).

## What does not work (honest negatives)

1. **Cancer / minor-fraction detection** — CRC-32 reads ~100% blood. Sensitivity limit ~4–6% at
   chr1/3/6 panel depth × 0.28× coverage. Not enough signal, not a solver bug.
2. **Observed nucleosome-dyad channel** — too noisy at this depth; stacking acc+nuc HURTS recovery.
3. **Role-b (model-predicted references)** — the model regresses to a near-universal accessibility
   pattern (predicted per-cell-type signature corr 0.985 vs observed 0.549), collapsing all cells
   to one group. **Model limitation exposed:** FiLM conditioning captures per-cell-type *magnitude*,
   not *spatial pattern*, so predicted tracks lose the cell-type specificity that makes deconvolution
   possible. Observed (noisy) tracks are the better reference.

## The core tension

Everything that failed failed for the **same reason**: signal-to-noise at 0.28× coverage. Healthy
composition works because it's a coarse, high-fraction question (is this blood? yes). Cancer
detection, the nucleosome channel, and model denoising all need cell-type-specific structure that
survives per-region Poisson noise at ~1.6 fragments/region — and it doesn't. The model can't rescue
this because it learned magnitude, not spatial specificity.

## Untried levers (for the idea-generation phase)

- **Genome-wide markers** (chr1/3/6 → all chroms): more sites → better minor-fraction sensitivity,
  model-independent. The cheapest untried lever.
- **Fragment-length (DELFI) feature block**: orthogonal signal that separates cancer from healthy
  independent of coverage-dip.
- **Model profile head** as the Role-b target instead of counts (may carry spatial info counts lack;
  uncertain).
- **Fix spatial conditioning** in the model (architecture/training change; biggest effort).
- **Higher-depth cfDNA** samples (data acquisition, not method).
