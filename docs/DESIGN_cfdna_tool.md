# Design doc — a new cfDNA deconvolution tool (regular WGS substrate)

_2026-07-07. Scope: **regular WGS cfDNA fragmentomics only** — methylation is deferred to a
documented slot (§6), not built now. Focus: what we can **train** with the data we already have.
Companion to `SUMMARY_cfdna_nucleosome.md`, `LITREVIEW_and_IDEAS.md`._

## 0. What this design fixes

The current tool deconvolves cfDNA using **one feature** (accessibility dip at chr1/3/6 ATAC
peaks) against **observed** scATAC tracks. It works for coarse composition (~90% hematopoietic,
validated) but fails at cancer / minor-fraction detection. The two diagnosed root causes:

1. **Single, depth-fragile feature.** Per-region dip magnitude at 0.28× → ~1.6 frags/region →
   Poisson noise swamps the cell-type signal.
2. **Reference is compartment-limited *and* observed-noisy.** ATAC only covers open chromatin;
   observed tracks carry shot noise; the model-denoised version (Role-b) collapsed to a universal
   pattern because it was trained on compartment-limited labels.

The corrected biology (from the design discussion): for the **accessibility-dip** signal, ATAC and
the discriminative cfDNA signal **do** overlap (both live at open regulatory elements) — so ATAC is
*appropriate*, just insufficient alone. The fix is **more features + a genome-wide feature that
escapes the dip's depth limit + a better-trained reference**, not abandoning ATAC.

## 1. Architecture (target end state)

```
  cfDNA WGS fragments (regular)
        │
        ├─ Feature block A: accessibility dip      → ref: scATAC 346-cell tracks (have)
        │    (composite GC-corrected coverage-dip at open regulatory marker sites)
        │
        ├─ Feature block B: gene-body WPS-FFT       → ref: expression atlas (Tabula Sapiens)
        │    (nucleosome-spacing periodicity in ~20k gene bodies; genome-wide, depth-robust)
        │
        ├─ Feature block C: fragment-length (DELFI)  → no ref (global, per-sample)
        │    (short:long ratio in bins; cancer vs healthy separator)
        │
        └─ Feature block D: end-motifs / preferred ends → no ref (global; nuclease activity)
                │
                ▼
        Deconvolution / classifier
          - composition: simplex NNLS over blocks A+B (per-cell-type fractions)
          - detection:   supervised classifier over A+B+C+D (cancer / tissue-of-origin)
```

Blocks A+B are **reference-based** (give interpretable cell-type fractions). Blocks C+D are
**reference-free global** features that the literature shows separate cancer from healthy on their
own — they don't need the atlas and are cheap to add.

## 2. Data assets we already have (no acquisition needed)

| Asset | What | Location |
|---|---|---|
| scATAC corpus | 346 cell-type accessibility + nuc-dyad tracks | `/mnt/data/jganbat/scatac_corpus/staged/` |
| FiLM model | `model_346_film_512long` (sequence→acc/nuc) | GCS `nucleosome_deconv/models/` |
| cfDNA cohort | Tao-2023: **25 normal / 26 CRC / 35 gastric**, WGS ~0.28× | `gs://prima-mente-sequencing-public/tao_2023...` |
| Expression atlas | Tabula Sapiens (~400+ cell types) — **needs pulling**, public | (to stage) |

The Tao-2023 cohort is the key unlock: **86 samples with disease labels** = a real supervised
training set for the detection task we've only tried unsupervised.

## 3. Trainable directions, ranked by feasibility-with-current-data

### T1 — Supervised fragmentomics classifier on Tao-2023 labels **(train first)**
- **Target:** normal vs CRC vs gastric (and CRC-vs-normal binary, the competitor's benchmark).
- **Features:** blocks A (dip at 346-cell markers) + **B (gene-body WPS-FFT)** + C (DELFI) + D
  (end-motifs). Block B is the competitor's winning feature and is *genome-wide + depth-robust* —
  the single most important addition.
- **Model:** regularized linear / gradient-boosted trees first (n=86 is small — heavy CV, nested
  folds, leakage control by patient). This directly targets the failure (cancer detection) using
  labels we already have, and is trainable **this week**.
- **Bar to clear:** Nat Commun 2024 CRC 84.7% at comparable depth. If block B alone gets us close,
  it confirms the "wrong feature, not dead method" diagnosis.
- **Risk:** small n; must not over-claim. Report CV honestly, hold a patient-disjoint test fold.

### T2 — Add an **expression head** to the sequence model → genome-wide reference generator
- **Why:** we train on genome-wide *regions* (peaks + ~700k nonpeak background per cell), but the
  *label* is ATAC-derived — **defined everywhere, informative only in the open compartment**
  (nuc-dyad track nonzero over just 0.1–11% of the genome; measured). In background/gene-body
  regions the label is ≈0 for all 346 cell types, so it teaches the model nothing that separates
  them there. That is why Role-b's predicted per-cell signatures correlated 0.985 — the label had
  no genome-wide cell-type information to learn. Adding a per-cell-type **expression** output head
  (label = Tabula Sapiens pseudobulk expression) is the fix: expression is **nonzero and
  cell-type-specific in every gene body**, giving the trunk its first *genome-wide informative*
  cell-type supervision signal. The model then predicts cell-type signatures **in gene bodies**,
  not just peaks — exactly what block B's reference needs, and exactly the compartment ATAC misses.
- **Train:** extend `ConditionedMultiCellModel` with a third head pair (expression profile/count);
  joint loss with acc+nuc. Reuse the 346-cell trunk; add matched scRNA where cell types align.
- **Payoff:** a *learned, genome-wide, cell-type-resolved* reference for block B that no observed
  assay provides. This is the novel, defensible core of the tool. Higher effort than T1.

### T3 — Fix the reference model's **spatial conditioning** (profile head, not counts)
- Role-b used the **count** head (one scalar/region) → captures magnitude, not shape. Retry with the
  **profile** head (per-base shape) as the reference target; shape is where cell-type spatial
  specificity lives. Cheaper than T2, uncertain (the FiLM-captures-magnitude limitation may persist);
  worth a quick ablation before committing to T2's architecture change.

### T4 — cfDNA self-supervised model (speculative, later)
- Train a sequence model to predict cfDNA fragment ends / WPS directly from sequence (masked
  objective over Tao fragments), then use embeddings as features. Only if T1–T3 plateau; flagged
  for completeness, not scheduled.

## 4. Non-model levers to fold in (cheap, do alongside T1)
- **Genome-wide markers**: chr1/3/6 → all chroms for block A (more sites → better minor-fraction).
- **GC + mappability correction** already in place; extend to blocks B/C.
- **Unknown component** in the simplex (absorbs neutrophils / unmodeled types) — keep.

## 5. Validation plan
- **Spike-in** (synthetic mixtures from R) for blocks A+B solver correctness — already have harness.
- **Held-out patient fold** for T1 (never mix a patient across train/test).
- **Benchmark table** vs Nat Commun 2024 (CRC 84.7%), cfSort — same Tao data where possible.
- **Ablation**: A only / A+B / A+B+C+D, to show each block's marginal contribution.

## 6. Deferred slot — methylation (NOT built now)
When ready: add block E (fragment-level methylation from the MEDIP-seq cfDNA) against a public
methylation atlas (Loyfer 2023 / cfSort). It plugs in as another reference-based block alongside
A+B and is the field's proven cancer-detection channel. Design leaves the block interface open;
implementation deferred per instruction.

## 7. Recommended sequence
1. **T1 + block B + genome-wide markers** — trainable now, targets the failure, uses labels in hand,
   benchmarkable against the field. Fastest path to a yes/no on "can regular-WGS fragmentomics detect
   cancer here."
2. If T1 shows block B carries the signal → **T2** (expression-head reference generator) to make the
   reference genome-wide and learned — the novel core.
3. **T3** as a cheap ablation before/around T2.
4. Methylation (block E) and T4 later.
