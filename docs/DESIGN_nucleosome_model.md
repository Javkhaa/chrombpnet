# Phase 1 — sequence → nucleosome model (design spec)

_2026-07-24. Reframe: the deliverable is a **nucleosome occupancy model**, not a cancer
detector. cfDNA deconvolution becomes a downstream application, not the thing being
debugged. Training data: Snyder 2016 deep healthy plasma (GSE71378), realigned to hg38._

## 1. Why this, and why now

Every failure so far sits downstream of a missing model, not a missing feature:

- The dip deconvolution used **promoter accessibility as a proxy** for a fragmentation
  phenomenon. That misspecification is the most likely source of the σ=0.121 spread in the
  block-B permutation null — the noise floor was bias, not shot noise.
- Role-b failed because we asked a chromatin model to emit accessibility *references*.
- Track 1 died on exactly this gap: we could not test periodicity residualization because
  **no per-base positioning prior existed**. This model is that prior.

Decisive property: a nucleosome model is **evaluable on its own terms** — held-out
chromosomes, per-base correlation, NRL recovery. No cohort, no ctDNA fraction, no n=86,
no clinical label. We iterate against clean metrics daily instead of waiting on 1.4σ
p-values. Phases 1–2 are valuable even if phase 3 fails, which was never true of the
cancer-detection framing.

## 2. Training target — predict endpoints, derive WPS

Two candidates were considered. **We predict per-base fragment-endpoint density, not WPS.**

| | Endpoint density (chosen) | Adjusted WPS |
|---|---|---|
| Range | non-negative counts | signed, continuous |
| Loss | multinomial profile + Poisson count — **ChromBPNet's existing loss, unchanged** | MSE/Huber, new code |
| Depth confound | handled by the count/profile split | needs running-median detrending |
| Interpretation | nuclease cleavage propensity | derived summary of the above |

The insight making this work: **cfDNA fragment endpoints are the direct analogue of Tn5
insertion sites.** They are cut sites of apoptotic nucleases (DFFB/DNASE1L3) whose
accessibility is modulated by nucleosome protection. So ChromBPNet's entire framework —
profile multinomial + count Poisson + a bias model for enzyme sequence preference — transfers
1:1 with no reformulation.

WPS is then a **deterministic function** of predicted endpoints and the fragment-length
distribution, so it stays available as a derived quantity for evaluation and for the block-B
phasing feature, without being an awkward regression target.

Concretely, per 1 kb output window:
- **profile head** — per-base fragment endpoint density (5′ and 3′ ends pooled).
- **count head** — log total endpoints in the window (local coverage / depth).

## 3. Bias model — and a payoff

ChromBPNet's Tn5 bias module maps to **cfDNA nuclease sequence preference at cut sites**.
This is not a nuisance-only parallel: the sequence preference of cfDNA ends is exactly the
published **end-motif** signal (Jiang et al.), which was *block D* in the old multi-block
fragmentomics design.

So the bias model and the end-motif feature are the same object. Fitting the bias model
gives us block D for free, properly deconvolved from the positioning signal rather than
confounded with it.

Fit the bias model on the highest-entropy background (unprotected / high-nucleosome-free
regions) exactly as ChromBPNet fits Tn5 bias on background regions, then hold it frozen
while training the positioning model.

## 4. Data

| Sample | Coverage | Source |
|---|---|---|
| BH01 | 95.9x | SRR2129993, ENA (GSE71378 raw arm) |
| IH01 | 101.5x | SRR2130050 |
| IH02 | 21.6x | SRR2130051 — held in reserve |
| IA05–08 | 10–13x each | reserve |

**~200x healthy plasma**, aligned natively to hg38 with Parabricks fq2bam against
`/mnt/data/jganbat/scatac_corpus/refs/hg38.fa` — the same reference the 346-cell corpus
uses, so fragment coordinates and accessibility tracks share one space with no
reconciliation step. Fragments: proper pairs, MAPQ≥30, duplicates removed, 50–250 bp.

**Independent validation set:** Snyder's own ~13M published dyad calls
(GEO `.bb`, hg19 → liftOver hg38). External check that our pipeline reproduces the
reference result — the class of positive control that caught the block-B site-definition bug.

## 5. Architecture — reuse, don't rebuild

`chrombpnet/training/models/multitask_nucleosome_torch.py` already provides
`ProfileCountHead` and the dilated-conv trunk; `MultiCellMultiTaskModel` already provides
per-cell-type heads over a shared trunk for phase 3.

Phase 1 therefore ships **a new label track and dataset, not a new model**:
- trunk unchanged (`filters=512`, `n_dil_layers=8`, `inputlen=2114`, `outputlen=1000`);
- one `ProfileCountHead` on the endpoint-density target;
- `build_label_tracks.py` gains a cfDNA-endpoint track builder.

Receptive field must comfortably exceed the ~190 bp repeat and ideally span several
nucleosomes; 8 dilated layers at kernel 3 gives ~1 kb, covering ~5 nucleosomes. If NRL
recovery underperforms, add dilation depth before touching anything else.

## 6. Splits

Held-out **chromosomes**, not random windows — adjacent windows share nucleosome arrays,
so random splits leak.

- train: all autosomes except below
- val: chr8, chr10
- test: chr1, chr9 (untouched until the model is frozen)
- chrY excluded (BH01/IH01 sex unverified); chrM excluded

## 7. Evaluation battery

Ordered from "did it learn anything" to "is it a real nucleosome model":

1. **Per-base profile correlation** on held-out chromosomes (Pearson on log1p endpoint
   density), against a shuffled-sequence and a marginal-GC baseline.
2. **Count-head correlation** — log total endpoints per window.
3. **NRL recovery** — FFT of predicted WPS over held-out gene bodies must peak at
   ~190 bp. This is the sharpest single test: it cannot be passed by learning GC content.
4. **TSS array from sequence alone** — does the model reproduce the +1/+2/+3 phased array
   around held-out TSS without ever seeing an annotation? The canonical proof of
   sequence-encoded positioning.
5. **Agreement with Snyder's published dyads** — precision/recall of called dyads on the
   held-out test chromosomes.
6. **Bias ablation** — performance with the bias model frozen vs removed, confirming the
   positioning signal is not just end-motif preference.

Track 1 established that occupancy *level* and *phasing* are near-orthogonal (r=0.037), so
report profile and count metrics separately and never average them into one number.

## 8. Roadmap

- **Phase 1** — global sequence → endpoint density on pooled BH01+IH01. Cell-type-invariant
  by construction, so bulk pooling costs nothing. Target: the ~50% of positioning that is
  sequence-encoded.
- **Phase 2** — bias model / end-motif deconvolution (delivers block D).
- **Phase 3** — cell-type conditioning via the existing FiLM/per-cell-head machinery.

## 9. Principal risk — the Role-b failure mode returns

A sequence model preferentially learns the **invariant** component. That is exactly what
phase 1 wants, but it is also precisely how Role-b failed: predicted per-cell correlation
0.985 versus observed 0.549, collapsing to a single group.

Phase 3 must therefore be **evaluated on cell-type deviation, never on absolute prediction
quality** — a model that predicts the global backbone perfectly and the cell-type residual
not at all will score well on any absolute metric. Define that evaluation before building
phase 3, so we do not rediscover 0.985 the expensive way.

Weak supervision available for phase 3 when we get there: 14 Snyder per-*condition*
stable-nucleosome tracks already in hg38 (bladder, breast, CRC, Crohn's, liver, lung,
lupus, melanoma, pancreatic, prostate, testicular, UC, uterine), plus Teo aging
(25/70/100yo) and Gaffney bulk B-cell MNase.

## 10. Pipeline assets

- `/mnt/data/jganbat/nucprior/align_wps.sh` — fq2bam (GPU) → fragments → WPS
- `/mnt/data/jganbat/nucprior/wps_track.py` — Snyder K=120 WPS, difference-array + cumsum;
  validated to recover 190.0 bp on a synthetic nucleosome ladder
- `/mnt/data/jganbat/nucprior/refs/` — hg38 + classic BWA index
- `/mnt/data/jganbat/nucprior/snyder/geo/` — Snyder published dyad calls (validation)
