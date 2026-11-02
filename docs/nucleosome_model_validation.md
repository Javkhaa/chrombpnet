# External validation — sequence→nucleosome model (#3 + #1)

_2026-07-29. Independent-reference validation of the phase-1 model (`nuc_full.pt`,
held-out chr1+chr9). Two questions: (#3) does the model reproduce nucleosome positioning
against references it was never trained on — Snyder's own published dyad calls and the
canonical TSS array; and (#1) does it agree with a direct nucleosome assay (MNase). Phase-1
numbers reconfirmed unchanged: profile 0.679, count 0.624, NRL 200 bp PASS._

## #3a — TSS array from sequence (the canonical proof)

Aggregate predicted endpoint profile around **2,838 held-out protein-coding TSS**
(chr1+chr9, strand-oriented), vs the observed cfDNA track. Assets: `eval_nuc.py --tss`,
`models/tss_array.png`.

- **Predicted-vs-observed TSS profile r = 0.792.** The sequence model reproduces the
  observed cfDNA endpoint shape around TSS on chromosomes it never saw.
- **A phased +1/+2/+3 nucleosome array emerges downstream** of the TSS: a trough at the
  well-positioned +1 nucleosome (endpoints depleted / protected), a linker peak at ~+200,
  repeating. Downstream periodicity **182 bp** (zero-padded spectrum, 150–230 bp band).
- This is the textbook signature of sequence-encoded positioning — the +1 array is set by
  the promoter's sequence-directed nucleosome-depleted region, and the model recovers it
  from sequence alone.

## #3b — Agreement with Snyder's published dyad calls

Snyder 2016 (GSE71378) published per-sample nucleosome-center calls. We extracted the
chr1/chr9 centers, lifted hg19→hg38 (>99.97% mapped), and scored the model's predicted
**protection** (= −smoothed endpoint density) against them. Every metric is reported
against the **observed-data ceiling** — the same statistic computed on the observed pooled
endpoint track — because no sequence model can exceed what the observed cfDNA itself agrees
with the reference. Assets: `prep_snyder_dyads.py`, `eval_reference.py`,
`models/reference_agreement.png`. (n=6000 windows, dyad match tol ±50 bp.)

| Reference | model r | obs ceiling | frac of ceiling | precision | recall | dip ratio |
|---|---|---|---|---|---|---|
| **Snyder IH01** (in-training control) | 0.279 | 0.463 | **0.60** | 0.619 | 0.724 | 0.828 |
| **Snyder CH01** (cross-sample) | 0.320 | 0.545 | **0.59** | 0.692 | 0.751 | 0.814 |

- **The model recovers ~60% of the observed cfDNA's own agreement** with Snyder's
  independent dyad calls — from sequence alone, on held-out chromosomes.
- **Precision 0.62–0.69, recall 0.72–0.75** for predicted dyads within ±50 bp of a Snyder
  center.
- **Not circular:** CH01 is a *different* Snyder library (the model trained on BH01+IH01),
  and it agrees **as well or better** than the in-training IH01 — the agreement is with
  nucleosome positioning, not with one library's quirks.
- **Dip ratio 0.81–0.83:** predicted endpoint density falls to ~82% of the linker level at
  Snyder's called centers (see `reference_agreement.png` — clean dip at 0, symmetric linker
  bumps at ±~90 bp).

## #1 — Cross-assay MNase: an honest negative (reference/domain gap)

Goal: validate against a *direct* nucleosome assay to rule out a cfDNA-specific artifact.
The on-disk maps — Gaffney 2012 B-cell MNase (`GSE36979`, hg38) and Song
(`GSE81314`, hg38) — **do not provide usable cross-assay validation**, and the reason is
diagnostic, not a model failure:

- **Gaffney: model r −0.103, but the observed-data ceiling is also −0.079** — i.e. the
  observed pooled cfDNA endpoints themselves barely relate to the Gaffney occupancy track
  (r ≈ +0.05 with either value column, the *wrong sign* for nucleosome protection). There
  is essentially no co-registration to recover.
- **Song: obs ceiling ≈ 0.006, underpowered** (17k stable-nucleosome calls genome-wide;
  only 140 fell in the aggregate). Not aligned either.

Interpretation: **plasma cfDNA fragment endpoints and bulk B-cell MNase occupancy do not
co-register at nucleosome scale** in this data — an assay + cell-type domain gap (apoptotic
nuclease cutting in mixed-lineage plasma vs micrococcal digestion in a single cultured cell
type). This is a property of the *data*, confirmed on observed endpoints before the model
is even involved. It reinforces why the cfDNA-native Snyder reference (#3b) was the right
instrument, and leaves true cross-assay validation (a matched, independent nucleosome map)
as the one open item — the Gaffney/Song files are not it.

## Bottom line

- **#3 validates strongly.** The model reproduces cfDNA nucleosome positioning against
  references it never trained on: ~60% of the observed ceiling on Snyder's published dyads
  (precision ~0.65 / recall ~0.74), confirmed cross-sample (CH01), and it recovers the
  canonical TSS +1/+2/+3 array from sequence (r=0.79 vs observed).
- **#1 is a clean negative with a clear cause** — B-cell MNase doesn't co-register with
  plasma cfDNA even at the observed-data level. Cross-assay validation against a matched
  nucleosome map remains the open next step.

## Assets

- `eval_nuc.py --tss` · `models/tss_array.png`
- `prep_snyder_dyads.py` · `snyder/dyads_{IH01,CH01}_chr1_chr9_hg38.bed`
- `eval_reference.py` · `models/reference_agreement.png`
