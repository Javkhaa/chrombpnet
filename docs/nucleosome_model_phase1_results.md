# Phase 1 — sequence → nucleosome model: results

_2026-07-26. Outcome of the phase-1 build specified in `DESIGN_nucleosome_model.md`:
does a ChromBPNet-style model predict cfDNA nucleosome positioning from sequence alone,
and does it recover the nucleosome repeat on held-out chromosomes? **Yes to both.**_

> **Status (2026-11-02).** The "Next" items at the end of this file are done. See
> `nucleosome_model_phase2_3_results.md`, `nucleosome_model_validation.md` and
> `cfdna_nucleosome_report.html`.

## Result in one line

A dilated-conv sequence model predicts per-base cfDNA fragment-endpoint density on
**held-out chromosomes** at profile-correlation **0.68**, and its predictions reproduce
the **~190 bp nucleosome repeat from sequence alone** — confirming nucleosome
positioning is substantially sequence-encoded and that this architecture captures it.

## Data

| Sample | Coverage | Source |
|---|---|---|
| BH01 | 96× | Snyder 2016, GSE71378 / SRR2129993 |
| IH01 | 101× | SRR2130050 |

~200× pooled healthy plasma cfDNA, **realigned natively to hg38** (Parabricks fq2bam on
H100) against the same reference the scATAC corpus uses. Paired-end throughout; fragments
= proper pairs, MAPQ≥30, 50–250 bp, duplicates removed. 1.9B fragments total.

**Label track** (`pooled_endpoints_hg38.bw`): per-base fragment-endpoint density — each
fragment's 5′ (`start`) and 3′ (`end−1`) nuclease cut site, no Tn5 shift. 3.8B endpoints,
all 24 chromosomes. The endpoint density itself carries a **195.6 bp** periodicity
(whole-chromosome FFT) — the target is biologically meaningful, not noise.

**Pipeline validated end-to-end** (`/mnt/data/jganbat/nucprior/e2e/`): raw FASTQ → align →
fragments → WPS, checking all 24 chromosomes carry signal + periodicity. This test caught a
silent chromosome-loss bug (lexical vs karyotypic sort dropping 13/24 chromosomes from the
bigWig) before it reached training data.

## Model

`SeqNucleosomeModel` (single head) — the repo's exact dilated-conv trunk
(`multitask_nucleosome_torch.py`: first conv k=21, 8 residual dilated layers, 512 filters)
+ one `ProfileCountHead`. Single head because bulk cfDNA has no accessibility channel; the
head reuses the ChromBPNet loss unchanged — profile **multinomial NLL** + count **MSE** on
log total endpoints. bf16 AMP, Adam 1e-3, batch 256, inputlen 2114 → outputlen 1000.

**Splits are by chromosome** (adjacent windows share nucleosome arrays, so random splits
leak): test **chr1, chr9** (untouched during training), valid chr8/chr10, train the rest
(chrY excluded). 1.95M train / 260k valid / 308k test windows, tiled every 1 kb and kept
only where the endpoint track has real coverage (drops assembly gaps / unmappable blocks).

## Held-out results (test chr1 + chr9, never trained on)

| Metric | Value |
|---|---|
| Profile correlation (per-window, predicted prob vs observed density) | **0.679** |
| Count correlation (predicted vs observed log window-total) | **0.624** |
| **NRL recovery** (predicted power-spectrum peak) | **200 bp = observed 200 bp — PASS** |

Consistent across both test chromosomes (chr1 0.679, chr9 0.680). Trained to a plateau
(prof_corr 0.675→0.683 over 11 epochs, val_loss monotonic; stopped at plateau).

**On the NRL number:** 200 bp is the discrete FFT bin nearest ~190 at 1 kb window
resolution (achievable periods in-band: 250/200/167/143 bp). The whole-chromosome FFT of
the observed track gives the sharper 195.6 bp. The point is that the *predicted* profile —
built from sequence with no access to the observed track — reproduces the same periodicity
as the observed data. This is the sharp test: it cannot be passed by learning GC content.

## Interpretation

- **Sequence encodes nucleosome positioning.** 0.68 held-out profile correlation from
  sequence alone is a strong signal for a genome-wide (not peak-restricted) target.
- **NRL recovery is the load-bearing result.** A model predicting only local base
  composition would not reproduce the 190 bp periodicity; this one does, on chromosomes it
  never saw.
- This closes Track 1's dead end. Track 1 could not test periodicity residualization
  because *no per-base positioning prior existed*. This model **is** that prior, learned
  from sequence.

## What this does / doesn't establish

- **Establishes:** the global, cell-type-invariant sequence→positioning signal (phase 1's
  scope). ~50% of nucleosome positioning is sequence-encoded; this captures the sequence
  part.
- **Does not yet address:** cell-type-specific deviation (phase 3), or whether the model
  has merely learned the invariant backbone (the Role-b failure mode — must be evaluated on
  *deviation*, not absolute prediction).

## Assets

- Model: `/mnt/data/jganbat/nucprior/models/nuc_full.pt`
- Code: `train_nuc.py`, `eval_nuc.py`, `endpoint_track.py`, `make_regions.py`
- Label/data: `aligned/pooled_endpoints_hg38.bw`, `aligned/{IH01,BH01}_wps_hg38.bw`

## Next

1. **Phase 2 — bias / end-motif.** Fit the cfDNA nuclease sequence-preference bias model
   (the ChromBPNet bias slot). This is the published **end-motif** signal — the old
   fragmentomics *block D* — obtained for free and properly deconvolved from positioning.
   Then ablate: confirm positioning performance isn't just end-motif preference.
2. **Phase 3 — cell-type conditioning.** FiLM the trunk on a cell-type embedding
   (`ConditionedMultiCellModel` already exists). **Evaluate on cell-type deviation**, not
   absolute prediction — define that eval before building, to avoid rediscovering the
   0.985-cross-cell-correlation collapse the expensive way. Weak supervision available: 14
   Snyder per-condition stable-nucleosome tracks (hg38) + Teo aging.
3. **Sharper eval** (optional): TSS +1/+2/+3 array from sequence; agreement with Snyder's
   published dyad calls on the test chromosomes.
