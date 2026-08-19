# Differential nucleosome fine-tuning results

## Status

The authoritative corrected evidence is under
`/mnt/data/jganbat/scatac_corpus/run/diffnuc_finetune/pilot_metricfix` and
`/mnt/data/jganbat/scatac_corpus/run/diffnuc_finetune/final_metricfix`. It uses
pooled eligible-row differential metrics. The earlier `pilot`, `pilot_ampfix`,
and `final` outputs are immutable audit artifacts, but their context-weighted
effect and shift summaries are invalid scientific aggregation evidence.

The challenge result below is a single corrective re-evaluation after that
aggregation defect, not untouched test use and not a new selection pass. The
checkpoint remains experimental.

## Scope and protocol

Fine-tuning changed only the 244 adult rows of `nuc_profile_w`. All
`nuc_profile_b` rows, all 102 non-adult `nuc_profile_w` rows, and every other
model tensor were frozen. Training used the fold's training chromosomes,
selection used `chr8` and `chr20`, and the corrective challenge used `chr1`,
`chr3`, and `chr6` once after validation approval.

The corrected three-lambda pilot ran for 2,000 steps per candidate from the
original base checkpoint with lambda `0.25`, `1.0`, and `4.0`. It retained the
approved learning rate `1e-5`, 100 warmup steps, gradient clipping at 1.0,
validation every 250 steps, seed 0, and the `--amp` compatibility flag. All
24/24 snapshots passed corrected gates. Production selection and an independent
row-level audit selected lambda `4.0`, step `2,000`.

The 10,000-step final then restarted from the original base weights, consumed
only the selected lambda and bound provenance, and selected step `9,750` among
40/40 passing validation snapshots. W&B records exactly 10,000 finite sequential
training records and 40 validation records.

| Run | W&B |
|---|---|
| Pilot lambda 0.25 | [`ekuf78h4`](https://wandb.ai/prima-mente/jg_experiments/runs/ekuf78h4) |
| Pilot lambda 1.0 | [`om1s37je`](https://wandb.ai/prima-mente/jg_experiments/runs/om1s37je) |
| Pilot lambda 4.0 | [`gmt6kl1z`](https://wandb.ai/prima-mente/jg_experiments/runs/gmt6kl1z) |
| Final lambda 4.0 | [`2zwf8wzw`](https://wandb.ai/prima-mente/jg_experiments/runs/2zwf8wzw) |

## Artifacts and provenance

| Artifact | SHA-256 |
|---|---|
| Base checkpoint `/mnt/data/jganbat/scatac_corpus/run/model_346_hybrid_long.pt` | `c1e75296de4d75bc65c92d6b2394ebc5d3d2d1660d1c591868f1c1dda5913b07` |
| Training calls `/mnt/data/jganbat/scatac_corpus/run/adult_diffnuc_train/differential_nucleosomes.parquet` | `b5a04d06ae0dd475e7d7bf00c823617befcc010f26dbb8b5a9b63b1674dfcd00` |
| Validation calls `/mnt/data/jganbat/scatac_corpus/run/adult_diffnuc_valid/differential_nucleosomes.parquet` | `b62f6fc4900b70f5492dd39f3fefb4d169349c77bcd58ebe7bd2622a4aec26b4` |
| Corrected pilot selector `/mnt/data/jganbat/scatac_corpus/run/diffnuc_finetune/pilot_metricfix/selected_pilot.json` | `8a21999d9c6f336c15ad6c44060fafaae3bd1b9329a0ff493f0b6c063d6fe9ae` |
| Final checkpoint `/mnt/data/jganbat/scatac_corpus/run/diffnuc_finetune/final_metricfix/final.pt` | `9fd997d1a36ea43cafa99bfdea116a67a69c82decc5b364e1593c75d3ae698e9` |
| Validation report `/mnt/data/jganbat/scatac_corpus/run/diffnuc_finetune/final_metricfix/final_validation.json` | `46472f3db9ca2e58ebda0fa9c04bb424d230dffccb10d625330471268a36e194` |
| Challenge calls `/mnt/data/jganbat/scatac_corpus/run/adult_diffnuc_pilot_rescore/differential_nucleosomes.parquet` | `ad4223f76995edcb31b3be258fec8b69b77add7ea1b4658d1571c898ee5a91e5` |
| Corrective test report `/mnt/data/jganbat/scatac_corpus/run/diffnuc_finetune/final_metricfix/test_comparison.json` | `649a35065680d724d542d6a4cc9df6d9d73f796fae540de65725a9b1608e3b84` |

The final checkpoint and validation report bind the original base hash, lambda
`4.0`, selected step `9,750`, and corrected selector hash. The production
evaluation binder accepted those artifacts before challenge access.

## Corrected validation

Validation contains 2,050 context units, 29,468 target-locus rows, and 2,000
ordinary examples. For both base and tuned evidence, deviation Pearson, effect
Spearman, and AP score all 29,468 rows. AP has 662 positives and
expected=eligible=scored=`29,468`. Shift metrics use the complete approved
observed-shift subset: expected `29,468`, eligible=scored=`9,722`.

| Differential metric | Base | Tuned | Delta |
|---|---:|---:|---:|
| Deviation Pearson | 0.246052538 | 0.248868813 | +0.002816275 |
| Effect Spearman | 0.148749280 | 0.153021952 | +0.004272672 |
| Shift-direction accuracy | 0.507303024 | 0.510080230 | +0.002777206 |
| Shift MAE (bp) | 27.579716108 | 27.552972639 | -0.026743468 |
| Differential AUPRC | 0.039483704 | 0.040412406 | +0.000928702 |

| Ordinary metric | Base | Tuned | Delta |
|---|---:|---:|---:|
| Nucleosome profile Pearson | 0.061394005 | 0.061380984 | -0.000013021 |
| Nucleosome profile NLL | 100.078401489 | 100.086640564 | +0.008239075 |
| Nucleosome count MSE | 1.412060146 | 1.412060146 | 0 |
| Accessibility profile Pearson | 0.179430316 | 0.179430316 | 0 |
| Accessibility profile NLL | 374.091688721 | 374.091688721 | 0 |
| Accessibility count MSE | 2.860693146 | 2.860693146 | 0 |

Independent recomputation matched all eight stored gates at every snapshot:
differential values were finite, pooled coverage matched, nucleosome Pearson and
NLL stayed within their preservation thresholds, shift accuracy did not regress,
and accessibility, non-adult rows, and profile bias remained unchanged.

## Preservation audit

The base and corrected final checkpoints have identical state-dict keys,
checkpoint arguments, and 346-cell ordering. All 244 adult `nuc_profile_w` rows
changed. All 102 non-adult rows are bitwise identical. `nuc_profile_b` is
bitwise identical for all 346 rows, and every accessibility, count-head,
shared-trunk, embedding, FiLM, and other tensor is bitwise identical.

## Corrective challenge result

Validation approval was recorded at `2026-08-19T04:09:45Z`; challenge access
began afterward at `2026-08-19T04:12:06Z`. The production evaluate command ran
exactly once and exited successfully. Its report explicitly records
`evaluation_kind=corrective_re_evaluation`.

The ascertained 553-locus challenge contributes 4,993 context units, 44,082
target-locus rows, and 2,000 ordinary examples. AP expected=eligible=scored is
`44,082` with 983 positives. Shift expected is `44,082`, with
eligible=scored=`18,781`. Base and tuned counts match and the fail-closed
evaluation gate passes.

| Differential metric | Base | Tuned | Delta |
|---|---:|---:|---:|
| Deviation Pearson | 0.204142844 | 0.205677818 | +0.001534974 |
| Effect Spearman | 0.084267499 | 0.088297475 | +0.004029976 |
| Shift-direction accuracy | 0.511687344 | 0.511900325 | +0.000212981 |
| Shift MAE (bp) | 32.486768543 | 32.480166125 | -0.006602417 |
| Differential AUPRC | 0.033734887 | 0.033704408 | -0.000030479 |

| Ordinary metric | Base | Tuned | Delta |
|---|---:|---:|---:|
| Nucleosome profile Pearson | 0.080069422 | 0.080089416 | +0.000019994 |
| Nucleosome profile NLL | 150.879351807 | 150.863146362 | -0.016205444 |
| Nucleosome count MSE | 1.320733236 | 1.320733236 | 0 |
| Accessibility profile Pearson | 0.221462815 | 0.221462815 | 0 |
| Accessibility profile NLL | 568.993482910 | 568.993482910 | 0 |
| Accessibility count MSE | 2.247407997 | 2.247407997 | 0 |

## Interpretation and limitations

Corrected pooled evidence shows small validation gains in every differential
endpoint while preserving the ordinary and frozen-model gates. On the
corrective challenge, deviation correlation, effect rank, shift accuracy, shift
MAE, and ordinary nucleosome metrics improve slightly, while AUPRC is essentially
flat with a small negative absolute change. These effect sizes are modest and do
not establish broad or material improvement.

The 553 challenge loci were ascertained by an earlier mixed-cell differential
screen; 387 currently have at least one adult differential call. They are not an
unbiased genome-wide sample, so this evaluation does not estimate genome-wide
performance, prevalence, or calibration. Adult tracks are pseudobulks, tissue
context is not a molecular-lineage claim, and donor-level variance is not
estimated. The corrected checkpoint is experimental and should not replace the
base model without additional independent validation.
