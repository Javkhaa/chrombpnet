# Differential nucleosome fine-tuning results

## Scope

This experiment fine-tuned only the 244 adult rows of the hybrid model's
nucleosome profile weight and bias. Training used differential loci on the fold's
training chromosomes, checkpoint selection used `chr8` and `chr20`, and the final
test used `chr1`, `chr3`, and `chr6` exactly once after validation selection.

The full run used lambda 4.0, 10,000 steps, learning rate `1e-5`, 100 warmup
steps, gradient clipping at 1.0, validation every 250 steps, seed 0, and CUDA AMP.
[W&B run `qp7v1uj2`](https://wandb.ai/prima-mente/jg_experiments/runs/qp7v1uj2)
uses project `jg_experiments` and run name `hybrid346_diffnuc_final`.

## Pilot correction

The original sweep in `/mnt/data/jganbat/scatac_corpus/run/diffnuc_finetune/pilot`
is archived but invalid. CUDA AMP left profile forwards in reduced precision and
the Pearson calculation was not stable enough for the selection metric. Those
reports are compatibility diagnostics, not evidence that the biological gates
failed.

After fixing profile-forward precision and stable Pearson without changing gates
or hyperparameters, the exact three-lambda protocol was rerun from the original
base checkpoint into `pilot_ampfix`. All 24 candidate-step gate checks passed.
The corrected selector chose lambda 4.0 at step 2,000 because it had the highest
passing validation deviation Pearson.

| Pilot artifact | Value |
|---|---|
| Corrected selector | `/mnt/data/jganbat/scatac_corpus/run/diffnuc_finetune/pilot_ampfix/selected_pilot.json` |
| Selector SHA-256 | `8c5724a1da31c3aa625b03e5f3626b389659a5aadbacb7a4ef18601a43cf4814` |
| Selected pilot checkpoint SHA-256 | `a44a0e90b4b74a9921f3753fd179b957aae4af7a1df296011959bbc104bb9af6` |
| Selected lambda / pilot step | 4.0 / 2,000 |

The full run consumed only lambda and provenance from this selector. It restarted
weights from the original base checkpoint and did not resume the pilot checkpoint.

## Artifacts and provenance

| Artifact | SHA-256 |
|---|---|
| Base checkpoint `/mnt/data/jganbat/scatac_corpus/run/model_346_hybrid_long.pt` | `c1e75296de4d75bc65c92d6b2394ebc5d3d2d1660d1c591868f1c1dda5913b07` |
| Training calls `/mnt/data/jganbat/scatac_corpus/run/adult_diffnuc_train/differential_nucleosomes.parquet` | `b5a04d06ae0dd475e7d7bf00c823617befcc010f26dbb8b5a9b63b1674dfcd00` |
| Validation calls `/mnt/data/jganbat/scatac_corpus/run/adult_diffnuc_valid/differential_nucleosomes.parquet` | `b62f6fc4900b70f5492dd39f3fefb4d169349c77bcd58ebe7bd2622a4aec26b4` |
| Final checkpoint `/mnt/data/jganbat/scatac_corpus/run/diffnuc_finetune/final/final.pt` | `a47581d1cdba3a0d7842b0bc1f8e6b77654201d40ade31e4a9b627e8b8b4048f` |
| Validation report `/mnt/data/jganbat/scatac_corpus/run/diffnuc_finetune/final/final_validation.json` | `35cfdcb04f4ae9681b48930396c0769d23cd170a35edb08ead764c933d17e46b` |
| Test calls `/mnt/data/jganbat/scatac_corpus/run/adult_diffnuc_pilot_rescore/differential_nucleosomes.parquet` | `ad4223f76995edcb31b3be258fec8b69b77add7ea1b4658d1571c898ee5a91e5` |
| Test report `/mnt/data/jganbat/scatac_corpus/run/diffnuc_finetune/final/test_comparison.json` | `79e9036a860ff8b9a7253cf00c7a6eb653614011a6ad84988121db2ebb448813` |

The final checkpoint records the base path and SHA-256 above, lambda 4.0, and
selected step 9,500. Its parent is the original base checkpoint, not the selected
pilot checkpoint. All report metrics are finite.

## Validation

All 40 validation snapshots at 250-step intervals passed. An independent audit
recomputed every gate and selected step 9,500 as the highest passing deviation
Pearson. Validation contains 2,050 context units, 29,468 target-locus rows, and
2,000 ordinary examples.

| Differential metric | Base | Tuned | Delta |
|---|---:|---:|---:|
| Deviation Pearson | 0.246052538 | 0.248848759 | +0.002796221 |
| Effect Spearman | 0.091179285 | 0.093419485 | +0.002240201 |
| Shift-direction accuracy | 0.508776084 | 0.510972428 | +0.002196345 |
| Shift MAE (bp) | 25.184626965 | 25.161224310 | -0.023402655 |
| Differential AUPRC | 0.039483704 | 0.040288907 | +0.000805203 |

| Ordinary metric | Base | Tuned |
|---|---:|---:|
| Nucleosome profile Pearson | 0.061394005 | 0.061387822 |
| Nucleosome profile NLL | 100.078401489 | 100.087699585 |
| Nucleosome count MSE | 1.412060146 | 1.412060146 |
| Accessibility profile Pearson | 0.179430316 | 0.179430316 |
| Accessibility profile NLL | 374.091688721 | 374.091688721 |
| Accessibility count MSE | 2.860693146 | 2.860693146 |

Ordinary nucleosome Pearson dropped by 0.00000618, below the allowed 0.01,
and NLL increased by 0.00930, or 0.00929%, below the allowed 2%. Shift-direction
accuracy improved. Accessibility outputs were unchanged.

## Preservation audit

The base and final state dictionaries have the same keys, checkpoint arguments,
and 346-cell ordering. The only changed tensors are `nuc_profile_w` and
`nuc_profile_b`. All 244 adult rows changed in both tensors; all 102 non-adult rows
are bitwise identical. Every other tensor is bitwise identical, including the
accessibility head, count heads, shared trunk, cell embeddings, FiLM layers, and
other frozen parameters.

## One-time test result

The test evaluate command ran once, exited successfully, and was not rerun for
scientific selection. The report covers 553 unique loci, 134,932 input rows, 244
adult cell types, 16 tissue contexts, 4,993 evaluable context units, 44,082 scored
target-locus rows, and 2,000 ordinary examples.

| Differential metric | Base | Tuned | Delta |
|---|---:|---:|---:|
| Deviation Pearson | 0.204142844 | 0.205641934 | +0.001499090 |
| Effect Spearman | 0.075204163 | 0.076543357 | +0.001339194 |
| Shift-direction accuracy | 0.515222482 | 0.514864725 | -0.000357756 |
| Shift MAE (bp) | 28.862620353 | 28.848653026 | -0.013967327 |
| Differential AUPRC | 0.033734887 | 0.033697456 | -0.000037431 |

| Ordinary metric | Base | Tuned | Delta |
|---|---:|---:|---:|
| Nucleosome profile Pearson | 0.080069422 | 0.080086911 | +0.000017489 |
| Nucleosome profile NLL | 150.879351807 | 150.866054932 | -0.013296875 |
| Nucleosome count MSE | 1.320733236 | 1.320733236 | 0 |
| Accessibility profile Pearson | 0.221462815 | 0.221462815 | 0 |
| Accessibility profile NLL | 568.993482910 | 568.993482910 | 0 |
| Accessibility count MSE | 2.247407997 | 2.247407997 | 0 |

Test deviation Pearson, effect Spearman, shift MAE, and ordinary nucleosome profile
metrics improved slightly. Shift-direction accuracy and differential AUPRC declined
slightly. The effects are mixed and very small, while accessibility and frozen-row
invariants remain preserved.

## Interpretation and limitations

The result supports a narrow conclusion: adult-row profile-head fine-tuning can
produce a small improvement in tissue-context deviation agreement without changing
the accessibility model or non-adult behavior. It does not establish a material
improvement in every differential endpoint, and it does not justify unfreezing the
shared model.

The 553 test loci are an ascertained differential challenge set selected by an
earlier mixed-cell differential screen. Of these, 387 currently have at least one
adult differential call. They are not an unbiased genome-wide sample, so this test
does not estimate genome-wide performance, prevalence, calibration, or expected
behavior at ordinary loci. The adult tracks are pseudobulks, tissue context is not
a molecular lineage claim, and the analysis does not estimate donor-level variance.
