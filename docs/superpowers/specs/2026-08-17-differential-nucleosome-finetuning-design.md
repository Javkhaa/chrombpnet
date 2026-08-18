# Differential Nucleosome Fine-Tuning Design

## Goal

Fine-tune the existing 346-cell hybrid ChromBPNet checkpoint to preserve adult
cell-type-specific nucleosome positioning at differential loci without leaking the
held-out test chromosomes or degrading ordinary nucleosome-profile performance.

The fine-tuning target is adult tissue-context deviation: for the same genomic
sequence, the model should reproduce how each adult cell type's normalized dyad
profile differs from the leave-one-out consensus of related adult cell types.

## Non-Goals

- Do not train on the existing 553 loci from `chr1`, `chr3`, and `chr6`.
- Do not train directly on robust z-scores, shift calls, or Boolean differential
  labels.
- Do not change the accessibility head, count heads, shared sequence trunk, fetal
  heads, or cell-line heads in the first fine-tuning stage.
- Do not claim that performance on the 553-locus challenge set estimates unbiased
  genome-wide performance; those loci were selected for observed differential
  behavior.

## Data Splits And Leakage Boundary

The existing fold remains authoritative:

- training: `chr2`, `chr4`, `chr5`, `chr7`, `chr9`-`chr19`, `chr21`, `chr22`,
  `chrX`, and `chrY`, excluding `chr8`;
- validation: `chr8` and `chr20`;
- final test: `chr1`, `chr3`, and `chr6`.

Adult differential discovery is run independently within each genomic split. The
training call set is eligible for gradient updates. The validation call set is used
for checkpoint and hyperparameter selection. The existing 553-locus adult re-score
on the test chromosomes is opened only once after the final checkpoint is selected.

Every generated artifact records its chromosome list and source manifest. The
fine-tuning entry point must reject overlapping train, validation, and test
coordinates and must reject any training row from `chr1`, `chr3`, or `chr6`.

## Cohort And Comparator Meaning

The adult cohort contains 244 pseudobulk tracks mapped back to their original
indices in the 346-cell checkpoint. The current `lineage` field is a tissue-context
stratum, not a molecular lineage claim. A locus-context unit is eligible when at
least four depth-qualified members are present, giving each target at least three
leave-one-out comparators.

Contexts with fewer than four total adult tracks can contribute global evaluation
metrics but cannot contribute the primary tissue-context fine-tuning loss.

## Differential Discovery

Training and validation candidates are built from adult peak summits restricted to
their own chromosomes. The first experiment uses the existing bounded peak-union
caller with:

- up to 5,000 training candidates;
- up to 2,000 validation candidates;
- 501 bp scoring windows;
- 20 bp Gaussian smoothing;
- minimum depth 20;
- minimum total-variation effect 0.05;
- minimum robust z-score 3;
- at least three leave-one-out comparators.

The resulting Parquet tables identify interesting locus-context units and retain
all target rows. Differential calls select fine-tuning locations; raw BigWig dyad
profiles at those locations remain the supervision.

Sampling first chooses uniformly among eligible tissue contexts and then uniformly
among eligible loci in that context, so cardiac, PBMC, and bone-marrow call counts
do not dominate the objective. Within a selected locus-context unit, all
depth-qualified members are evaluated together. Variable-sized units are handled
with gradient accumulation rather than dropping members from large contexts.

## Model Parameters

Fine-tuning starts from the selected `HybridMultiCellModel` checkpoint and preserves
the checkpoint's complete 346-cell ordering.

Stage 1 trains only adult rows of `nuc_profile_w`. The scalar
`nuc_profile_b` adds the same offset at every profile position and is therefore
softmax-invariant; every bias row remains frozen and is checked bitwise.

The shared convolutional trunk, cell embeddings, FiLM layers, accessibility head,
and nucleosome count parameters remain frozen. Non-adult `nuc_profile_w` rows are
protected by an explicit gradient mask and a zero-weight-decay optimizer group. A
post-step invariant verifies that non-adult weight rows and all bias rows are
bitwise unchanged.

Stage 1 is the only implementation in this scope. Unfreezing FiLM or late trunk
layers requires a separate experiment and approval after the heads-only result is
evaluated.

## Paired Fine-Tuning Objective

For one locus-context unit, the same one-hot DNA sequence is repeated across its
depth-qualified adult cell-type indices. The model returns one nucleosome profile
logit vector per cell. Softmax converts logits into profile probabilities.

Observed dyad profiles are Gaussian-smoothed and L1-normalized. For target cell
`c`, observed and predicted leave-one-out deviations are:

```text
D_obs[c]  = P_obs[c]  - mean(P_obs[j]  for j != c)
D_pred[c] = P_pred[c] - mean(P_pred[j] for j != c)
```

The primary contrast loss is:

```text
L_dev = 1 - mean_c Pearson(D_pred[c], D_obs[c])
```

The fine-tuning objective also retains absolute profile supervision:

```text
L_total = L_replay + alpha * L_diff_profile + lambda_dev * L_dev
```

- `L_diff_profile` is the existing multinomial nucleosome-profile loss at selected
  differential positions, using the training target smoothing of 6 bp.
- `L_replay` is the same nucleosome-profile loss on ordinary adult peak/nonpeak
  examples from training chromosomes.
- The initial differential-to-replay unit ratio is 1:1.
- The initial `alpha` is 1.0.
- `lambda_dev` is selected from `0.25`, `1.0`, and `4.0` using validation data.

Count losses and accessibility losses are not included because their parameters are
frozen. They remain evaluation gates.

## Optimization And Checkpoint Selection

Fine-tuning uses a fresh AdamW optimizer with learning rate `1e-5`, zero weight
decay, gradient clipping at `1.0`, 100 linear warmup steps, and cosine decay to
`1e-6`. Each `lambda_dev` candidate receives a 2,000-step seed-0 pilot with
validation every 250 steps. The best candidate that passes every preservation gate
is restarted from the base checkpoint for a 10,000-step run. W&B logs the component
losses, differential metrics, ordinary profile metrics, learning rate, gradient
norm, and frozen-parameter invariants.

The selected checkpoint maximizes mean validation deviation correlation subject to
all of these gates:

- ordinary adult validation nucleosome-profile Pearson drops by no more than 0.01
  absolute from the base checkpoint;
- ordinary validation nucleosome-profile NLL increases by no more than 2%;
- accessibility profile and count outputs are numerically unchanged within floating
  point tolerance;
- non-adult nucleosome-head rows are bitwise unchanged;
- validation shift-direction accuracy does not decline.

If no checkpoint passes every gate, fine-tuning is reported as unsuccessful and
the base checkpoint remains the production model.

## Metrics

Validation and final test reports compare the base and fine-tuned checkpoints on:

- mean Pearson correlation between predicted and observed tissue-context
  deviations;
- Spearman correlation between predicted and observed total-variation effects;
- shift-direction accuracy for observed absolute shifts of at least 10 bp;
- shift mean absolute error in bp;
- AUPRC for observed differential target-locus labels, using predicted
  tissue-context robust z-score as the ranking value;
- ordinary nucleosome profile Pearson, multinomial NLL, and count MSE;
- ordinary accessibility profile and count metrics.

Predicted differential scores use model profiles but observed depth eligibility.
They do not invent split-half sampling noise from deterministic model predictions.

The final test report also states the challenge-set ascertainment: the 553 loci were
selected from an earlier mixed-cell differential screen, and 387 currently have at
least one adult differential call.

## Components And Interfaces

The implementation will add isolated components for:

1. split-specific differential-locus preparation and leakage validation;
2. paired locus-context loading from sequence and adult dyad BigWigs;
3. differentiable leave-one-out deviation loss;
4. adult-row gradient masking and frozen-parameter invariants;
5. fine-tuning orchestration, W&B logging, checkpointing, and early stopping;
6. base-versus-fine-tuned validation and test reporting.

The existing ordinary dataset and `multinomial_nll` remain the source of replay
examples and profile supervision. The existing `DeviationProbe` math may be reused
after it is generalized to tissue-context leave-one-out units; the notebook and
differential caller remain analysis readers and do not own training behavior.

## Failure Handling

The fine-tuning command fails before optimization when:

- checkpoint cell-type names do not exactly cover the adult metadata rows;
- a training or validation artifact contains a forbidden chromosome;
- a locus-context unit has fewer than four depth-qualified members;
- required BigWigs, reference sequence, or differential artifacts are missing;
- stored output length or smoothing assumptions disagree with the checkpoint;
- a frozen parameter changes during a verification step.

Runs write to a new output prefix and never overwrite the base checkpoint or the
adult differential-calling artifacts.

## Testing And Verification

Unit tests use synthetic profiles with known cell-specific shifts and verify:

- leave-one-out deviation construction;
- lower deviation loss for correctly shifted predictions;
- correct adult-to-checkpoint index mapping;
- zero gradients and zero parameter change for frozen and non-adult rows;
- split-overlap and forbidden-test-chromosome rejection;
- context-balanced sampling and minimum-comparator enforcement;
- stable metric behavior for empty, shallow, and zero-variance profiles.

An integration test fine-tunes a tiny hybrid model on synthetic shifted loci and
must improve held-out deviation correlation without changing accessibility outputs.
The first real run is a short bounded pilot. The full fine-tuning run proceeds only
if the pilot passes the checkpoint gates and produces a complete base-versus-tuned
validation report.

## Delivery Sequence

1. Generate and validate adult differential artifacts on train and validation
   chromosomes.
2. Implement paired data loading, loss, and parameter masks with synthetic tests.
3. Add the fine-tuning CLI, checkpoint selection, and W&B logging.
4. Run a bounded pilot and inspect all preservation gates.
5. Run the selected full fine-tuning configuration.
6. Evaluate once on the untouched 553-locus test challenge set and update the
   project report.
