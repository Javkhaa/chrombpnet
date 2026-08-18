# Differential Nucleosome Fine-Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fine-tune only the adult rows of the hybrid model's per-cell nucleosome profile head on training-chromosome differential loci, select on held-out validation chromosomes, and evaluate once on the untouched 553-locus test challenge set.

**Architecture:** Split-specific adult differential Parquets select locus-context units, while raw dyad BigWigs remain the supervision. A paired loader repeats the same sequence across all depth-qualified adult members of one tissue context; a differentiable leave-one-out deviation loss trains adult nucleosome profile-head rows alongside ordinary adult replay examples. The shared trunk, conditioning, accessibility, count parameters, and non-adult head rows remain frozen and are guarded by invariants.

**Tech Stack:** Python 3.11, PyTorch, Polars, NumPy, SciPy, pyBigWig, pyfaidx, unittest, W&B, uv.

**Spec:** `docs/superpowers/specs/2026-08-17-differential-nucleosome-finetuning-design.md`

## Global Constraints

- Preserve `chr1`, `chr3`, and `chr6` as final-test-only chromosomes; reject them from gradient-bearing artifacts.
- Use `chr8` and `chr20` only for validation and checkpoint selection.
- Use the remaining chromosomes listed in `refs/folds/fold_0.json` for fine-tuning.
- Preserve the checkpoint's complete 346-cell ordering; map the 244 adult rows by exact `cell_type` string.
- Train only adult rows of `nuc_profile_w`; preserve every `nuc_profile_b` row
  bitwise because the scalar profile bias is softmax-invariant.
- Keep the caller's 501 bp differential window; fetch the checkpoint's full 1,000 bp target for profile NLL and use a renormalized central 501 bp crop for deviation loss and differential metrics.
- Use observed BigWig profiles as supervision; z-scores, shifts, and call flags select or evaluate examples but are not direct regression targets.
- Do not overwrite the base checkpoint, adult call artifacts, or existing run outputs.
- Preserve all unrelated dirty-worktree changes. Stage only files named by each task.
- Use unittest, not pytest, for the repository test commands.

---

### Task 0: Consolidate Approved Prerequisites And Create The Execution Worktree

**Files:**
- Modify: `.gitignore`
- Commit this implementation plan: `docs/superpowers/plans/2026-08-18-differential-nucleosome-finetuning.md`
- Commit existing approved PyTorch files: `chrombpnet/multitask/evaluate_multicell.py`, `chrombpnet/multitask/train_loop.py`, `chrombpnet/multitask/train_multicell_torch.py`, `chrombpnet/training/models/multitask_nucleosome_torch.py`, `chrombpnet/multitask/deviation.py`
- Commit existing approved adult-analysis files: `chrombpnet/multitask/differential_nucleosomes.py`, `chrombpnet/multitask/differential_nucleosomes_runner.py`, `tests/test_differential_nucleosomes.py`, `tests/test_adult_diffnuc_notebook.py`, `notebooks/adult_differential_nucleosomes.ipynb`, `docs/adult_differential_nucleosomes.md`, `docs/superpowers/specs/2026-08-17-adult-differential-nucleosome-explorer-design.md`, `docs/superpowers/plans/2026-08-17-adult-differential-nucleosome-explorer.md`, `pyproject.toml`, `uv.lock`

**Interfaces:**
- Consumes: the already implemented and verified dirty-worktree prerequisites.
- Produces: a reproducible branch baseline containing `HybridMultiCellModel`, `DeviationProbe`, and the adult differential caller before a feature worktree is created.

- [ ] **Step 1: Verify and commit the existing PyTorch prerequisite as one isolated change**

Run:

```bash
uv run python tests/smoke_test_multitask_torch.py
uv run python tests/smoke_test_pipeline_torch.py
```

Then stage exactly:

```bash
git add chrombpnet/multitask/evaluate_multicell.py chrombpnet/multitask/train_loop.py chrombpnet/multitask/train_multicell_torch.py chrombpnet/training/models/multitask_nucleosome_torch.py chrombpnet/multitask/deviation.py
git commit -m "feat: preserve multicell nucleosome deviations"
```

- [ ] **Step 2: Verify and commit the existing adult differential workflow separately**

Run:

```bash
uv run python -m unittest tests.test_differential_nucleosomes tests.test_adult_diffnuc_notebook -v
uv lock --check
```

Then stage exactly the adult-analysis files listed in this task and commit:

```bash
git add chrombpnet/multitask/differential_nucleosomes.py chrombpnet/multitask/differential_nucleosomes_runner.py tests/test_differential_nucleosomes.py tests/test_adult_diffnuc_notebook.py notebooks/adult_differential_nucleosomes.ipynb docs/adult_differential_nucleosomes.md docs/superpowers/specs/2026-08-17-adult-differential-nucleosome-explorer-design.md docs/superpowers/plans/2026-08-17-adult-differential-nucleosome-explorer.md pyproject.toml uv.lock
git commit -m "feat: add adult differential nucleosome workflow"
```

Do not stage the other untracked reports in the parent checkout.

- [ ] **Step 3: Commit the approved implementation plan**

```bash
git add docs/superpowers/plans/2026-08-18-differential-nucleosome-finetuning.md
git commit -m "docs: plan differential nucleosome fine-tuning"
```

- [ ] **Step 4: Ignore local feature worktrees**

Add this line to `.gitignore`:

```gitignore
.worktrees/
```

Commit it independently:

```bash
git add .gitignore
git commit -m "chore: ignore local feature worktrees"
```

- [ ] **Step 5: Create an isolated feature worktree**

```bash
git worktree add .worktrees/differential-nucleosome-finetune -b codex/differential-nucleosome-finetune
```

Run all remaining tasks with working directory `.worktrees/differential-nucleosome-finetune`. The NFS paths remain absolute and shared. The unrelated untracked reports remain only in the parent checkout.

---

### Task 1: Persist Split Provenance In Differential Artifacts

**Files:**
- Modify: `chrombpnet/multitask/differential_nucleosomes_runner.py:321-475`
- Modify: `tests/test_differential_nucleosomes.py`

**Interfaces:**
- Consumes: `call_differential_nucleosomes(..., chroms: set[str] | None, ...)`.
- Produces: `summary.json` keys `manifest`, `metadata`, and `chromosomes`; `validate_artifact_chromosomes(calls: pl.DataFrame, allowed: set[str], forbidden: set[str], label: str) -> None` will rely on these artifacts in Task 2.

- [ ] **Step 1: Write failing provenance tests**

Add assertions to the existing tiny-BigWig end-to-end test:

```python
summary = json.loads(artifacts.summary_path.read_text())
self.assertEqual(summary["chromosomes"], ["chr1"])
self.assertEqual(Path(summary["manifest"]), self.manifest.resolve())
self.assertEqual(Path(summary["metadata"]), self.metadata.resolve())
```

Add a test that passes `chroms={"chr1"}` and asserts every Parquet row is on `chr1`.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
uv run python -m unittest tests.test_differential_nucleosomes.ArtifactRunnerTests -v
```

Expected: FAIL because the three summary keys are absent.

- [ ] **Step 3: Add explicit provenance to the runner**

Before writing `summary.json`, calculate chromosomes from the filtered loci and serialize resolved input paths:

```python
summary = {
    "analysis": "adult_lineage_aware_differential_nucleosome_positioning",
    "manifest": str(Path(manifest_path).resolve()),
    "metadata": str(Path(metadata_path).resolve()),
    "chromosomes": sorted(loci["chrom"].unique().to_list()),
    # existing fields remain unchanged
}
```

- [ ] **Step 4: Verify the artifact tests and caller CLI**

Run:

```bash
uv run python -m unittest tests.test_differential_nucleosomes -v
uv run chrombpnet-call-diff-nucleosomes --help
```

Expected: all differential tests pass and CLI exits zero.

- [ ] **Step 5: Commit only Task 1 files**

```bash
git add chrombpnet/multitask/differential_nucleosomes_runner.py tests/test_differential_nucleosomes.py
git commit -m "feat: record differential artifact provenance"
```

---

### Task 2: Build Leakage-Safe Locus-Context Units

**Files:**
- Create: `chrombpnet/multitask/differential_finetune_data.py`
- Create: `tests/test_differential_finetune_data.py`

**Interfaces:**
- Consumes: differential-call Parquet, adult metadata TSV, checkpoint `cell_types`, fold chromosome sets, genome FASTA, and adult nucleosome BigWigs.
- Produces:
  - `ContextUnit(locus_id, chrom, center, tissue_context, cell_types, checkpoint_indices, observed_calls)`;
  - `validate_artifact_chromosomes(calls, allowed, forbidden, label) -> None`;
  - `map_adult_checkpoint_indices(metadata, checkpoint_cell_types) -> dict[str, int]`;
  - `build_context_units(calls, checkpoint_index, min_members=4) -> list[ContextUnit]`;
  - `PairedContextDataset.__getitem__(index) -> PairedContextExample`;
  - `ContextBalancedSampler`.

- [ ] **Step 1: Write failing split, mapping, and grouping tests**

Create tests with small Polars frames:

```python
def test_rejects_test_chromosome_in_training_artifact(self):
    calls = pl.DataFrame({"chrom": ["chr2", "chr1"]})
    with self.assertRaisesRegex(ValueError, "forbidden.*chr1"):
        validate_artifact_chromosomes(
            calls, allowed={"chr2", "chr4"}, forbidden={"chr1", "chr3", "chr6"}, label="train"
        )

def test_maps_adults_to_checkpoint_order(self):
    metadata = pl.DataFrame({
        "cell_type": ["adult_b", "adult_a"],
        "lineage": ["brain", "brain"],
        "is_adult": [True, True],
        "is_cell_line": [False, False],
    })
    self.assertEqual(
        map_adult_checkpoint_indices(metadata, ["fetal", "adult_a", "adult_b"]),
        {"adult_b": 2, "adult_a": 1},
    )
```

Add a grouping test where one locus-context has four depth-qualified rows and one has three; only the four-member unit is returned.

- [ ] **Step 2: Run the new test module and verify RED**

```bash
uv run python -m unittest tests.test_differential_finetune_data -v
```

Expected: import failure because `differential_finetune_data.py` does not exist.

- [ ] **Step 3: Implement immutable units and strict split checks**

Use this public shape:

```python
@dataclass(frozen=True)
class ContextUnit:
    locus_id: str
    chrom: str
    center: int
    tissue_context: str
    cell_types: tuple[str, ...]
    checkpoint_indices: tuple[int, ...]
    observed_calls: tuple[bool, ...]

def validate_artifact_chromosomes(
    calls: pl.DataFrame,
    *,
    allowed: set[str],
    forbidden: set[str],
    label: str,
) -> None:
    observed = set(calls["chrom"].unique().to_list())
    blocked = observed & forbidden
    outside = observed - allowed
    if blocked or outside:
        raise ValueError(
            f"{label} differential artifact has forbidden={sorted(blocked)} "
            f"outside_allowed={sorted(outside)}"
        )
```

`build_context_units` must filter `passes_depth`, group by `(locus_id, lineage)`, preserve checkpoint order, require at least four members, and retain each row's `is_differential` flag.

- [ ] **Step 4: Write failing tiny FASTA/BigWig dataset tests**

Use the existing `tests.bigwig_test_utils` helpers to construct four cell-type dyad tracks. Assert:

```python
example = dataset[0]
self.assertEqual(example.sequence.shape, (checkpoint_inputlen, 4))
self.assertEqual(example.raw_profiles.shape, (4, checkpoint_outputlen))
self.assertEqual(example.checkpoint_indices.tolist(), [1, 2, 3, 4])
self.assertEqual(example.deviation_slice.stop - example.deviation_slice.start, 501)
```

Also assert the same sequence is used for every cell by storing one sequence per example rather than duplicating it in the dataset.

- [ ] **Step 5: Implement lazy paired BigWig loading**

Use this return type:

```python
@dataclass(frozen=True)
class PairedContextExample:
    sequence: torch.Tensor
    raw_profiles: torch.Tensor
    checkpoint_indices: torch.Tensor
    observed_calls: torch.Tensor
    deviation_slice: slice
    locus_id: str
    tissue_context: str
```

`PairedContextDataset` opens FASTA and BigWigs lazily, implements `__getstate__` to remove handles for worker safety, fetches `checkpoint_outputlen` bases centered on the call, and calculates the odd 501 bp center slice as:

```python
start = checkpoint_outputlen // 2 - 501 // 2
deviation_slice = slice(start, start + 501)
```

Do not apply jitter or reverse-complement augmentation in the paired path.

- [ ] **Step 6: Implement and test context-balanced sampling**

`ContextBalancedSampler` groups dataset indices by `tissue_context`. For each draw it samples a context uniformly, then samples one of that context's units uniformly, using `np.random.default_rng(seed)`. Test deterministic equality for equal seeds and approximate equal context counts over 6,000 draws.

- [ ] **Step 7: Verify Task 2**

```bash
uv run python -m unittest tests.test_differential_finetune_data -v
uv run --with ruff ruff check chrombpnet/multitask/differential_finetune_data.py tests/test_differential_finetune_data.py
```

Expected: all tests and lint pass.

- [ ] **Step 8: Commit Task 2**

```bash
git add chrombpnet/multitask/differential_finetune_data.py tests/test_differential_finetune_data.py
git commit -m "feat: load paired differential nucleosome contexts"
```

---

### Task 3: Add The Paired Deviation Objective And Adult-Row Guard

**Files:**
- Create: `chrombpnet/multitask/differential_finetune.py`
- Create: `tests/test_differential_finetune.py`

**Interfaces:**
- Consumes: model nucleosome logits, paired raw profiles, 501 bp center slice, adult checkpoint indices, and `HybridMultiCellModel`.
- Produces:
  - `smooth_profiles(x: torch.Tensor, sigma: float) -> torch.Tensor`;
  - `normalize_profiles(x: torch.Tensor) -> torch.Tensor`;
  - `leave_one_out_deviation(probabilities: torch.Tensor) -> torch.Tensor`;
  - `paired_deviation_loss(predicted, observed, center_slice, sigma=20.0) -> torch.Tensor`;
  - `DifferentialLoss(total, profile_nll, deviation)`;
  - `AdultProfileHeadGuard`;
  - `configure_adult_profile_finetune(model, adult_indices) -> AdultProfileHeadGuard`.

- [ ] **Step 1: Write failing deviation-math tests**

```python
def test_leave_one_out_deviation(self):
    profiles = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.5, 0.5]])
    deviation = leave_one_out_deviation(profiles)
    expected_first = profiles[0] - profiles[1:].mean(0)
    torch.testing.assert_close(deviation[0], expected_first)

def test_correct_shift_has_lower_deviation_loss(self):
    observed = shifted_four_cell_profiles(shift=12)
    correct = observed.clone()
    collapsed = observed.mean(0, keepdim=True).repeat(4, 1)
    self.assertLess(
        paired_deviation_loss(correct.log(), observed, slice(0, observed.shape[-1])).item(),
        paired_deviation_loss(collapsed.log(), observed, slice(0, observed.shape[-1])).item(),
    )
```

Use strictly positive normalized synthetic profiles before `.log()`.

- [ ] **Step 2: Run objective tests and verify RED**

```bash
uv run python -m unittest tests.test_differential_finetune.DifferentialObjectiveTests -v
```

Expected: import failure for the new objective module.

- [ ] **Step 3: Implement differentiable smoothing, center normalization, and correlation**

Implement Gaussian smoothing with `torch.nn.functional.conv1d`. In `paired_deviation_loss`:

```python
pred = torch.softmax(predicted, dim=-1)[:, center_slice]
obs = observed.float()[:, center_slice]
pred = normalize_profiles(smooth_profiles(pred, sigma))
obs = normalize_profiles(smooth_profiles(obs, sigma))
pred_dev = leave_one_out_deviation(pred)
obs_dev = leave_one_out_deviation(obs)
pred_centered = pred_dev - pred_dev.mean(-1, keepdim=True)
obs_centered = obs_dev - obs_dev.mean(-1, keepdim=True)
denominator = pred_centered.norm(dim=-1) * obs_centered.norm(dim=-1)
correlation = (pred_centered * obs_centered).sum(-1) / denominator.clamp_min(1e-8)
return 1.0 - correlation.mean()
```

- [ ] **Step 4: Add the combined differential loss**

Use the existing `multinomial_nll` for the full checkpoint output. Smooth observed counts with sigma 6 while preserving each profile's sum, then return:

```python
@dataclass(frozen=True)
class DifferentialLoss:
    total: torch.Tensor
    profile_nll: torch.Tensor
    deviation: torch.Tensor

def differential_loss(logits, observed_raw, center_slice, *, alpha, lambda_dev):
    profile_target = sum_preserving_smooth(observed_raw, sigma=6.0)
    profile_nll = multinomial_nll(profile_target, logits)
    deviation = paired_deviation_loss(logits, observed_raw, center_slice, sigma=20.0)
    return DifferentialLoss(alpha * profile_nll + lambda_dev * deviation, profile_nll, deviation)
```

- [ ] **Step 5: Write failing adult-row gradient and invariant tests**

Instantiate a tiny `HybridMultiCellModel` with six cells and adult indices `[1, 4]`. After one optimizer step assert:

```python
self.assertFalse(torch.equal(before_w[1], model.nuc_profile_w[1]))
self.assertFalse(torch.equal(before_w[4], model.nuc_profile_w[4]))
torch.testing.assert_close(before_w[[0, 2, 3, 5]], model.nuc_profile_w[[0, 2, 3, 5]], rtol=0, atol=0)
torch.testing.assert_close(before_b, model.nuc_profile_b, rtol=0, atol=0)
self.assertTrue(all(not p.requires_grad for name, p in model.named_parameters() if name != "nuc_profile_w"))
```

- [ ] **Step 6: Implement gradient masks and frozen-row guard**

`configure_adult_profile_finetune` freezes every parameter, enables
`nuc_profile_w`, registers its adult-row gradient mask, and snapshots non-adult
weight rows plus the complete bias tensor. `AdultProfileHeadGuard.assert_unchanged()`
performs bitwise `torch.equal` checks after optimizer steps. The optimizer must
receive only `nuc_profile_w` and use zero weight decay.

- [ ] **Step 7: Verify Task 3**

```bash
uv run python -m unittest tests.test_differential_finetune -v
uv run --with ruff ruff check chrombpnet/multitask/differential_finetune.py tests/test_differential_finetune.py
```

Expected: objective, gradient-mask, and invariant tests pass.

- [ ] **Step 8: Commit Task 3**

```bash
git add chrombpnet/multitask/differential_finetune.py tests/test_differential_finetune.py
git commit -m "feat: add adult differential profile objective"
```

---

### Task 4: Add Differential Prediction Metrics

**Files:**
- Create: `chrombpnet/multitask/differential_finetune_metrics.py`
- Create: `tests/test_differential_finetune_metrics.py`

**Interfaces:**
- Consumes: predicted logits and observed raw profiles grouped by locus-context, observed call flags, and the 501 bp center slice.
- Produces: `DifferentialMetrics` and `score_context_predictions(...) -> DifferentialMetrics`.

- [ ] **Step 1: Write failing known-shift metric tests**

Create synthetic four-cell contexts with one profile shifted right by 15 bp. Test:

```python
metrics = score_context_predictions(perfect_logits, observed, observed_calls, center_slice)
self.assertGreater(metrics.deviation_pearson, 0.99)
self.assertGreater(metrics.effect_spearman, 0.99)
self.assertEqual(metrics.shift_direction_accuracy, 1.0)
self.assertEqual(metrics.shift_mae_bp, 0.0)
self.assertEqual(metrics.average_precision, 1.0)
```

Add zero-variance and no-positive-call tests that return `NaN` for undefined metrics rather than raising.

- [ ] **Step 2: Run metrics tests and verify RED**

```bash
uv run python -m unittest tests.test_differential_finetune_metrics -v
```

Expected: import failure for the metrics module.

- [ ] **Step 3: Implement metrics using existing scoring primitives**

Use `normalize_profiles`, `_robust_z`, `_total_variation`, and `estimate_shift` from `differential_nucleosomes.py` for NumPy evaluation. Calculate:

```python
@dataclass(frozen=True)
class DifferentialMetrics:
    deviation_pearson: float
    effect_spearman: float
    shift_direction_accuracy: float
    shift_mae_bp: float
    average_precision: float
    target_locus_rows: int
```

Average precision is implemented without a new dependency:

```python
order = np.argsort(-predicted_robust_z, kind="stable")
labels = observed_calls[order].astype(bool)
precision = np.cumsum(labels) / np.arange(1, labels.size + 1)
ap = float(precision[labels].mean()) if labels.any() else float("nan")
```

Shift direction uses only rows with observed absolute shift at least 10 bp.

- [ ] **Step 4: Verify Task 4**

```bash
uv run python -m unittest tests.test_differential_finetune_metrics -v
uv run --with ruff ruff check chrombpnet/multitask/differential_finetune_metrics.py tests/test_differential_finetune_metrics.py
```

Expected: all metric tests pass.

- [ ] **Step 5: Commit Task 4**

```bash
git add chrombpnet/multitask/differential_finetune_metrics.py tests/test_differential_finetune_metrics.py
git commit -m "feat: score predicted differential nucleosome profiles"
```

---

### Task 5: Implement Fine-Tuning And Evaluation CLI

**Files:**
- Create: `chrombpnet/multitask/fine_tune_differential_torch.py`
- Create: `tests/test_fine_tune_differential_cli.py`
- Modify: `pyproject.toml:59-67`
- Modify: `uv.lock`

**Interfaces:**
- Consumes: base hybrid checkpoint, adult manifest and metadata, fold JSON, train/validation call Parquets, genome FASTA, and Task 2-4 interfaces.
- Produces:
  - `load_hybrid_checkpoint(path, device) -> tuple[HybridMultiCellModel, dict]`;
  - `build_adult_replay_loader(...) -> DataLoader`;
  - `run_single_finetune(config) -> RunResult`;
  - `run_pilot_sweep(config, lambda_grid) -> SweepResult`;
  - `load_selected_lambda(selected_pilot_json: Path) -> float`;
  - CLI subcommands `train` and `evaluate`;
  - entry point `chrombpnet-finetune-diff-nucleosomes`.

- [ ] **Step 1: Write failing checkpoint and CLI contract tests**

Test that a tiny hybrid checkpoint reloads with exact cell order and that a conditioned/shared-head checkpoint is rejected:

```python
model, checkpoint = load_hybrid_checkpoint(path, torch.device("cpu"))
self.assertEqual(checkpoint["cell_types"], ["a", "b", "c", "d"])
self.assertIsInstance(model, HybridMultiCellModel)
```

Parse the CLI with `train --help` and `evaluate --help`; assert required options include checkpoint, manifest, metadata, fold, calls, genome, and output directory.

- [ ] **Step 2: Run CLI tests and verify RED**

```bash
uv run python -m unittest tests.test_fine_tune_differential_cli -v
```

Expected: import failure for the CLI module.

- [ ] **Step 3: Implement strict hybrid checkpoint loading**

Instantiate from stored checkpoint args:

```python
if not checkpoint["args"].get("hybrid"):
    raise ValueError("differential fine-tuning requires a HybridMultiCellModel checkpoint")
model = HybridMultiCellModel(
    len(checkpoint["cell_types"]),
    args["inputlen"],
    args["outputlen"],
    args["filters"],
    args["n_dil_layers"],
    args.get("embed_dim", 32),
    cond_mode=args.get("cond_mode", "film"),
)
model.load_state_dict(checkpoint["model_state_dict"])
```

- [ ] **Step 4: Implement adult replay loading with checkpoint indices**

Build one `MultiTaskRegionDataset` per adult manifest row on training chromosomes. Wrap each dataset with a fixed checkpoint index rather than the adult-manifest row number. Replay batches retain `(sequence, acc, acc_count, nuc, nuc_count, ct_idx)`, but the fine-tuning step uses only sequence, nucleosome profile, and checkpoint index.

- [ ] **Step 5: Write failing one-step and preservation-gate tests**

Use tiny BigWigs and a tiny hybrid model. Run one step with one paired context and one replay batch. Assert loss components are finite, adult rows change, non-adult rows do not, and accessibility outputs before/after are identical.

Add pure gate tests:

```python
self.assertTrue(passes_gates(base, tuned_good))
self.assertFalse(passes_gates(base, tuned_with_profile_drop_over_0_01))
self.assertFalse(passes_gates(base, tuned_with_nll_increase_over_2_percent))
```

- [ ] **Step 6: Implement one optimizer step**

For each step:

1. fetch one context-balanced paired example;
2. repeat its sequence for its checkpoint indices and run the model;
3. calculate Task 3 differential profile and deviation losses;
4. fetch one adult replay batch and calculate existing nucleosome profile NLL;
5. sum `replay_nll + differential.total`, backpropagate, clip at 1.0, step AdamW, and run the frozen-row guard;
6. update the 100-step warmup and cosine-to-`1e-6` schedule;
7. log component losses and gradient norm.

- [ ] **Step 7: Implement validation and gate selection**

At each validation interval, score every validation context unit with Task 4 metrics and score a fixed ordinary adult validation subset. Save a checkpoint only when all preservation gates pass. Rank passing checkpoints by deviation Pearson, then shift-direction accuracy.

Store:

```text
pilot_lambda_0.25.pt / .json
pilot_lambda_1.pt / .json
pilot_lambda_4.pt / .json
selected_pilot.json
final.pt
final_validation.json
```

The sweep runs each lambda for 2,000 steps from the same base checkpoint, selects the best passing lambda, then restarts from the base checkpoint for 10,000 steps.

The `train` subcommand accepts either `--lambda-dev-grid` for pilot selection or
`--selected-pilot-json` for a full run. These options are mutually exclusive. The
selected-pilot path reads the exact passing lambda from Task 8 and still reloads the
original `--checkpoint`; it never resumes pilot weights.

- [ ] **Step 8: Implement W&B logging and artifact metadata**

Log under project `jg_experiments` by default. Include base checkpoint path and hash, call-artifact paths and hashes, fold chromosome lists, adult cell list, optimizer values, lambda, losses, metrics, gate status, and selected step. W&B failure emits a warning and continues local training.

- [ ] **Step 9: Register and lock the CLI**

Add:

```toml
chrombpnet-finetune-diff-nucleosomes = "chrombpnet.multitask.fine_tune_differential_torch:main"
```

Then run:

```bash
uv lock
uv lock --check
uv run chrombpnet-finetune-diff-nucleosomes --help
```

- [ ] **Step 10: Verify Task 5**

```bash
uv run python -m unittest tests.test_fine_tune_differential_cli -v
uv run --with ruff ruff check chrombpnet/multitask/fine_tune_differential_torch.py tests/test_fine_tune_differential_cli.py
```

Expected: CLI, one-step, gate, and W&B-fallback tests pass.

- [ ] **Step 11: Commit Task 5**

```bash
git add chrombpnet/multitask/fine_tune_differential_torch.py tests/test_fine_tune_differential_cli.py pyproject.toml uv.lock
git commit -m "feat: fine-tune adult differential nucleosome profiles"
```

---

### Task 6: Add End-To-End Synthetic Fine-Tuning Verification

**Files:**
- Create: `tests/smoke_test_differential_finetune.py`
- Modify: `docs/adult_differential_nucleosomes.md`

**Interfaces:**
- Consumes: all fine-tuning modules and CLI from Tasks 2-5.
- Produces: a CPU-safe synthetic smoke test proving learning and preservation end to end.

- [ ] **Step 1: Write the synthetic integration test**

Construct six cell types, mark four as adult in one context, and generate train/validation loci where one adult target has a sequence-dependent 12 bp shift. Start from a tiny hybrid checkpoint and run 80 fine-tuning steps with seed 0, AdamW learning rate `1e-2`, zero weight decay, no warmup, one paired unit per step, and one replay batch per step. Assert:

```python
self.assertGreater(tuned_metrics.deviation_pearson, base_metrics.deviation_pearson)
self.assertGreaterEqual(tuned_metrics.shift_direction_accuracy, base_metrics.shift_direction_accuracy)
torch.testing.assert_close(base_accessibility, tuned_accessibility, rtol=0, atol=0)
torch.testing.assert_close(base_nonadult_rows, tuned_nonadult_rows, rtol=0, atol=0)
```

- [ ] **Step 2: Run the smoke test**

```bash
uv run python tests/smoke_test_differential_finetune.py
```

Expected: `DIFFERENTIAL_FINETUNE_SMOKE_OK` with improved deviation correlation and unchanged accessibility/non-adult assertions.

- [ ] **Step 3: Document exact train, validate, and evaluate commands**

Update `docs/adult_differential_nucleosomes.md` with the NFS paths and CLI commands from Tasks 7-9. State clearly that the existing 553 test loci are not accepted by the `train` subcommand.

- [ ] **Step 4: Run the complete local verification suite**

```bash
uv run python -m unittest tests.test_differential_nucleosomes tests.test_differential_finetune_data tests.test_differential_finetune tests.test_differential_finetune_metrics tests.test_fine_tune_differential_cli -v
uv run python tests/smoke_test_multitask_torch.py
uv run python tests/smoke_test_pipeline_torch.py
uv run python tests/smoke_test_differential_finetune.py
uv run --with ruff ruff check chrombpnet/multitask tests
uv run --with ruff ruff format --check chrombpnet/multitask tests
uv lock --check
```

Expected: all tests, smoke checks, lint, formatting, and lock validation pass.

- [ ] **Step 5: Commit Task 6**

```bash
git add tests/smoke_test_differential_finetune.py docs/adult_differential_nucleosomes.md
git commit -m "test: verify differential nucleosome fine-tuning"
```

---

### Task 7: Generate Real Train And Validation Differential Artifacts

**Files:**
- Write NFS outputs only:
  - `/mnt/data/jganbat/scatac_corpus/run/adult_diffnuc_train/`
  - `/mnt/data/jganbat/scatac_corpus/run/adult_diffnuc_valid/`

**Interfaces:**
- Consumes: adult manifest, adult metadata, staged peaks and dyad BigWigs, fold JSON.
- Produces: split-specific call Parquets, summaries, BEDs, and profile caches.

- [ ] **Step 1: Run training-chromosome differential discovery**

```bash
uv run chrombpnet-call-diff-nucleosomes \
  --manifest /mnt/data/jganbat/scatac_corpus/staged/manifest_adult346.tsv \
  --metadata /mnt/data/jganbat/scatac_corpus/staged/adult_cell_metadata.tsv \
  --out-dir /mnt/data/jganbat/scatac_corpus/run/adult_diffnuc_train \
  --chroms chr2 chr4 chr5 chr7 chr9 chr10 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19 chr21 chr22 chrX chrY \
  --per-cell-peak-limit 50 --max-loci 5000 --merge-bp 100 \
  --window 501 --smooth-sigma 20 --max-shift 100 \
  --min-counts 20 --min-effect 0.05 --min-robust-z 3 \
  --min-lineage-comparators 3 --chunk-size 32 --cache-loci 100
```

- [ ] **Step 2: Run validation-chromosome differential discovery**

```bash
uv run chrombpnet-call-diff-nucleosomes \
  --manifest /mnt/data/jganbat/scatac_corpus/staged/manifest_adult346.tsv \
  --metadata /mnt/data/jganbat/scatac_corpus/staged/adult_cell_metadata.tsv \
  --out-dir /mnt/data/jganbat/scatac_corpus/run/adult_diffnuc_valid \
  --chroms chr8 chr20 --per-cell-peak-limit 50 --max-loci 2000 --merge-bp 100 \
  --window 501 --smooth-sigma 20 --max-shift 100 \
  --min-counts 20 --min-effect 0.05 --min-robust-z 3 \
  --min-lineage-comparators 3 --chunk-size 32 --cache-loci 100
```

- [ ] **Step 3: Validate split purity and cohort counts**

Run a Polars validation script through the Task 2 public functions. Assert 244 cell types, no `Fetal` or `Pierce2021` names, train chromosomes exactly within the train set, validation chromosomes exactly within `{chr8, chr20}`, and zero overlap with `{chr1, chr3, chr6}`.

- [ ] **Step 4: Record real artifact summaries**

Write `/mnt/data/jganbat/scatac_corpus/run/adult_diffnuc_artifact_index.json` containing paths, file hashes, candidate counts, target-locus rows, primary call counts, unique called loci, tissue-context counts, and generation parameters.

No Git commit is created for NFS outputs.

---

### Task 8: Run The Three-Lambda Bounded Pilot

> Historical execution record. The final-review correction protocol below
> supersedes these runtime paths without modifying their artifacts.

**Files:**
- Write NFS outputs: `/mnt/data/jganbat/scatac_corpus/run/diffnuc_finetune/pilot/`

**Interfaces:**
- Consumes: base checkpoint, Task 7 train/validation Parquets, adult manifest and metadata.
- Produces: three pilot checkpoints/reports and `selected_pilot.json`.

- [ ] **Step 1: Launch the pilot sweep with W&B**

```bash
uv run chrombpnet-finetune-diff-nucleosomes train \
  --checkpoint /mnt/data/jganbat/scatac_corpus/run/model_346_hybrid_long.pt \
  --manifest /mnt/data/jganbat/scatac_corpus/staged/manifest_adult346.tsv \
  --manifest-root /mnt/data/jganbat/scatac_corpus/staged \
  --metadata /mnt/data/jganbat/scatac_corpus/staged/adult_cell_metadata.tsv \
  --genome /mnt/data/jganbat/scatac_corpus/refs/hg38.fa \
  --fold-json /mnt/data/jganbat/scatac_corpus/refs/folds/fold_0.json \
  --train-calls /mnt/data/jganbat/scatac_corpus/run/adult_diffnuc_train/differential_nucleosomes.parquet \
  --valid-calls /mnt/data/jganbat/scatac_corpus/run/adult_diffnuc_valid/differential_nucleosomes.parquet \
  --out-dir /mnt/data/jganbat/scatac_corpus/run/diffnuc_finetune/pilot \
  --lambda-dev-grid 0.25 1.0 4.0 --pilot-steps 2000 --full-steps 0 \
  --learning-rate 1e-5 --warmup-steps 100 --grad-clip 1.0 \
  --validate-every 250 --seed 0 --device cuda --amp \
  --wandb --wandb-project jg_experiments --wandb-run-name hybrid346_diffnuc_pilot
```

- [ ] **Step 2: Monitor runtime health**

Confirm W&B and local logs show finite replay NLL, differential profile NLL, deviation loss, validation deviation Pearson, ordinary-profile gates, gradient norm, learning rate, and frozen-row checks. Abort a candidate immediately on non-finite loss or invariant failure.

- [ ] **Step 3: Audit automatic gate selection**

Verify `selected_pilot.json` chooses the highest validation deviation Pearson among candidates that satisfy all gates. Recompute gate decisions from the three JSON reports in an independent read-only command and compare the selected lambda and step.

- [ ] **Step 4: Stop if no pilot passes**

If no candidate passes, write a pilot failure report and keep the base checkpoint. Do not run Task 9. Diagnose whether the failure is deviation learning, ordinary-profile regression, or an invariant violation before proposing a revised design.

No Git commit is created for run outputs.

---

### Task 9: Run The Selected Full Fine-Tune And Untouched Test Evaluation

> Historical execution record. Any post-fix challenge use is a corrective
> re-evaluation, not a new untouched evaluation.

**Files:**
- Write NFS outputs: `/mnt/data/jganbat/scatac_corpus/run/diffnuc_finetune/final/`
- Modify: `docs/adult_differential_nucleosomes.md`
- Create: `docs/differential_nucleosome_finetuning_results.md`

**Interfaces:**
- Consumes: selected passing lambda from Task 8, base checkpoint, train/validation calls, and the untouched test Parquet.
- Produces: final checkpoint, validation report, one-time test report, and updated project documentation.

- [ ] **Step 1: Run the selected configuration from the base checkpoint**

```bash
uv run chrombpnet-finetune-diff-nucleosomes train \
  --checkpoint /mnt/data/jganbat/scatac_corpus/run/model_346_hybrid_long.pt \
  --manifest /mnt/data/jganbat/scatac_corpus/staged/manifest_adult346.tsv \
  --manifest-root /mnt/data/jganbat/scatac_corpus/staged \
  --metadata /mnt/data/jganbat/scatac_corpus/staged/adult_cell_metadata.tsv \
  --genome /mnt/data/jganbat/scatac_corpus/refs/hg38.fa \
  --fold-json /mnt/data/jganbat/scatac_corpus/refs/folds/fold_0.json \
  --train-calls /mnt/data/jganbat/scatac_corpus/run/adult_diffnuc_train/differential_nucleosomes.parquet \
  --valid-calls /mnt/data/jganbat/scatac_corpus/run/adult_diffnuc_valid/differential_nucleosomes.parquet \
  --out-dir /mnt/data/jganbat/scatac_corpus/run/diffnuc_finetune/final \
  --selected-pilot-json /mnt/data/jganbat/scatac_corpus/run/diffnuc_finetune/pilot/selected_pilot.json \
  --pilot-steps 0 --full-steps 10000 --learning-rate 1e-5 \
  --warmup-steps 100 --grad-clip 1.0 --validate-every 250 \
  --seed 0 --device cuda --amp --wandb --wandb-project jg_experiments \
  --wandb-run-name hybrid346_diffnuc_final
```

- [ ] **Step 2: Verify final validation gates before opening test data**

Require the final selected checkpoint to pass all ordinary-profile, accessibility, frozen-row, and shift-direction gates. Verify the stored checkpoint parent is the original base checkpoint rather than a pilot checkpoint.

- [ ] **Step 3: Evaluate once on the untouched 553-locus challenge set**

```bash
uv run chrombpnet-finetune-diff-nucleosomes evaluate \
  --base-checkpoint /mnt/data/jganbat/scatac_corpus/run/model_346_hybrid_long.pt \
  --tuned-checkpoint /mnt/data/jganbat/scatac_corpus/run/diffnuc_finetune/final/final.pt \
  --validation-report /mnt/data/jganbat/scatac_corpus/run/diffnuc_finetune/final/final_validation.json \
  --manifest /mnt/data/jganbat/scatac_corpus/staged/manifest_adult346.tsv \
  --manifest-root /mnt/data/jganbat/scatac_corpus/staged \
  --metadata /mnt/data/jganbat/scatac_corpus/staged/adult_cell_metadata.tsv \
  --genome /mnt/data/jganbat/scatac_corpus/refs/hg38.fa \
  --fold-json /mnt/data/jganbat/scatac_corpus/refs/folds/fold_0.json \
  --calls /mnt/data/jganbat/scatac_corpus/run/adult_diffnuc_pilot_rescore/differential_nucleosomes.parquet \
  --split test \
  --out-json /mnt/data/jganbat/scatac_corpus/run/diffnuc_finetune/final/test_comparison.json \
  --device cuda --amp
```

- [ ] **Step 4: Update the validation report**

Document base versus tuned deviation Pearson, effect Spearman, shift-direction accuracy, shift MAE, differential AUPRC, ordinary nucleosome metrics, accessibility invariance, and frozen-row verification. State that the 553 loci are an ascertained differential challenge set, not an unbiased genome-wide sample.

- [ ] **Step 5: Run final repository verification**

```bash
uv run python -m unittest tests.test_differential_nucleosomes tests.test_differential_finetune_data tests.test_differential_finetune tests.test_differential_finetune_metrics tests.test_fine_tune_differential_cli -v
uv run python tests/smoke_test_multitask_torch.py
uv run python tests/smoke_test_pipeline_torch.py
uv run python tests/smoke_test_differential_finetune.py
uv run --with ruff ruff check chrombpnet/multitask tests
uv run --with ruff ruff format --check chrombpnet/multitask tests
uv lock --check
git diff --check
```

Expected: all tests and checks pass; the final JSON reports exist and contain finite metrics.

- [ ] **Step 6: Commit documentation only after the reports exist**

```bash
git add docs/differential_nucleosome_finetuning_results.md docs/adult_differential_nucleosomes.md
git commit -m "docs: report differential nucleosome fine-tuning"
```

## Final-Review Scientific Correction Protocol

Preserve `diffnuc_finetune/pilot`, `pilot_ampfix`, and `final` byte-identically
as superseded audit evidence. After code approval, rerun the exact three-lambda
sweep from the original base checkpoint into `pilot_metricfix`. If and only if
that selector passes, restart the exact 10,000-step configuration from the original
base checkpoint into `final_metricfix`, consuming only
`pilot_metricfix/selected_pilot.json`. Then run the challenge evaluation once as
a documented corrective re-evaluation, with
`final_metricfix/final_validation.json` supplied through the required
`--validation-report` binding.

The CUDA commands retain `--amp` only as a CLI compatibility and provenance flag.
This feature currently performs no active mixed-precision model forward and uses
no GradScaler scaling; every model forward is float32 to preserve profile-logit
resolution. Candidate, final, and evaluation schema-v2 JSON must contain pooled
endpoint counts, exact checkpoint hashes, and protocol bindings; prior JSON without
these fields is not accepted for the corrective evaluation.
