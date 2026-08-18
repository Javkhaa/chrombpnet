"""Fine-tune and evaluate adult differential nucleosome profile heads."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from chrombpnet.multitask.differential_finetune import (
    AdultProfileHeadGuard,
    configure_adult_profile_finetune,
    differential_loss,
)
from chrombpnet.multitask.differential_finetune_data import (
    ContextBalancedSampler,
    PairedContextDataset,
    PairedContextExample,
    build_context_units,
    map_adult_checkpoint_indices,
    validate_artifact_chromosomes,
)
from chrombpnet.multitask.differential_finetune_metrics import (
    DifferentialMetrics,
    DifferentialRankingInputs,
    pooled_average_precision,
    score_context_predictions,
    score_context_ranking_inputs,
)
from chrombpnet.multitask.differential_nucleosomes import validate_adult_cohort
from chrombpnet.multitask.differential_nucleosomes_runner import (
    read_manifest,
    read_metadata,
)
from chrombpnet.multitask.torch_data import MultiTaskRegionDataset
from chrombpnet.multitask.train_multitask_torch import load_regions
from chrombpnet.training.models.multitask_nucleosome_torch import (
    HybridMultiCellModel,
    multinomial_nll,
)

APPROVED_TRAIN_CHROMOSOMES = frozenset(
    {
        "chr2",
        "chr4",
        "chr5",
        "chr7",
        "chr9",
        "chr10",
        "chr11",
        "chr12",
        "chr13",
        "chr14",
        "chr15",
        "chr16",
        "chr17",
        "chr18",
        "chr19",
        "chr21",
        "chr22",
        "chrX",
        "chrY",
    }
)
APPROVED_VALID_CHROMOSOMES = frozenset({"chr8", "chr20"})
APPROVED_TEST_CHROMOSOMES = frozenset({"chr1", "chr3", "chr6"})
APPROVED_PILOT_LAMBDAS = (0.25, 1.0, 4.0)


@dataclass(frozen=True)
class StepResult:
    total_loss: float
    replay_nll: float
    differential_profile_nll: float
    deviation_loss: float
    gradient_norm: float


@dataclass(frozen=True)
class FineTuneConfig:
    checkpoint: Path
    manifest: Path
    metadata: Path
    fold_json: Path
    train_calls: Path
    valid_calls: Path
    genome: Path
    out_dir: Path
    manifest_root: Path | None = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    amp: bool = False
    lambda_dev: float = 1.0
    alpha: float = 1.0
    steps: int = 10_000
    pilot_steps: int = 2_000
    learning_rate: float = 1e-5
    min_learning_rate: float = 1e-6
    warmup_steps: int = 100
    grad_clip: float = 1.0
    validate_every: int = 250
    replay_batch_size: int = 64
    validation_replay_examples: int = 2_000
    num_workers: int = 0
    seed: int = 0
    output_name: str = "final"
    wandb: bool = False
    wandb_project: str = "jg_experiments"
    wandb_run_name: str | None = None


@dataclass(frozen=True)
class RunResult:
    lambda_dev: float
    selected_step: int | None
    checkpoint_path: Path | None
    report_path: Path
    validation: dict[str, object]
    passes_gates: bool


@dataclass(frozen=True)
class SweepResult:
    selected_lambda: float | None
    runs: tuple[RunResult, ...]
    selection_path: Path


class WandbRun:
    """A W&B run that degrades to local-only logging after any client failure."""

    def __init__(self, run=None) -> None:
        self.run = run

    def log(self, values: Mapping[str, object], *, step: int | None = None) -> None:
        if self.run is None:
            return
        try:
            self.run.log(dict(values), step=step)
        except Exception as error:  # noqa: BLE001 - logging must never abort training
            warnings.warn(
                f"W&B logging failed; continuing local training: {error}",
                RuntimeWarning,
                stacklevel=2,
            )
            self.run = None

    def finish(self) -> None:
        if self.run is None:
            return
        try:
            self.run.finish()
        except Exception as error:  # noqa: BLE001 - finalization is best effort
            warnings.warn(
                f"W&B finish failed; local artifacts are complete: {error}",
                RuntimeWarning,
                stacklevel=2,
            )
        finally:
            self.run = None

    def update_config(self, values: Mapping[str, object]) -> None:
        if self.run is None:
            return
        try:
            self.run.config.update(dict(values), allow_val_change=True)
        except Exception as error:  # noqa: BLE001 - logging must never abort training
            warnings.warn(
                f"W&B config update failed; continuing local training: {error}",
                RuntimeWarning,
                stacklevel=2,
            )
            self.run = None


class _CleanupStack:
    def __init__(self) -> None:
        self._callbacks: list[tuple[str, object]] = []

    def callback(self, label: str, callback) -> None:
        self._callbacks.append((label, callback))

    def close(self) -> None:
        while self._callbacks:
            label, callback = self._callbacks.pop()
            try:
                callback()
            except Exception as error:  # noqa: BLE001 - cleanup must continue
                warnings.warn(
                    f"cleanup failed for {label}: {error}",
                    RuntimeWarning,
                    stacklevel=2,
                )


def start_wandb_run(
    *,
    enabled: bool,
    project: str,
    name: str | None,
    config: Mapping[str, object],
) -> WandbRun:
    """Start W&B without allowing service failures to abort local training."""
    if not enabled:
        return WandbRun()
    try:
        import wandb

        return WandbRun(wandb.init(project=project, name=name, config=dict(config)))
    except Exception as error:  # noqa: BLE001 - W&B is an optional boundary
        warnings.warn(
            f"W&B initialization failed; continuing local training: {error}",
            RuntimeWarning,
            stacklevel=2,
        )
        return WandbRun()


def learning_rate_for_step(
    step: int,
    total_steps: int,
    warmup_steps: int,
    peak_learning_rate: float,
    min_learning_rate: float,
) -> float:
    """Return linear-warmup then cosine-decay learning rate for a 1-based step."""
    if step < 1 or total_steps < 1 or step > total_steps:
        raise ValueError("step must be within 1..total_steps")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if warmup_steps and step <= warmup_steps:
        return peak_learning_rate * step / warmup_steps
    decay_steps = total_steps - warmup_steps
    if decay_steps <= 0:
        return peak_learning_rate
    progress = (step - warmup_steps) / decay_steps
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_learning_rate + (peak_learning_rate - min_learning_rate) * cosine


def assert_production_cardinalities(
    checkpoint_cell_types: list[str], adult_metadata: pl.DataFrame
) -> None:
    """Enforce the verified production cohort sizes before generic name mapping."""
    if len(checkpoint_cell_types) != 346 or len(set(checkpoint_cell_types)) != 346:
        raise ValueError(
            "production checkpoint must have exactly 346 unique cell types"
        )
    required = {"cell_type", "is_adult", "is_cell_line"}
    missing = sorted(required.difference(adult_metadata.columns))
    if missing:
        raise ValueError(
            f"production metadata missing required columns: {', '.join(missing)}"
        )
    normalized = adult_metadata
    for column in ("is_adult", "is_cell_line"):
        if normalized.schema[column] == pl.Boolean:
            continue
        values = pl.col(column).cast(pl.String).str.to_lowercase().str.strip_chars()
        valid = ["0", "1", "false", "no", "true", "yes"]
        if normalized.filter(~values.is_in(valid)).height:
            raise ValueError(f"production metadata has invalid {column} values")
        normalized = normalized.with_columns(
            values.is_in(["1", "true", "yes"]).alias(column)
        )
    adult_rows = normalized.filter(pl.col("is_adult") & ~pl.col("is_cell_line"))
    if adult_rows["cell_type"].n_unique() != 244 or adult_rows.height != 244:
        raise ValueError(
            "production metadata must have exactly 244 unique adult cell types"
        )


def _adult_metadata_cell_types(metadata: pl.DataFrame) -> list[str]:
    required = {"cell_type", "is_adult", "is_cell_line"}
    missing = sorted(required.difference(metadata.columns))
    if missing:
        raise ValueError(f"metadata missing required columns: {', '.join(missing)}")
    normalized = metadata
    for column in ("is_adult", "is_cell_line"):
        if normalized.schema[column] == pl.Boolean:
            continue
        values = pl.col(column).cast(pl.String).str.to_lowercase().str.strip_chars()
        normalized = normalized.with_columns(
            values.is_in(["1", "true", "t", "yes", "y"]).alias(column)
        )
    adults = normalized.filter(pl.col("is_adult") & ~pl.col("is_cell_line"))
    if adults["cell_type"].n_unique() != adults.height:
        raise ValueError("adult metadata contains duplicate cell_type rows")
    return [str(value) for value in adults["cell_type"].to_list()]


def validate_adult_manifest_coverage(
    adult_manifest: pl.DataFrame, metadata: pl.DataFrame
) -> None:
    """Require one manifest row for every adult metadata cell type, and no others."""
    if adult_manifest["cell_type"].n_unique() != adult_manifest.height:
        raise ValueError("adult manifest contains duplicate cell_type rows")
    manifest_names = set(adult_manifest["cell_type"].to_list())
    metadata_names = set(_adult_metadata_cell_types(metadata))
    missing = sorted(metadata_names - manifest_names)
    extra = sorted(manifest_names - metadata_names)
    if missing or extra:
        raise ValueError(
            "adult manifest does not exactly cover adult metadata: "
            f"missing={missing} extra={extra}"
        )


def validate_fold_boundaries(fold: Mapping[str, object]) -> None:
    """Require the approved train/validation/test chromosome partition exactly."""
    expected = {
        "train": APPROVED_TRAIN_CHROMOSOMES,
        "valid": APPROVED_VALID_CHROMOSOMES,
        "test": APPROVED_TEST_CHROMOSOMES,
    }
    observed: dict[str, set[str]] = {}
    for role, required in expected.items():
        value = fold.get(role)
        if not isinstance(value, list) or not all(
            isinstance(item, str) for item in value
        ):
            raise ValueError(f"fold role {role!r} must be a chromosome list")
        if len(value) != len(set(value)):
            raise ValueError(f"fold role {role!r} contains duplicate chromosomes")
        observed[role] = set(value)
        if observed[role] != set(required):
            raise ValueError(
                f"fold role {role!r} does not match approved boundary: "
                f"expected={sorted(required)} observed={sorted(observed[role])}"
            )
    if set(fold) != set(expected):
        raise ValueError(f"fold must contain exactly these roles: {sorted(expected)}")
    for first, second in (
        ("train", "valid"),
        ("train", "test"),
        ("valid", "test"),
    ):
        overlap = observed[first] & observed[second]
        if overlap:
            raise ValueError(
                f"fold roles {first!r} and {second!r} overlap: {sorted(overlap)}"
            )


def validate_call_splits(
    train_calls: pl.DataFrame,
    valid_calls: pl.DataFrame,
    train_chromosomes: set[str],
) -> None:
    """Reject gradient-bearing test data and any train/validation artifact mixing."""
    test_chromosomes = set(APPROVED_TEST_CHROMOSOMES)
    validation_chromosomes = set(APPROVED_VALID_CHROMOSOMES)
    blocked_fold = train_chromosomes & (test_chromosomes | validation_chromosomes)
    if blocked_fold:
        raise ValueError(
            f"fold training chromosomes include forbidden chromosomes: "
            f"{sorted(blocked_fold)}"
        )
    validate_artifact_chromosomes(
        train_calls,
        allowed=train_chromosomes,
        forbidden=test_chromosomes | validation_chromosomes,
        label="train",
    )
    validate_artifact_chromosomes(
        valid_calls,
        allowed=validation_chromosomes,
        forbidden=test_chromosomes | train_chromosomes,
        label="validation",
    )
    identity_columns = {"locus_id", "center", "lineage"}
    if identity_columns.issubset(train_calls.columns) and identity_columns.issubset(
        valid_calls.columns
    ):
        train_keys = set(
            train_calls.select(sorted(identity_columns)).unique().iter_rows()
        )
        valid_keys = set(
            valid_calls.select(sorted(identity_columns)).unique().iter_rows()
        )
        if train_keys & valid_keys:
            raise ValueError("train and validation differential artifacts overlap")


def aggregate_differential_metrics(
    metrics: list[DifferentialMetrics],
    ranking_inputs: list[DifferentialRankingInputs],
) -> dict[str, float | int]:
    """Weight context metrics by target rows and pool AP over every target row."""
    if len(metrics) != len(ranking_inputs):
        raise ValueError("differential metrics and ranking inputs must align")
    total_rows = sum(metric.target_locus_rows for metric in metrics)

    def weighted(name: str) -> float:
        pairs = [
            (float(getattr(metric, name)), metric.target_locus_rows)
            for metric in metrics
            if math.isfinite(float(getattr(metric, name)))
        ]
        denominator = sum(weight for _, weight in pairs)
        if denominator == 0:
            return float("nan")
        return sum(value * weight for value, weight in pairs) / denominator

    return {
        "deviation_pearson": weighted("deviation_pearson"),
        "effect_spearman": weighted("effect_spearman"),
        "shift_direction_accuracy": weighted("shift_direction_accuracy"),
        "shift_mae_bp": weighted("shift_mae_bp"),
        "average_precision": pooled_average_precision(ranking_inputs),
        "target_locus_rows": total_rows,
        "context_units": len(metrics),
    }


def _json_value(value):
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            _json_value(dict(payload)), indent=2, sort_keys=True, allow_nan=False
        )
        + "\n"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class _PreparedRun:
    model: HybridMultiCellModel
    checkpoint: dict
    adult_manifest: pl.DataFrame
    checkpoint_index: dict[str, int]
    train_dataset: PairedContextDataset
    valid_dataset: PairedContextDataset
    train_replay: DataLoader
    valid_replay: DataLoader
    fold: dict[str, list[str]]


def _prepare_run(
    config: FineTuneConfig,
    device: torch.device,
    cleanup: _CleanupStack | None = None,
) -> _PreparedRun:
    model, checkpoint = load_hybrid_checkpoint(config.checkpoint, device)
    checkpoint_cell_types = checkpoint["cell_types"]
    manifest = read_manifest(config.manifest, config.manifest_root)
    metadata = read_metadata(config.metadata)
    adult_manifest = validate_adult_cohort(manifest, metadata)
    assert_production_cardinalities(checkpoint_cell_types, metadata)
    validate_adult_manifest_coverage(adult_manifest, metadata)
    checkpoint_index = map_adult_checkpoint_indices(
        adult_manifest, checkpoint_cell_types
    )
    fold = json.loads(config.fold_json.read_text())
    validate_fold_boundaries(fold)
    train_chromosomes = set(fold["train"])
    train_calls = pl.read_parquet(config.train_calls)
    valid_calls = pl.read_parquet(config.valid_calls)
    validate_call_splits(train_calls, valid_calls, train_chromosomes)
    train_units = build_context_units(train_calls, checkpoint_index)
    valid_units = build_context_units(valid_calls, checkpoint_index)
    if not train_units:
        raise ValueError("training artifact has no eligible paired context units")
    if not valid_units:
        raise ValueError("validation artifact has no eligible paired context units")

    args = checkpoint["args"]
    nuc_bigwigs = {
        str(row["cell_type"]): str(row["nuc_bw"])
        for row in adult_manifest.iter_rows(named=True)
    }
    train_dataset = PairedContextDataset(
        train_units,
        config.genome,
        nuc_bigwigs,
        args["inputlen"],
        args["outputlen"],
    )
    if cleanup is not None:
        cleanup.callback("training paired dataset", train_dataset.close)
    valid_dataset = PairedContextDataset(
        valid_units,
        config.genome,
        nuc_bigwigs,
        args["inputlen"],
        args["outputlen"],
    )
    if cleanup is not None:
        cleanup.callback("validation paired dataset", valid_dataset.close)
    train_replay = build_adult_replay_loader(
        adult_manifest,
        checkpoint_index,
        train_chromosomes,
        config.genome,
        inputlen=args["inputlen"],
        outputlen=args["outputlen"],
        batch_size=config.replay_batch_size,
        num_workers=config.num_workers,
        shuffle=True,
        seed=config.seed,
    )
    if cleanup is not None:
        cleanup.callback(
            "training replay loader", lambda: _close_replay_loader(train_replay)
        )
    valid_replay = build_adult_replay_loader(
        adult_manifest,
        checkpoint_index,
        {"chr8", "chr20"},
        config.genome,
        inputlen=args["inputlen"],
        outputlen=args["outputlen"],
        batch_size=config.replay_batch_size,
        num_workers=config.num_workers,
        shuffle=False,
        seed=config.seed + 1,
        max_examples=config.validation_replay_examples,
    )
    if cleanup is not None:
        cleanup.callback(
            "validation replay loader", lambda: _close_replay_loader(valid_replay)
        )
    return _PreparedRun(
        model=model,
        checkpoint=checkpoint,
        adult_manifest=adult_manifest,
        checkpoint_index=checkpoint_index,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        train_replay=train_replay,
        valid_replay=valid_replay,
        fold=fold,
    )


def _forward_model_float32(model, sequence: torch.Tensor, indices: torch.Tensor):
    """Run profile heads outside autocast to preserve positional logit signal."""
    with torch.autocast(sequence.device.type, dtype=torch.bfloat16, enabled=False):
        return model(sequence, indices)


def _score_differential(
    model: HybridMultiCellModel,
    dataset: PairedContextDataset,
    device: torch.device,
    amp: bool,
) -> dict[str, float | int]:
    metrics = []
    rankings = []
    model.eval()
    with torch.no_grad():
        for example in dataset:
            indices = example.checkpoint_indices.to(device)
            sequence = example.sequence.to(device)
            repeated = sequence.unsqueeze(0).expand(indices.numel(), -1, -1)
            logits = _forward_model_float32(model, repeated, indices)[2]
            predicted = logits.cpu().numpy()
            observed = example.raw_profiles.numpy()
            calls = example.observed_calls.numpy()
            metrics.append(
                score_context_predictions(
                    predicted,
                    observed,
                    calls,
                    example.deviation_slice,
                )
            )
            rankings.append(
                score_context_ranking_inputs(
                    predicted,
                    observed,
                    calls,
                    example.deviation_slice,
                )
            )
    return aggregate_differential_metrics(metrics, rankings)


def _centered_pearson(
    predicted_row: torch.Tensor, observed_row: torch.Tensor
) -> float | None:
    """Return a finite float64 Pearson correlation or None for undefined rows."""
    predicted = predicted_row.detach().to(dtype=torch.float64)
    observed = observed_row.detach().to(dtype=torch.float64)
    if not torch.isfinite(predicted).all() or not torch.isfinite(observed).all():
        return None
    predicted = predicted - predicted.mean()
    observed = observed - observed.mean()
    denominator = torch.linalg.vector_norm(predicted) * torch.linalg.vector_norm(
        observed
    )
    if not torch.isfinite(denominator) or denominator <= torch.finfo(torch.float64).eps:
        return None
    correlation = torch.dot(predicted, observed) / denominator
    if not torch.isfinite(correlation):
        return None
    return float(correlation)


def _row_pearsons(logits: torch.Tensor, observed: torch.Tensor) -> list[float]:
    predicted = torch.softmax(logits, dim=-1).detach().to(dtype=torch.float64)
    raw = observed.detach().to(dtype=torch.float64)
    totals = raw.sum(dim=1, keepdim=True)
    normalized = torch.zeros_like(raw)
    positive_totals = totals.squeeze(1) > 0
    normalized[positive_totals] = raw[positive_totals] / totals[positive_totals]
    values = []
    for predicted_row, observed_row in zip(predicted, normalized, strict=True):
        correlation = _centered_pearson(predicted_row, observed_row)
        if correlation is not None:
            values.append(correlation)
    return values


def _score_ordinary_pair(
    base_model: HybridMultiCellModel,
    tuned_model: HybridMultiCellModel,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
) -> tuple[dict[str, float], dict[str, float], bool]:
    base_values = {
        "acc_profile_nll": 0.0,
        "acc_count_mse": 0.0,
        "nuc_profile_nll": 0.0,
        "nuc_count_mse": 0.0,
    }
    tuned_values = dict(base_values)
    base_pearsons = {"acc": [], "nuc": []}
    tuned_pearsons = {"acc": [], "nuc": []}
    examples = 0
    accessibility_unchanged = True
    base_model.eval()
    tuned_model.eval()
    with torch.no_grad():
        for sequence, acc, acc_count, nuc, nuc_count, indices in loader:
            sequence = sequence.to(device)
            acc = acc.to(device)
            acc_count = acc_count.to(device)
            nuc = nuc.to(device)
            nuc_count = nuc_count.to(device)
            indices = indices.to(device)
            base_output = _forward_model_float32(base_model, sequence, indices)
            tuned_output = _forward_model_float32(tuned_model, sequence, indices)
            batch_size = sequence.shape[0]
            examples += batch_size
            for values, output in (
                (base_values, base_output),
                (tuned_values, tuned_output),
            ):
                values["acc_profile_nll"] += (
                    float(multinomial_nll(acc, output[0])) * batch_size
                )
                values["acc_count_mse"] += (
                    float(torch.nn.functional.mse_loss(output[1].float(), acc_count))
                    * batch_size
                )
                values["nuc_profile_nll"] += (
                    float(multinomial_nll(nuc, output[2])) * batch_size
                )
                values["nuc_count_mse"] += (
                    float(torch.nn.functional.mse_loss(output[3].float(), nuc_count))
                    * batch_size
                )
            base_pearsons["acc"].extend(_row_pearsons(base_output[0], acc))
            tuned_pearsons["acc"].extend(_row_pearsons(tuned_output[0], acc))
            base_pearsons["nuc"].extend(_row_pearsons(base_output[2], nuc))
            tuned_pearsons["nuc"].extend(_row_pearsons(tuned_output[2], nuc))
            for base_accessibility, tuned_accessibility in zip(
                base_output[:2], tuned_output[:2], strict=True
            ):
                accessibility_unchanged &= torch.allclose(
                    base_accessibility.float(),
                    tuned_accessibility.float(),
                    rtol=1e-6,
                    atol=1e-7,
                )
    if examples == 0:
        raise ValueError("ordinary adult validation subset is empty")
    for values in (base_values, tuned_values):
        for name in (
            "acc_profile_nll",
            "acc_count_mse",
            "nuc_profile_nll",
            "nuc_count_mse",
        ):
            values[name] /= examples
        values["examples"] = float(examples)
    for name, pearsons in (
        ("base", base_pearsons),
        ("tuned", tuned_pearsons),
    ):
        values = base_values if name == "base" else tuned_values
        for track in ("acc", "nuc"):
            values[f"{track}_profile_pearson"] = (
                float(np.mean(pearsons[track])) if pearsons[track] else float("nan")
            )
    return base_values, tuned_values, bool(accessibility_unchanged)


def _non_adult_rows_unchanged(
    base_model: HybridMultiCellModel,
    tuned_model: HybridMultiCellModel,
    adult_indices: list[int],
) -> bool:
    adult_mask = torch.zeros(base_model.n_cell_types, dtype=torch.bool)
    adult_mask[adult_indices] = True
    non_adult = torch.arange(base_model.n_cell_types)[~adult_mask]
    return bool(
        torch.equal(
            base_model.nuc_profile_w.detach().cpu()[non_adult],
            tuned_model.nuc_profile_w.detach().cpu()[non_adult],
        )
        and torch.equal(
            base_model.nuc_profile_b.detach().cpu()[non_adult],
            tuned_model.nuc_profile_b.detach().cpu()[non_adult],
        )
    )


def evaluate_checkpoints(
    *,
    base_checkpoint_path: Path,
    tuned_checkpoint_path: Path,
    manifest_path: Path,
    manifest_root: Path | None,
    metadata_path: Path,
    fold_json: Path,
    calls_path: Path,
    genome: Path,
    out_json: Path,
    device_name: str,
    amp: bool,
    replay_batch_size: int = 64,
    replay_examples: int = 2_000,
    num_workers: int = 0,
) -> dict[str, object]:
    """Evaluate base and tuned checkpoints once on test-only context units."""
    if out_json.exists():
        raise FileExistsError(f"refusing to overwrite existing evaluation: {out_json}")
    fold = json.loads(fold_json.read_text())
    validate_fold_boundaries(fold)
    device = torch.device(device_name)
    base_model, base_checkpoint = load_hybrid_checkpoint(base_checkpoint_path, device)
    tuned_model, tuned_checkpoint = load_hybrid_checkpoint(
        tuned_checkpoint_path, device
    )
    if base_checkpoint["cell_types"] != tuned_checkpoint["cell_types"]:
        raise ValueError("base and tuned checkpoints have different cell ordering")
    expected_hash = _sha256(base_checkpoint_path)
    if tuned_checkpoint.get("base_checkpoint_sha256") != expected_hash:
        raise ValueError(
            "tuned checkpoint was not derived from the supplied base checkpoint"
        )

    manifest = read_manifest(manifest_path, manifest_root)
    metadata = read_metadata(metadata_path)
    adult_manifest = validate_adult_cohort(manifest, metadata)
    cell_types = base_checkpoint["cell_types"]
    assert_production_cardinalities(cell_types, metadata)
    validate_adult_manifest_coverage(adult_manifest, metadata)
    checkpoint_index = map_adult_checkpoint_indices(adult_manifest, cell_types)
    calls = pl.read_parquet(calls_path)
    test_chromosomes = set(APPROVED_TEST_CHROMOSOMES)
    validate_artifact_chromosomes(
        calls,
        allowed=test_chromosomes,
        forbidden=set(fold["train"]) | {"chr8", "chr20"},
        label="test",
    )
    units = build_context_units(calls, checkpoint_index, min_members=2)
    if not units:
        raise ValueError("test artifact has no evaluable context units")
    nuc_bigwigs = {
        str(row["cell_type"]): str(row["nuc_bw"])
        for row in adult_manifest.iter_rows(named=True)
    }
    args = base_checkpoint["args"]
    cleanup = _CleanupStack()
    try:
        context_dataset = PairedContextDataset(
            units,
            genome,
            nuc_bigwigs,
            args["inputlen"],
            args["outputlen"],
        )
        cleanup.callback("evaluation context dataset", context_dataset.close)
        replay_loader = build_adult_replay_loader(
            adult_manifest,
            checkpoint_index,
            test_chromosomes,
            genome,
            inputlen=args["inputlen"],
            outputlen=args["outputlen"],
            batch_size=replay_batch_size,
            num_workers=num_workers,
            shuffle=False,
            seed=0,
            max_examples=replay_examples,
        )
        cleanup.callback(
            "evaluation replay loader",
            lambda: _close_replay_loader(replay_loader),
        )
    except Exception:
        cleanup.close()
        raise
    try:
        base_differential = _score_differential(
            base_model, context_dataset, device, amp
        )
        tuned_differential = _score_differential(
            tuned_model, context_dataset, device, amp
        )
        base_ordinary, tuned_ordinary, accessibility_unchanged = _score_ordinary_pair(
            base_model, tuned_model, replay_loader, device, amp
        )
        non_adult_unchanged = _non_adult_rows_unchanged(
            base_model,
            tuned_model,
            list(checkpoint_index.values()),
        )
        report = {
            "analysis": "differential_nucleosome_test_challenge",
            "ascertainment": (
                "These loci are an ascertained differential challenge set, "
                "not an unbiased genome-wide sample."
            ),
            "chromosomes": sorted(test_chromosomes),
            "base_checkpoint": str(base_checkpoint_path.resolve()),
            "base_checkpoint_sha256": expected_hash,
            "tuned_checkpoint": str(tuned_checkpoint_path.resolve()),
            "tuned_checkpoint_sha256": _sha256(tuned_checkpoint_path),
            "calls": str(calls_path.resolve()),
            "calls_sha256": _sha256(calls_path),
            "base": {
                "differential": base_differential,
                "ordinary": base_ordinary,
            },
            "tuned": {
                "differential": tuned_differential,
                "ordinary": tuned_ordinary,
            },
            "accessibility_unchanged": accessibility_unchanged,
            "non_adult_unchanged": non_adult_unchanged,
        }
        _write_json(out_json, report)
        return report
    finally:
        cleanup.close()


def _validation_snapshot(
    base_model: HybridMultiCellModel,
    tuned_model: HybridMultiCellModel,
    prepared: _PreparedRun,
    device: torch.device,
    amp: bool,
    base_differential: dict[str, object],
) -> tuple[dict[str, object], dict[str, object], bool]:
    tuned_differential = _score_differential(
        tuned_model, prepared.valid_dataset, device, amp
    )
    base_ordinary, tuned_ordinary, accessibility_unchanged = _score_ordinary_pair(
        base_model, tuned_model, prepared.valid_replay, device, amp
    )
    adult_indices = list(prepared.checkpoint_index.values())
    non_adult_unchanged = _non_adult_rows_unchanged(
        base_model, tuned_model, adult_indices
    )
    base_gate = {
        **base_ordinary,
        "shift_direction_accuracy": base_differential["shift_direction_accuracy"],
    }
    tuned_gate = {
        **tuned_ordinary,
        "shift_direction_accuracy": tuned_differential["shift_direction_accuracy"],
        "accessibility_unchanged": accessibility_unchanged,
        "non_adult_unchanged": non_adult_unchanged,
    }
    validation = {
        **tuned_differential,
        **tuned_ordinary,
        "accessibility_unchanged": accessibility_unchanged,
        "non_adult_unchanged": non_adult_unchanged,
    }
    comparison = {
        "base": {"differential": base_differential, "ordinary": base_ordinary},
        "tuned": {
            "differential": tuned_differential,
            "ordinary": tuned_ordinary,
        },
    }
    return validation, comparison, passes_gates(base_gate, tuned_gate)


def _save_finetuned_checkpoint(
    path: Path,
    model: HybridMultiCellModel,
    prepared: _PreparedRun,
    config: FineTuneConfig,
    step: int,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": prepared.checkpoint["args"],
            "cell_types": prepared.checkpoint["cell_types"],
            "base_checkpoint": str(config.checkpoint.resolve()),
            "base_checkpoint_sha256": _sha256(config.checkpoint),
            "lambda_dev": config.lambda_dev,
            "selected_step": step,
        },
        path,
    )


def _close_replay_loader(loader: DataLoader) -> None:
    dataset = loader.dataset
    if isinstance(dataset, Subset):
        dataset = dataset.dataset
    if not isinstance(dataset, _DatasetCollection):
        return
    for wrapper in dataset.datasets:
        source = wrapper.dataset
        for attribute in ("_genome", "_acc_bw", "_nuc_bw"):
            handle = getattr(source, attribute, None)
            if handle is not None:
                try:
                    handle.close()
                except Exception as error:  # noqa: BLE001 - close remaining handles
                    warnings.warn(
                        f"cleanup failed for replay handle {attribute}: {error}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                finally:
                    setattr(source, attribute, None)


def _run_single_finetune_owned(
    config: FineTuneConfig, cleanup: _CleanupStack
) -> RunResult:
    if config.steps < 1:
        raise ValueError("fine-tuning steps must be positive")
    if config.validate_every < 1:
        raise ValueError("validate_every must be positive")
    config.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = config.out_dir / f"{config.output_name}.pt"
    report_path = config.out_dir / (
        "final_validation.json"
        if config.output_name == "final"
        else f"{config.output_name}.json"
    )
    for output in (checkpoint_path, report_path):
        if output.exists():
            raise FileExistsError(
                f"refusing to overwrite existing run output: {output}"
            )

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device(config.device)
    prepared = _prepare_run(config, device, cleanup)
    base_model, base_checkpoint = load_hybrid_checkpoint(config.checkpoint, device)
    if base_checkpoint["cell_types"] != prepared.checkpoint["cell_types"]:
        raise AssertionError(
            "base checkpoint cell ordering changed while preparing run"
        )
    base_ordinary, _, _ = _score_ordinary_pair(
        base_model, base_model, prepared.valid_replay, device, config.amp
    )
    if not math.isfinite(_metric(base_ordinary, "nuc_profile_pearson")):
        raise ValueError(
            "base ordinary validation requires a finite nucleosome profile Pearson "
            "before optimization"
        )

    adult_indices = list(prepared.checkpoint_index.values())
    guard = configure_adult_profile_finetune(prepared.model, adult_indices)
    cleanup.callback("frozen-row guard", guard.teardown)
    optimizer = guard.make_optimizer(config.learning_rate)
    sampler = ContextBalancedSampler(
        prepared.train_dataset,
        num_samples=config.steps,
        seed=config.seed,
    )
    replay_iterator = iter(prepared.train_replay)
    base_differential = _score_differential(
        base_model, prepared.valid_dataset, device, config.amp
    )
    provenance = {
        "base_checkpoint": str(config.checkpoint.resolve()),
        "base_checkpoint_sha256": _sha256(config.checkpoint),
        "train_calls": str(config.train_calls.resolve()),
        "train_calls_sha256": _sha256(config.train_calls),
        "valid_calls": str(config.valid_calls.resolve()),
        "valid_calls_sha256": _sha256(config.valid_calls),
        "manifest": str(config.manifest.resolve()),
        "metadata": str(config.metadata.resolve()),
        "fold": {key: sorted(value) for key, value in prepared.fold.items()},
        "adult_cell_types": prepared.adult_manifest["cell_type"].to_list(),
        "optimizer": {
            "type": "AdamW",
            "learning_rate": config.learning_rate,
            "min_learning_rate": config.min_learning_rate,
            "warmup_steps": config.warmup_steps,
            "weight_decay": 0.0,
            "gradient_clip": config.grad_clip,
        },
        "lambda_dev": config.lambda_dev,
        "alpha": config.alpha,
        "steps": config.steps,
        "seed": config.seed,
    }
    logger = start_wandb_run(
        enabled=config.wandb,
        project=config.wandb_project,
        name=config.wandb_run_name,
        config=provenance,
    )
    cleanup.callback("W&B run", logger.finish)
    validation_history = []
    selected_validation: dict[str, object] = {}
    selected_comparison: dict[str, object] = {}
    last_validation: dict[str, object] = {}
    last_comparison: dict[str, object] = {}
    selected_step = None
    last_step_result = None
    best_rank = (float("-inf"), float("-inf"))
    try:
        for step, paired_index in enumerate(sampler, start=1):
            try:
                replay = next(replay_iterator)
            except StopIteration:
                replay_iterator = iter(prepared.train_replay)
                replay = next(replay_iterator)
            learning_rate = learning_rate_for_step(
                step,
                config.steps,
                config.warmup_steps,
                config.learning_rate,
                config.min_learning_rate,
            )
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            last_step_result = train_one_step(
                prepared.model,
                guard,
                optimizer,
                prepared.train_dataset[paired_index],
                replay,
                device=device,
                alpha=config.alpha,
                lambda_dev=config.lambda_dev,
                grad_clip=config.grad_clip,
                amp=config.amp,
            )
            logger.log(
                {
                    "train/total_loss": last_step_result.total_loss,
                    "train/replay_nll": last_step_result.replay_nll,
                    "train/differential_profile_nll": (
                        last_step_result.differential_profile_nll
                    ),
                    "train/deviation_loss": last_step_result.deviation_loss,
                    "train/gradient_norm": last_step_result.gradient_norm,
                    "train/learning_rate": learning_rate,
                },
                step=step,
            )
            if step % config.validate_every != 0 and step != config.steps:
                continue
            validation, comparison, gate_status = _validation_snapshot(
                base_model,
                prepared.model,
                prepared,
                device,
                config.amp,
                base_differential,
            )
            last_validation = validation
            last_comparison = comparison
            validation_history.append(
                {"step": step, "passes_gates": gate_status, **validation}
            )
            logger.log(
                {
                    **{
                        f"validation/{key}": value
                        for key, value in validation.items()
                        if isinstance(value, (int, float))
                    },
                    "validation/passes_gates": gate_status,
                },
                step=step,
            )
            rank = (
                _metric(validation, "deviation_pearson"),
                _metric(validation, "shift_direction_accuracy"),
            )
            rank = tuple(
                value if math.isfinite(value) else float("-inf") for value in rank
            )
            if gate_status and rank > best_rank:
                best_rank = rank
                selected_step = step
                selected_validation = validation
                selected_comparison = comparison
                _save_finetuned_checkpoint(
                    checkpoint_path, prepared.model, prepared, config, step
                )
        report = {
            "status": "passing"
            if selected_step is not None
            else "no_passing_checkpoint",
            **provenance,
            "selected_step": selected_step,
            "passes_gates": selected_step is not None,
            "validation": selected_validation or last_validation,
            "comparison": selected_comparison or last_comparison,
            "validation_history": validation_history,
            "last_train_step": (
                vars(last_step_result) if last_step_result is not None else None
            ),
        }
        _write_json(report_path, report)
        selection = {
            "selected_step": selected_step,
            "lambda_dev": config.lambda_dev,
            "passes_gates": selected_step is not None,
            "selected_checkpoint": (
                str(checkpoint_path.resolve()) if selected_step is not None else None
            ),
        }
        logger.log({f"selection/{key}": value for key, value in selection.items()})
        logger.update_config(selection)
        return RunResult(
            lambda_dev=config.lambda_dev,
            selected_step=selected_step,
            checkpoint_path=checkpoint_path if selected_step is not None else None,
            report_path=report_path,
            validation=selected_validation or last_validation,
            passes_gates=selected_step is not None,
        )
    finally:
        pass


def run_single_finetune(config: FineTuneConfig) -> RunResult:
    """Run one lambda candidate from the original base checkpoint."""
    cleanup = _CleanupStack()
    try:
        return _run_single_finetune_owned(config, cleanup)
    finally:
        cleanup.close()


def run_pilot_sweep(config: FineTuneConfig, lambda_grid) -> SweepResult:
    """Run every pilot from the same base checkpoint and select passing metrics."""
    values = tuple(sorted(float(value) for value in lambda_grid))
    if values != APPROVED_PILOT_LAMBDAS:
        raise ValueError(
            f"lambda_grid must contain exactly {list(APPROVED_PILOT_LAMBDAS)}"
        )
    config.out_dir.mkdir(parents=True, exist_ok=True)
    selection_path = config.out_dir / "selected_pilot.json"
    if selection_path.exists():
        raise FileExistsError(
            f"refusing to overwrite existing run output: {selection_path}"
        )
    runs = []
    for lambda_dev in values:
        candidate = replace(
            config,
            lambda_dev=lambda_dev,
            steps=config.pilot_steps,
            output_name=f"pilot_lambda_{lambda_dev:g}",
        )
        runs.append(run_single_finetune(candidate))
    passing = [run for run in runs if run.passes_gates]
    selected = max(
        passing,
        key=lambda run: (
            _metric(run.validation, "deviation_pearson"),
            _metric(run.validation, "shift_direction_accuracy"),
        ),
        default=None,
    )
    payload = {
        "passes_gates": selected is not None,
        "lambda_dev": selected.lambda_dev if selected is not None else None,
        "selected_step": selected.selected_step if selected is not None else None,
        "checkpoint": (
            selected.checkpoint_path if selected is not None else config.checkpoint
        ),
        "base_checkpoint": config.checkpoint.resolve(),
        "base_checkpoint_sha256": _sha256(config.checkpoint),
        "candidates": [
            {
                "lambda_dev": run.lambda_dev,
                "passes_gates": run.passes_gates,
                "selected_step": run.selected_step,
                "checkpoint": run.checkpoint_path,
                "report": run.report_path,
                "validation": run.validation,
            }
            for run in runs
        ],
    }
    _write_json(selection_path, payload)
    return SweepResult(
        selected_lambda=selected.lambda_dev if selected is not None else None,
        runs=tuple(runs),
        selection_path=selection_path,
    )


class _CheckpointIndexedDataset(Dataset):
    def __init__(self, dataset: MultiTaskRegionDataset, checkpoint_index: int) -> None:
        self.dataset = dataset
        self.checkpoint_index = checkpoint_index

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        return (
            *self.dataset[index],
            torch.tensor(self.checkpoint_index, dtype=torch.long),
        )


class _DatasetCollection(Dataset):
    def __init__(self, datasets: list[_CheckpointIndexedDataset]) -> None:
        self.datasets = datasets
        lengths = [len(dataset) for dataset in datasets]
        self.cumulative_lengths = np.cumsum(lengths)

    def __len__(self) -> int:
        if not self.datasets:
            return 0
        return int(self.cumulative_lengths[-1])

    def __getitem__(self, index: int):
        dataset_index = int(
            np.searchsorted(self.cumulative_lengths, index, side="right")
        )
        previous = (
            int(self.cumulative_lengths[dataset_index - 1]) if dataset_index > 0 else 0
        )
        return self.datasets[dataset_index][index - previous]


def build_adult_replay_loader(
    adult_manifest: pl.DataFrame,
    checkpoint_index: Mapping[str, int],
    chromosomes: set[str],
    genome_fasta: str | Path,
    *,
    inputlen: int,
    outputlen: int,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    seed: int,
    max_examples: int | None = None,
) -> DataLoader:
    """Load ordinary adult examples tagged with their fixed checkpoint row."""
    required = {"cell_type", "peaks", "nonpeaks", "acc_bw", "nuc_bw"}
    missing = sorted(required.difference(adult_manifest.columns))
    if missing:
        raise ValueError(
            f"adult manifest missing required columns: {', '.join(missing)}"
        )
    datasets = []
    for row_number, row in enumerate(adult_manifest.iter_rows(named=True)):
        cell_type = str(row["cell_type"])
        if cell_type not in checkpoint_index:
            raise ValueError(
                f"adult manifest cell type absent from checkpoint: {cell_type}"
            )
        dataset = MultiTaskRegionDataset(
            load_regions(row["peaks"], chromosomes),
            load_regions(row["nonpeaks"], chromosomes),
            str(genome_fasta),
            row["acc_bw"],
            row["nuc_bw"],
            inputlen,
            outputlen,
            0,
            negative_sampling_ratio=1.0,
            add_revcomp=False,
            shuffle=False,
            seed=seed + row_number,
            nuc_smooth_sigma=0.0,
        )
        datasets.append(_CheckpointIndexedDataset(dataset, checkpoint_index[cell_type]))
    collection = _DatasetCollection(datasets)
    if len(collection) == 0:
        raise ValueError("adult replay dataset is empty on the requested chromosomes")
    generator = torch.Generator().manual_seed(seed)
    selected: Dataset = collection
    if max_examples is not None and max_examples < len(collection):
        if max_examples < 1:
            raise ValueError("max_examples must be positive")
        indices = torch.randperm(len(collection), generator=generator)[:max_examples]
        selected = Subset(collection, sorted(indices.tolist()))
    return DataLoader(
        selected,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def train_one_step(
    model: HybridMultiCellModel,
    guard: AdultProfileHeadGuard,
    optimizer: torch.optim.Optimizer,
    paired: PairedContextExample,
    replay,
    *,
    device: torch.device,
    alpha: float,
    lambda_dev: float,
    grad_clip: float,
    amp: bool = False,
) -> StepResult:
    """Apply one paired-context update and one ordinary adult replay update."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    indices = paired.checkpoint_indices.to(device)
    sequence = paired.sequence.to(device)
    repeated_sequence = sequence.unsqueeze(0).expand(indices.numel(), -1, -1)
    paired_logits = _forward_model_float32(model, repeated_sequence, indices)[2]
    differential = differential_loss(
        paired_logits,
        paired.raw_profiles.to(device),
        paired.deviation_slice,
        alpha=alpha,
        lambda_dev=lambda_dev,
    )

    replay_sequence, _, _, replay_nuc, _, replay_indices = replay
    replay_logits = _forward_model_float32(
        model, replay_sequence.to(device), replay_indices.to(device)
    )[2]
    replay_nll = multinomial_nll(replay_nuc.to(device), replay_logits)
    total = replay_nll + differential.total
    if not torch.isfinite(total):
        raise FloatingPointError("non-finite fine-tuning loss")
    total.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        (guard.profile_w, guard.profile_b), grad_clip
    )
    if not torch.isfinite(gradient_norm):
        raise FloatingPointError("non-finite fine-tuning gradient norm")
    optimizer.step()
    guard.assert_unchanged()
    return StepResult(
        total_loss=float(total.detach()),
        replay_nll=float(replay_nll.detach()),
        differential_profile_nll=float(differential.profile_nll.detach()),
        deviation_loss=float(differential.deviation.detach()),
        gradient_norm=float(gradient_norm.detach()),
    )


def _metric(metrics: Mapping[str, object], name: str) -> float:
    value = metrics.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return float("nan")
    return float(value)


def passes_gates(base: Mapping[str, object], tuned: Mapping[str, object]) -> bool:
    """Apply all approved preservation thresholds to a candidate checkpoint."""
    base_pearson = _metric(base, "nuc_profile_pearson")
    tuned_pearson = _metric(tuned, "nuc_profile_pearson")
    base_nll = _metric(base, "nuc_profile_nll")
    tuned_nll = _metric(tuned, "nuc_profile_nll")
    base_shift = _metric(base, "shift_direction_accuracy")
    tuned_shift = _metric(tuned, "shift_direction_accuracy")
    if not all(
        math.isfinite(value)
        for value in (
            base_pearson,
            tuned_pearson,
            base_nll,
            tuned_nll,
            base_shift,
            tuned_shift,
        )
    ):
        return False
    if tuned_pearson < base_pearson - 0.01:
        return False
    if tuned_nll > base_nll * 1.02:
        return False
    if tuned_shift < base_shift:
        return False
    return (
        tuned.get("accessibility_unchanged") is True
        and tuned.get("non_adult_unchanged") is True
    )


def load_selected_lambda(
    selected_pilot_json: Path,
    *,
    expected_checkpoint: Path | None = None,
    expected_checkpoint_sha256: str | None = None,
) -> float:
    """Read the exact lambda from a passing pilot-selection artifact."""
    if expected_checkpoint is None and expected_checkpoint_sha256 is None:
        raise ValueError("expected checkpoint path or SHA-256 is required")
    selection = json.loads(selected_pilot_json.read_text())
    if selection.get("passes_gates") is not True:
        raise ValueError("selected pilot JSON does not identify a passing candidate")
    value = selection.get("lambda_dev")
    if (
        not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) not in APPROVED_PILOT_LAMBDAS
    ):
        raise ValueError("selected pilot JSON has no approved lambda_dev")
    stored_path = selection.get("base_checkpoint")
    stored_hash = selection.get("base_checkpoint_sha256")
    if stored_path is None or stored_hash is None:
        raise ValueError("selected pilot JSON is missing base checkpoint provenance")
    if not isinstance(stored_path, str) or not isinstance(stored_hash, str):
        raise TypeError("selected pilot base checkpoint provenance must be strings")
    if expected_checkpoint is not None:
        resolved = expected_checkpoint.resolve()
        if Path(stored_path).resolve() != resolved:
            raise ValueError("selected pilot base checkpoint path does not match")
        expected_checkpoint_sha256 = _sha256(resolved)
    if stored_hash != expected_checkpoint_sha256:
        raise ValueError("selected pilot base checkpoint SHA-256 does not match")
    return float(value)


def load_hybrid_checkpoint(
    path: str | Path, device: torch.device
) -> tuple[HybridMultiCellModel, dict]:
    """Load a hybrid checkpoint without changing its cell-type row ordering."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    args = checkpoint.get("args")
    if not isinstance(args, dict) or not args.get("hybrid"):
        raise ValueError(
            "differential fine-tuning requires a HybridMultiCellModel checkpoint"
        )
    cell_types = checkpoint.get("cell_types")
    if not isinstance(cell_types, list) or not cell_types:
        raise ValueError("checkpoint must contain a non-empty cell_types list")
    if not all(isinstance(cell_type, str) and cell_type for cell_type in cell_types):
        raise ValueError("checkpoint cell_types must contain non-empty names")
    if len(set(cell_types)) != len(cell_types):
        raise ValueError("checkpoint contains duplicate cell_type entries")

    model = HybridMultiCellModel(
        len(cell_types),
        args["inputlen"],
        args["outputlen"],
        args["filters"],
        args["n_dil_layers"],
        args.get("embed_dim", 32),
        cond_mode=args.get("cond_mode", "film"),
        profile_kernel_size=args.get("profile_kernel_size", 75),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device), checkpoint


def _add_shared_data_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--manifest-root", type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--fold-json", required=True, type=Path)
    parser.add_argument("--genome", required=True, type=Path)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Accepted, but profile-head forwards stay float32 until head-safe AMP exists.",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for training and one-time evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="fine-tune adult profile rows")
    train.add_argument("--checkpoint", required=True, type=Path)
    _add_shared_data_arguments(train)
    train.add_argument("--train-calls", required=True, type=Path)
    train.add_argument("--valid-calls", required=True, type=Path)
    train.add_argument("--out-dir", required=True, type=Path)
    mode = train.add_mutually_exclusive_group(required=True)
    mode.add_argument("--lambda-dev-grid", nargs="+", type=float)
    mode.add_argument("--selected-pilot-json", type=Path)
    train.add_argument("--learning-rate", type=float, default=1e-5)
    train.add_argument("--min-learning-rate", type=float, default=1e-6)
    train.add_argument("--alpha", type=float, default=1.0)
    train.add_argument("--pilot-steps", type=int, default=2_000)
    train.add_argument("--full-steps", type=int, default=10_000)
    train.add_argument("--warmup-steps", type=int, default=100)
    train.add_argument("--grad-clip", type=float, default=1.0)
    train.add_argument("--validate-every", type=int, default=250)
    train.add_argument("--replay-batch-size", type=int, default=64)
    train.add_argument("--validation-replay-examples", type=int, default=2_000)
    train.add_argument("--num-workers", type=int, default=0)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--wandb", action="store_true")
    train.add_argument("--wandb-project", default="jg_experiments")
    train.add_argument("--wandb-run-name")

    evaluate = subparsers.add_parser("evaluate", help="compare base and tuned models")
    evaluate.add_argument("--base-checkpoint", required=True, type=Path)
    evaluate.add_argument("--tuned-checkpoint", required=True, type=Path)
    _add_shared_data_arguments(evaluate)
    evaluate.add_argument("--calls", required=True, type=Path)
    evaluate.add_argument("--split", choices=("test",), default="test")
    evaluate.add_argument("--out-json", required=True, type=Path)
    evaluate.add_argument("--replay-batch-size", type=int, default=64)
    evaluate.add_argument("--replay-examples", type=int, default=2_000)
    evaluate.add_argument("--num-workers", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the requested fine-tuning or evaluation command."""
    args = build_parser().parse_args(argv)
    if args.command == "evaluate":
        evaluate_checkpoints(
            base_checkpoint_path=args.base_checkpoint,
            tuned_checkpoint_path=args.tuned_checkpoint,
            manifest_path=args.manifest,
            manifest_root=args.manifest_root,
            metadata_path=args.metadata,
            fold_json=args.fold_json,
            calls_path=args.calls,
            genome=args.genome,
            out_json=args.out_json,
            device_name=args.device,
            amp=args.amp,
            replay_batch_size=args.replay_batch_size,
            replay_examples=args.replay_examples,
            num_workers=args.num_workers,
        )
        print(f"wrote evaluation: {args.out_json}")
        return 0

    config = FineTuneConfig(
        checkpoint=args.checkpoint,
        manifest=args.manifest,
        manifest_root=args.manifest_root,
        metadata=args.metadata,
        fold_json=args.fold_json,
        train_calls=args.train_calls,
        valid_calls=args.valid_calls,
        genome=args.genome,
        out_dir=args.out_dir,
        device=args.device,
        amp=args.amp,
        alpha=args.alpha,
        pilot_steps=args.pilot_steps,
        learning_rate=args.learning_rate,
        min_learning_rate=args.min_learning_rate,
        warmup_steps=args.warmup_steps,
        grad_clip=args.grad_clip,
        validate_every=args.validate_every,
        replay_batch_size=args.replay_batch_size,
        validation_replay_examples=args.validation_replay_examples,
        num_workers=args.num_workers,
        seed=args.seed,
        wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )
    if args.lambda_dev_grid is not None:
        sweep = run_pilot_sweep(config, args.lambda_dev_grid)
        print(f"wrote pilot selection: {sweep.selection_path}")
        return 0 if sweep.selected_lambda is not None else 2

    selected_lambda = load_selected_lambda(
        args.selected_pilot_json, expected_checkpoint=args.checkpoint
    )
    result = run_single_finetune(
        replace(
            config,
            lambda_dev=selected_lambda,
            steps=args.full_steps,
            output_name="final",
        )
    )
    print(f"wrote validation report: {result.report_path}")
    return 0 if result.passes_gates else 2


if __name__ == "__main__":
    raise SystemExit(main())
