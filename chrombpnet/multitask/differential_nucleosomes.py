"""Adult, lineage-aware differential nucleosome-positioning analysis.

The statistical core operates on in-memory dyad count profiles so it can be
validated independently of the corpus storage layer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy.ndimage import gaussian_filter1d


@dataclass(frozen=True)
class CallConfig:
    """Thresholds and profile-processing parameters for one differential call."""

    sigma: float = 20.0
    max_shift: int = 100
    min_counts: float = 20.0
    min_effect: float = 0.05
    min_robust_z: float = 3.0
    noise_multiplier: float = 1.0
    min_lineage_comparators: int = 3
    seed: int = 0


def _require_columns(frame: pl.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{name} missing required columns: {', '.join(missing)}")


def _coerce_boolean(frame: pl.DataFrame, column: str) -> pl.DataFrame:
    dtype = frame.schema[column]
    if dtype == pl.Boolean:
        return frame
    truthy = {"1", "true", "t", "yes", "y"}
    falsy = {"0", "false", "f", "no", "n"}
    normalized = pl.col(column).cast(pl.String).str.to_lowercase().str.strip_chars()
    invalid = frame.filter(~normalized.is_in(sorted(truthy | falsy)))
    if invalid.height:
        values = invalid[column].unique().cast(pl.String).to_list()
        raise ValueError(f"metadata column {column} has invalid booleans: {values}")
    return frame.with_columns(normalized.is_in(sorted(truthy)).alias(column))


def validate_adult_cohort(
    manifest: pl.DataFrame, metadata: pl.DataFrame
) -> pl.DataFrame:
    """Join manifest paths to explicit cohort metadata and retain adult tissues.

    Every manifest row must have metadata, including rows later filtered out. This
    makes accidental cohort drift visible instead of silently dropping unknown data.
    """

    manifest_required = {"cell_type", "peaks", "nonpeaks", "acc_bw", "nuc_bw"}
    metadata_required = {"cell_type", "lineage", "is_adult"}
    _require_columns(manifest, manifest_required, "manifest")
    _require_columns(metadata, metadata_required, "metadata")

    if manifest["cell_type"].n_unique() != manifest.height:
        raise ValueError("manifest contains duplicate cell_type rows")
    if metadata["cell_type"].n_unique() != metadata.height:
        raise ValueError("metadata contains duplicate cell_type rows")

    metadata = _coerce_boolean(metadata, "is_adult")
    if "is_cell_line" not in metadata.columns:
        metadata = metadata.with_columns(pl.lit(False).alias("is_cell_line"))
    else:
        metadata = _coerce_boolean(metadata, "is_cell_line")

    joined = (
        manifest.with_row_index("_manifest_order")
        .join(
            metadata.select("cell_type", "lineage", "is_adult", "is_cell_line"),
            on="cell_type",
            how="left",
        )
        .sort("_manifest_order")
    )
    missing = joined.filter(pl.col("lineage").is_null() | pl.col("is_adult").is_null())
    if missing.height:
        names = ", ".join(missing["cell_type"].to_list())
        raise ValueError(f"manifest rows missing metadata: {names}")

    cohort = joined.filter(pl.col("is_adult") & ~pl.col("is_cell_line"))
    empty_lineage = cohort.filter(
        pl.col("lineage").cast(pl.String).str.strip_chars().fill_null("") == ""
    )
    if empty_lineage.height:
        names = ", ".join(empty_lineage["cell_type"].to_list())
        raise ValueError(f"adult rows missing lineage labels: {names}")
    if cohort.is_empty():
        raise ValueError(
            "adult cohort is empty after filtering fetal and cell-line rows"
        )
    return cohort.drop("_manifest_order")


def normalize_profiles(raw: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """Return sum-preserving smoothed counts and L1-normalized profile shapes."""

    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim != 2:
        raise ValueError(
            f"profiles must have shape [cell_type, position], got {raw.shape}"
        )
    if np.any(raw < 0):
        raise ValueError("dyad count profiles cannot contain negative values")
    if sigma < 0:
        raise ValueError("sigma must be non-negative")

    smoothed = (
        gaussian_filter1d(raw, sigma=sigma, axis=1, mode="constant")
        if sigma
        else raw.copy()
    ).astype(np.float32, copy=False)
    raw_sums = raw.sum(axis=1, keepdims=True, dtype=np.float64)
    smooth_sums = smoothed.sum(axis=1, keepdims=True, dtype=np.float64)
    scale = np.divide(
        raw_sums,
        smooth_sums,
        out=np.ones_like(raw_sums),
        where=smooth_sums > 0,
    )
    smoothed = (smoothed * scale).astype(np.float32)
    totals = smoothed.sum(axis=1, keepdims=True, dtype=np.float64)
    normalized = np.divide(
        smoothed,
        totals,
        out=np.zeros_like(smoothed),
        where=totals > 0,
    ).astype(np.float32)
    return smoothed, normalized


def estimate_shift(target: np.ndarray, consensus: np.ndarray, max_shift: int) -> int:
    """Estimate target displacement relative to consensus in base pairs.

    Positive values mean the target profile is shifted right. Correlations are
    evaluated only on the overlapping portion for each candidate displacement.
    """

    target = np.asarray(target, dtype=np.float64)
    consensus = np.asarray(consensus, dtype=np.float64)
    if target.shape != consensus.shape or target.ndim != 1:
        raise ValueError(
            "target and consensus must be one-dimensional equal-length profiles"
        )
    if max_shift < 0:
        raise ValueError("max_shift must be non-negative")
    max_shift = min(max_shift, target.size - 2)

    scores: list[float] = []
    shifts = range(-max_shift, max_shift + 1)
    for shift in shifts:
        if shift < 0:
            left, right = target[:shift], consensus[-shift:]
        elif shift > 0:
            left, right = target[shift:], consensus[:-shift]
        else:
            left, right = target, consensus
        left = left - left.mean()
        right = right - right.mean()
        denom = np.linalg.norm(left) * np.linalg.norm(right)
        scores.append(float(np.dot(left, right) / denom) if denom else -np.inf)
    return int(list(shifts)[int(np.argmax(scores))])


def _total_variation(left: np.ndarray, right: np.ndarray) -> float:
    return float(0.5 * np.abs(left - right).sum())


def _robust_z(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(values)
    result = np.full(values.shape, np.nan, dtype=np.float64)
    if not valid.any():
        return result
    median = np.median(values[valid])
    mad = np.median(np.abs(values[valid] - median))
    scale = max(1.4826 * mad, 1e-6)
    result[valid] = (values[valid] - median) / scale
    return result


def _split_half_noise(raw: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    counts = np.rint(np.clip(raw, 0, None)).astype(np.int64)
    noise = np.zeros(counts.shape[0], dtype=np.float64)
    for index, profile in enumerate(counts):
        rng = np.random.default_rng(seed + index * 1_000_003)
        first = rng.binomial(profile, 0.5)
        second = profile - first
        _, normalized = normalize_profiles(np.stack([first, second]), sigma)
        noise[index] = _total_variation(normalized[0], normalized[1])
    return noise


def score_profiles(
    raw: np.ndarray,
    cell_types: list[str],
    lineages: list[str],
    config: CallConfig,
    *,
    chrom: str = "",
    center: int = 0,
) -> tuple[pl.DataFrame, dict[str, np.ndarray]]:
    """Score one genomic locus across cell types.

    The returned table has one row per target cell type. Cached arrays retain the
    full cross-cell profile matrix for visualization of selected loci.
    """

    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim != 2:
        raise ValueError("raw must have shape [cell_type, position]")
    n_cells, width = raw.shape
    if len(cell_types) != n_cells or len(lineages) != n_cells:
        raise ValueError("cell_types and lineages must align with raw profile rows")
    if len(set(cell_types)) != len(cell_types):
        raise ValueError("cell_types must be unique")
    if n_cells < 2:
        raise ValueError("at least two cell types are required")

    smoothed, normalized = normalize_profiles(raw, config.sigma)
    depth = raw.sum(axis=1, dtype=np.float64)
    passes_depth = depth >= config.min_counts
    split_noise = _split_half_noise(raw, config.sigma, config.seed)
    global_consensus = np.full_like(normalized, np.nan)
    lineage_consensus = np.full_like(normalized, np.nan)
    global_effect = np.full(n_cells, np.nan, dtype=np.float64)
    lineage_effect = np.full(n_cells, np.nan, dtype=np.float64)
    global_shift = np.full(n_cells, np.nan, dtype=np.float64)
    lineage_shift = np.full(n_cells, np.nan, dtype=np.float64)
    global_comparators = np.zeros(n_cells, dtype=np.int64)
    lineage_comparators = np.zeros(n_cells, dtype=np.int64)

    lineage_array = np.asarray(lineages, dtype=object)
    indices = np.arange(n_cells)
    for index in range(n_cells):
        global_mask = (indices != index) & passes_depth
        global_comparators[index] = int(global_mask.sum())
        if passes_depth[index] and global_comparators[index]:
            global_consensus[index] = normalized[global_mask].mean(axis=0)
            global_effect[index] = _total_variation(
                normalized[index], global_consensus[index]
            )
            global_shift[index] = estimate_shift(
                normalized[index], global_consensus[index], config.max_shift
            )

        lineage_mask = (
            (lineage_array == lineages[index]) & (indices != index) & passes_depth
        )
        comparator_count = int(lineage_mask.sum())
        lineage_comparators[index] = comparator_count
        if passes_depth[index] and comparator_count >= config.min_lineage_comparators:
            lineage_consensus[index] = normalized[lineage_mask].mean(axis=0)
            lineage_effect[index] = _total_variation(
                normalized[index], lineage_consensus[index]
            )
            lineage_shift[index] = estimate_shift(
                normalized[index], lineage_consensus[index], config.max_shift
            )

    global_z = _robust_z(global_effect)
    lineage_z = np.full(n_cells, np.nan, dtype=np.float64)
    for lineage in dict.fromkeys(lineages):
        indices = np.flatnonzero(lineage_array == lineage)
        lineage_z[indices] = _robust_z(lineage_effect[indices])

    passes_global_noise = global_effect >= config.noise_multiplier * split_noise
    passes_lineage_noise = lineage_effect >= config.noise_multiplier * split_noise
    is_global = (
        passes_depth
        & (global_effect >= config.min_effect)
        & (global_z >= config.min_robust_z)
        & passes_global_noise
    )
    is_lineage = (
        passes_depth
        & np.isfinite(lineage_effect)
        & (lineage_effect >= config.min_effect)
        & (lineage_z >= config.min_robust_z)
        & passes_lineage_noise
    )

    half = width // 2
    rows = {
        "chrom": [chrom] * n_cells,
        "start": [int(center - half)] * n_cells,
        "end": [int(center + half + (width % 2))] * n_cells,
        "center": [int(center)] * n_cells,
        "cell_type": cell_types,
        "lineage": lineages,
        "depth": depth,
        "passes_depth": passes_depth,
        "split_half_noise": split_noise,
        "global_comparators": global_comparators,
        "lineage_comparators": lineage_comparators,
        "lineage_effect": lineage_effect,
        "lineage_robust_z": lineage_z,
        "lineage_shift_bp": [
            int(value) if np.isfinite(value) else None for value in lineage_shift
        ],
        "passes_lineage_noise": passes_lineage_noise,
        "global_effect": global_effect,
        "global_robust_z": global_z,
        "global_shift_bp": [
            int(value) if np.isfinite(value) else None for value in global_shift
        ],
        "passes_global_noise": passes_global_noise,
        "is_differential": is_lineage,
        "is_global_differential": is_global,
    }
    profiles = {
        "raw": raw,
        "smoothed": smoothed,
        "normalized": normalized,
        "lineage_consensus": lineage_consensus,
        "global_consensus": global_consensus,
    }
    return pl.DataFrame(rows), profiles
