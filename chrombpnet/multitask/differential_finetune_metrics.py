"""Differential nucleosome metrics for one locus-context prediction group."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from chrombpnet.multitask.differential_nucleosomes import (
    _robust_z,
    _total_variation,
    estimate_shift,
    normalize_profiles,
)


@dataclass(frozen=True)
class DifferentialEndpointInputs:
    """Per-row values needed to score every differential endpoint."""

    deviation_pearsons: tuple[float, ...]
    deviation_eligible: tuple[bool, ...]
    predicted_effects: tuple[float, ...]
    observed_effects: tuple[float, ...]
    predicted_shifts: tuple[float, ...]
    observed_shifts: tuple[float, ...]
    predicted_robust_z: tuple[float, ...]
    observed_calls: tuple[bool, ...]

    def __post_init__(self) -> None:
        lengths = {
            len(self.deviation_pearsons),
            len(self.deviation_eligible),
            len(self.predicted_effects),
            len(self.observed_effects),
            len(self.predicted_shifts),
            len(self.observed_shifts),
            len(self.predicted_robust_z),
            len(self.observed_calls),
        }
        if len(lengths) != 1:
            raise ValueError("differential endpoint inputs must align by target row")

    @property
    def expected_rows(self) -> int:
        return len(self.observed_calls)


@dataclass(frozen=True)
class DifferentialMetrics:
    """Agreement and row coverage for differential profile endpoints."""

    deviation_pearson: float
    effect_spearman: float
    shift_direction_accuracy: float
    shift_mae_bp: float
    average_precision: float
    target_locus_rows: int
    counts: dict[str, dict[str, int]] = field(default_factory=dict)
    coverage_complete: bool = False
    endpoint_inputs: DifferentialEndpointInputs | None = field(
        default=None, repr=False, compare=False
    )

    @classmethod
    def from_endpoint_inputs(
        cls, inputs: DifferentialEndpointInputs
    ) -> DifferentialMetrics:
        expected = inputs.expected_rows
        deviation = np.asarray(inputs.deviation_pearsons, dtype=np.float64)
        deviation_eligible = np.asarray(inputs.deviation_eligible, dtype=bool)
        deviation_scored = deviation_eligible & np.isfinite(deviation)

        predicted_effect = np.asarray(inputs.predicted_effects, dtype=np.float64)
        observed_effect = np.asarray(inputs.observed_effects, dtype=np.float64)
        effect_eligible = np.isfinite(observed_effect)
        effect_scored = effect_eligible & np.isfinite(predicted_effect)

        predicted_shift = np.asarray(inputs.predicted_shifts, dtype=np.float64)
        observed_shift = np.asarray(inputs.observed_shifts, dtype=np.float64)
        shift_eligible = np.isfinite(observed_shift) & (np.abs(observed_shift) >= 10)
        shift_scored = shift_eligible & np.isfinite(predicted_shift)

        predicted_robust_z = np.asarray(inputs.predicted_robust_z, dtype=np.float64)
        observed_calls = np.asarray(inputs.observed_calls, dtype=bool)
        ranking_eligible = np.ones(expected, dtype=bool)
        ranking_scored = np.isfinite(predicted_robust_z)

        if shift_scored.any():
            shift_direction_accuracy = float(
                (
                    np.sign(predicted_shift[shift_scored])
                    == np.sign(observed_shift[shift_scored])
                ).mean()
            )
            shift_mae_bp = float(
                np.abs(
                    predicted_shift[shift_scored] - observed_shift[shift_scored]
                ).mean()
            )
        else:
            shift_direction_accuracy = float("nan")
            shift_mae_bp = float("nan")

        counts = {
            "deviation_pearson": _endpoint_counts(
                expected, deviation_eligible, deviation_scored
            ),
            "effect_spearman": _endpoint_counts(
                expected, effect_eligible, effect_scored
            ),
            "shift_direction_accuracy": _endpoint_counts(
                expected, shift_eligible, shift_scored
            ),
            "shift_mae_bp": _endpoint_counts(expected, shift_eligible, shift_scored),
            "average_precision": {
                **_endpoint_counts(expected, ranking_eligible, ranking_scored),
                "positives": int(observed_calls.sum()),
            },
        }
        return cls(
            deviation_pearson=_mean_defined(deviation[deviation_scored]),
            effect_spearman=_spearman(
                predicted_effect[effect_scored], observed_effect[effect_scored]
            ),
            shift_direction_accuracy=shift_direction_accuracy,
            shift_mae_bp=shift_mae_bp,
            average_precision=_average_precision(
                predicted_robust_z[ranking_scored],
                observed_calls[ranking_scored],
            ),
            target_locus_rows=expected,
            counts=counts,
            coverage_complete=all(
                values["eligible"] == values["scored"] for values in counts.values()
            ),
            endpoint_inputs=inputs,
        )


def _endpoint_counts(
    expected: int, eligible: np.ndarray, scored: np.ndarray
) -> dict[str, int]:
    return {
        "expected": expected,
        "eligible": int(eligible.sum()),
        "scored": int(scored.sum()),
    }


@dataclass(frozen=True)
class DifferentialRankingInputs:
    """Per-target ranking values and observed labels from one locus context."""

    predicted_robust_z: tuple[float, ...]
    observed_calls: tuple[bool, ...]

    def __post_init__(self) -> None:
        if len(self.predicted_robust_z) != len(self.observed_calls):
            raise ValueError("predicted_robust_z and observed_calls must align")


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def _leave_one_out_deviation(profiles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    target_count = profiles.shape[0]
    deviation = (
        target_count
        * (profiles - profiles.mean(axis=0, keepdims=True, dtype=np.float64))
        / (target_count - 1)
    )
    return deviation, profiles - deviation


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    left = left[valid]
    right = right[valid]
    if left.size < 2 or np.std(left) == 0 or np.std(right) == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    left = left[valid]
    right = right[valid]
    if left.size < 2:
        return float("nan")
    return _pearson(_ranks(left), _ranks(right))


def _mean_defined(values: np.ndarray) -> float:
    valid = values[np.isfinite(values)]
    return float(valid.mean()) if valid.size else float("nan")


def _shift_if_defined(
    target: np.ndarray, consensus: np.ndarray, max_shift: int
) -> float:
    if (
        not np.isfinite(target).all()
        or not np.isfinite(consensus).all()
        or np.std(target) == 0
        or np.std(consensus) == 0
    ):
        return float("nan")
    return float(estimate_shift(target, consensus, max_shift))


def _average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    valid = np.isfinite(scores)
    scores = scores[valid]
    labels = labels[valid]
    if not labels.any():
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    labels = labels[order].astype(bool)
    precision = np.cumsum(labels) / np.arange(1, labels.size + 1)
    return float(precision[labels].mean())


def pooled_average_precision(groups: Sequence[DifferentialRankingInputs]) -> float:
    """Calculate one stable average-precision value across all context rows."""
    if not groups:
        return float("nan")
    scores = np.concatenate(
        [np.asarray(group.predicted_robust_z, dtype=np.float64) for group in groups]
    )
    labels = np.concatenate(
        [np.asarray(group.observed_calls, dtype=bool) for group in groups]
    )
    return _average_precision(scores, labels)


def _normalized_context_profiles(
    predicted_logits: np.ndarray,
    observed_raw: np.ndarray,
    center_slice: slice,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    observed_crop = observed_raw[:, center_slice]
    if np.isfinite(predicted_logits).all():
        predicted_crop = _softmax(predicted_logits)[:, center_slice]
    else:
        predicted_crop = np.full_like(observed_crop, np.nan, dtype=np.float64)
    if predicted_crop.shape[1] == 0:
        raise ValueError("center_slice must select at least one position")
    _, predicted_profiles = normalize_profiles(predicted_crop, sigma)
    _, observed_profiles = normalize_profiles(observed_crop, sigma)
    return predicted_profiles, observed_profiles


def _ranking_inputs_from_profiles(
    predicted_profiles: np.ndarray, observed_calls: np.ndarray
) -> DifferentialRankingInputs:
    target_locus_rows = predicted_profiles.shape[0]
    if target_locus_rows < 2:
        predicted_robust_z = np.full(target_locus_rows, np.nan)
    else:
        _, consensus = _leave_one_out_deviation(predicted_profiles)
        effects = np.asarray(
            [
                _total_variation(predicted_profiles[index], consensus[index])
                for index in range(target_locus_rows)
            ]
        )
        predicted_robust_z = _robust_z(effects)
    return DifferentialRankingInputs(
        tuple(float(value) for value in predicted_robust_z),
        tuple(bool(value) for value in observed_calls),
    )


def score_context_ranking_inputs(
    predicted_logits: np.ndarray,
    observed_profiles: np.ndarray,
    observed_calls: np.ndarray,
    center_slice: slice,
    *,
    sigma: float = 20.0,
) -> DifferentialRankingInputs:
    """Return per-target predicted ranking scores and observed call labels."""
    predicted_logits = np.asarray(predicted_logits, dtype=np.float64)
    observed_profiles = np.asarray(observed_profiles, dtype=np.float64)
    observed_calls = np.asarray(observed_calls, dtype=bool)
    if predicted_logits.ndim != 2 or observed_profiles.ndim != 2:
        raise ValueError(
            "predicted_logits and observed_profiles must have shape [cell, position]"
        )
    if predicted_logits.shape != observed_profiles.shape:
        raise ValueError(
            "predicted_logits and observed_profiles must have matching shapes"
        )
    if observed_calls.shape != (predicted_logits.shape[0],):
        raise ValueError("observed_calls must have one value per target cell")
    if (
        not np.isfinite(predicted_logits).all()
        or not np.isfinite(observed_profiles).all()
    ):
        raise ValueError("predicted_logits and observed_profiles must be finite")
    if predicted_logits.shape[0] < 2:
        return DifferentialRankingInputs(
            tuple(float("nan") for _ in observed_calls),
            tuple(bool(value) for value in observed_calls),
        )
    predicted_profiles, _ = _normalized_context_profiles(
        predicted_logits, observed_profiles, center_slice, sigma
    )
    return _ranking_inputs_from_profiles(predicted_profiles, observed_calls)


def score_context_predictions(
    predicted_logits: np.ndarray,
    observed_raw: np.ndarray,
    observed_calls: np.ndarray,
    center_slice: slice,
    *,
    sigma: float = 20.0,
    max_shift: int = 100,
) -> DifferentialMetrics:
    """Score predictions for all target cells in one locus tissue context.

    Both profile sources are smoothed and normalized in the supplied central
    window before leave-one-out differential effects are evaluated.
    """

    predicted_logits = np.asarray(predicted_logits, dtype=np.float64)
    observed_raw = np.asarray(observed_raw, dtype=np.float64)
    observed_calls = np.asarray(observed_calls, dtype=bool)
    if predicted_logits.ndim != 2 or observed_raw.ndim != 2:
        raise ValueError(
            "predicted_logits and observed_raw must have shape [cell, position]"
        )
    if predicted_logits.shape != observed_raw.shape:
        raise ValueError("predicted_logits and observed_raw must have matching shapes")
    if observed_calls.shape != (predicted_logits.shape[0],):
        raise ValueError("observed_calls must have one value per target cell")
    if not np.isfinite(observed_raw).all():
        raise ValueError("observed_raw must be finite")

    target_locus_rows = predicted_logits.shape[0]
    if target_locus_rows < 2:
        return DifferentialMetrics.from_endpoint_inputs(
            DifferentialEndpointInputs(
                deviation_pearsons=tuple(
                    float("nan") for _ in range(target_locus_rows)
                ),
                deviation_eligible=tuple(False for _ in range(target_locus_rows)),
                predicted_effects=tuple(float("nan") for _ in range(target_locus_rows)),
                observed_effects=tuple(float("nan") for _ in range(target_locus_rows)),
                predicted_shifts=tuple(float("nan") for _ in range(target_locus_rows)),
                observed_shifts=tuple(float("nan") for _ in range(target_locus_rows)),
                predicted_robust_z=tuple(
                    float("nan") for _ in range(target_locus_rows)
                ),
                observed_calls=tuple(bool(value) for value in observed_calls),
            )
        )

    predicted_profiles, observed_profiles = _normalized_context_profiles(
        predicted_logits, observed_raw, center_slice, sigma
    )

    predicted_deviation, predicted_consensus = _leave_one_out_deviation(
        predicted_profiles
    )
    observed_deviation, observed_consensus = _leave_one_out_deviation(observed_profiles)
    deviation_pearsons = np.asarray(
        [
            _pearson(predicted_deviation[index], observed_deviation[index])
            for index in range(target_locus_rows)
        ]
    )
    deviation_eligible = np.asarray(
        [
            math.isfinite(
                _pearson(observed_deviation[index], observed_deviation[index])
            )
            for index in range(target_locus_rows)
        ]
    )

    predicted_effect = np.asarray(
        [
            _total_variation(predicted_profiles[index], predicted_consensus[index])
            for index in range(target_locus_rows)
        ]
    )
    observed_effect = np.asarray(
        [
            _total_variation(observed_profiles[index], observed_consensus[index])
            for index in range(target_locus_rows)
        ]
    )
    predicted_shift = np.asarray(
        [
            _shift_if_defined(
                predicted_profiles[index], predicted_consensus[index], max_shift
            )
            for index in range(target_locus_rows)
        ]
    )
    observed_shift = np.asarray(
        [
            _shift_if_defined(
                observed_profiles[index], observed_consensus[index], max_shift
            )
            for index in range(target_locus_rows)
        ]
    )
    ranking_inputs = _ranking_inputs_from_profiles(predicted_profiles, observed_calls)
    return DifferentialMetrics.from_endpoint_inputs(
        DifferentialEndpointInputs(
            deviation_pearsons=tuple(float(value) for value in deviation_pearsons),
            deviation_eligible=tuple(bool(value) for value in deviation_eligible),
            predicted_effects=tuple(float(value) for value in predicted_effect),
            observed_effects=tuple(float(value) for value in observed_effect),
            predicted_shifts=tuple(float(value) for value in predicted_shift),
            observed_shifts=tuple(float(value) for value in observed_shift),
            predicted_robust_z=ranking_inputs.predicted_robust_z,
            observed_calls=ranking_inputs.observed_calls,
        )
    )
