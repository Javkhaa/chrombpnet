"""Differential nucleosome metrics for one locus-context prediction group."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from chrombpnet.multitask.differential_nucleosomes import (
    _robust_z,
    _total_variation,
    estimate_shift,
    normalize_profiles,
)


@dataclass(frozen=True)
class DifferentialMetrics:
    """Agreement between predicted and observed differential profiles."""

    deviation_pearson: float
    effect_spearman: float
    shift_direction_accuracy: float
    shift_mae_bp: float
    average_precision: float
    target_locus_rows: int


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
    if np.std(target) == 0 or np.std(consensus) == 0:
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
    if not np.isfinite(predicted_logits).all() or not np.isfinite(observed_raw).all():
        raise ValueError("predicted_logits and observed_raw must be finite")

    target_locus_rows = predicted_logits.shape[0]
    if target_locus_rows < 2:
        undefined = float("nan")
        return DifferentialMetrics(
            undefined, undefined, undefined, undefined, undefined, target_locus_rows
        )

    predicted_crop = _softmax(predicted_logits)[:, center_slice]
    observed_crop = observed_raw[:, center_slice]
    if predicted_crop.shape[1] == 0:
        raise ValueError("center_slice must select at least one position")
    _, predicted_profiles = normalize_profiles(predicted_crop, sigma)
    _, observed_profiles = normalize_profiles(observed_crop, sigma)

    predicted_deviation, predicted_consensus = _leave_one_out_deviation(
        predicted_profiles
    )
    observed_deviation, observed_consensus = _leave_one_out_deviation(observed_profiles)
    deviation_pearson = _mean_defined(
        np.asarray(
            [
                _pearson(predicted_deviation[index], observed_deviation[index])
                for index in range(target_locus_rows)
            ]
        )
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
    effect_spearman = _spearman(predicted_effect, observed_effect)

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
    shift_rows = (
        np.isfinite(predicted_shift)
        & np.isfinite(observed_shift)
        & (np.abs(observed_shift) >= 10)
    )
    if shift_rows.any():
        shift_direction_accuracy = float(
            (
                np.sign(predicted_shift[shift_rows])
                == np.sign(observed_shift[shift_rows])
            ).mean()
        )
        shift_mae_bp = float(
            np.abs(predicted_shift[shift_rows] - observed_shift[shift_rows]).mean()
        )
    else:
        shift_direction_accuracy = float("nan")
        shift_mae_bp = float("nan")

    return DifferentialMetrics(
        deviation_pearson=deviation_pearson,
        effect_spearman=effect_spearman,
        shift_direction_accuracy=shift_direction_accuracy,
        shift_mae_bp=shift_mae_bp,
        average_precision=_average_precision(
            _robust_z(predicted_effect), observed_calls
        ),
        target_locus_rows=target_locus_rows,
    )
