from __future__ import annotations

import unittest

import numpy as np

from chrombpnet.multitask.differential_finetune_metrics import (
    score_context_predictions,
)


def known_shift_context() -> tuple[np.ndarray, np.ndarray, np.ndarray, slice]:
    positions = np.arange(503, dtype=np.float64)
    centers = np.array([266.0, 251.0, 251.0, 251.0])
    observed = np.exp(-0.5 * ((positions[None, :] - centers[:, None]) / 5.0) ** 2)
    observed = 100.0 * observed
    return (
        np.log(observed + 1e-12),
        observed,
        np.array([True, False, False, False]),
        slice(1, 502),
    )


class DifferentialFineTuneMetricsTests(unittest.TestCase):
    def test_perfect_known_shift_scores_all_defined_metrics(self):
        perfect_logits, observed, observed_calls, center_slice = known_shift_context()

        metrics = score_context_predictions(
            perfect_logits, observed, observed_calls, center_slice
        )

        self.assertGreater(metrics.deviation_pearson, 0.99)
        self.assertGreater(metrics.effect_spearman, 0.99)
        self.assertEqual(metrics.shift_direction_accuracy, 1.0)
        self.assertEqual(metrics.shift_mae_bp, 0.0)
        self.assertEqual(metrics.average_precision, 1.0)
        self.assertEqual(metrics.target_locus_rows, 4)

    def test_zero_variance_context_returns_nan_for_undefined_metrics(self):
        logits = np.zeros((4, 503), dtype=np.float32)
        observed = np.ones((4, 503), dtype=np.float32)

        metrics = score_context_predictions(
            logits, observed, np.array([True, False, False, False]), slice(1, 502)
        )

        self.assertTrue(np.isnan(metrics.deviation_pearson))
        self.assertTrue(np.isnan(metrics.effect_spearman))
        self.assertTrue(np.isnan(metrics.shift_direction_accuracy))
        self.assertTrue(np.isnan(metrics.shift_mae_bp))

    def test_no_positive_observed_calls_returns_nan_average_precision(self):
        perfect_logits, observed, _, center_slice = known_shift_context()

        metrics = score_context_predictions(
            perfect_logits, observed, np.zeros(4, dtype=bool), center_slice
        )

        self.assertTrue(np.isnan(metrics.average_precision))
