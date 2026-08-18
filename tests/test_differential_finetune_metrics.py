from __future__ import annotations

import unittest

import numpy as np

from chrombpnet.multitask.differential_finetune_metrics import (
    DifferentialRankingInputs,
    pooled_average_precision,
    score_context_predictions,
    score_context_ranking_inputs,
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

    def test_ranking_inputs_preserve_per_target_scores_and_labels(self):
        perfect_logits, observed, observed_calls, center_slice = known_shift_context()

        ranking_inputs = score_context_ranking_inputs(
            perfect_logits, observed, observed_calls, center_slice
        )
        self.assertEqual(ranking_inputs.observed_calls, (True, False, False, False))
        self.assertEqual(len(ranking_inputs.predicted_robust_z), 4)
        self.assertEqual(
            pooled_average_precision((ranking_inputs,)),
            score_context_predictions(
                perfect_logits, observed, observed_calls, center_slice
            ).average_precision,
        )

    def test_pooled_average_precision_scores_all_context_rows_once(self):
        first = DifferentialRankingInputs((0.2, 0.1), (True, False))
        second = DifferentialRankingInputs(
            (0.9, 0.8, 0.7, 0.6), (False, False, False, True)
        )
        pooled = pooled_average_precision((first, second))

        self.assertAlmostEqual(pooled, 0.325)
        self.assertNotAlmostEqual(
            pooled,
            (pooled_average_precision((first,)) + pooled_average_precision((second,)))
            / 2.0,
        )

    def test_pooled_average_precision_returns_nan_without_ranking_positives(self):
        self.assertTrue(np.isnan(pooled_average_precision(())))
        no_positive = DifferentialRankingInputs((0.2, -0.1), (False, False))
        self.assertTrue(np.isnan(pooled_average_precision((no_positive,))))
