from __future__ import annotations

import json
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
import polars as pl
import pyBigWig
import torch

from chrombpnet.multitask.differential_finetune import (
    configure_adult_profile_finetune,
)
from chrombpnet.multitask.differential_finetune_data import PairedContextExample
from chrombpnet.multitask.differential_finetune_metrics import (
    DifferentialMetrics,
    DifferentialRankingInputs,
)
from chrombpnet.multitask.fine_tune_differential_torch import (
    FineTuneConfig,
    RunResult,
    aggregate_differential_metrics,
    assert_production_cardinalities,
    build_adult_replay_loader,
    build_parser,
    learning_rate_for_step,
    load_hybrid_checkpoint,
    load_selected_lambda,
    passes_gates,
    run_pilot_sweep,
    start_wandb_run,
    train_one_step,
    validate_call_splits,
)
from chrombpnet.training.models.multitask_nucleosome_torch import (
    ConditionedMultiCellModel,
    HybridMultiCellModel,
)


def checkpoint_args(*, hybrid: bool) -> dict[str, object]:
    return {
        "hybrid": hybrid,
        "inputlen": 128,
        "outputlen": 20,
        "filters": 4,
        "n_dil_layers": 1,
        "embed_dim": 2,
        "cond_mode": "film",
    }


def tiny_hybrid(n_cell_types: int = 6) -> HybridMultiCellModel:
    return HybridMultiCellModel(
        n_cell_types=n_cell_types,
        inputlen=601,
        outputlen=503,
        filters=4,
        n_dil_layers=1,
        embed_dim=2,
    )


class HybridCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_reloads_hybrid_checkpoint_with_exact_cell_order(self):
        cell_types = ["a", "b", "c", "d"]
        args = checkpoint_args(hybrid=True)
        source = HybridMultiCellModel(
            len(cell_types),
            inputlen=args["inputlen"],
            outputlen=args["outputlen"],
            filters=args["filters"],
            n_dil_layers=args["n_dil_layers"],
            embed_dim=args["embed_dim"],
            cond_mode=args["cond_mode"],
        )
        path = self.root / "hybrid.pt"
        torch.save(
            {
                "args": args,
                "cell_types": cell_types,
                "model_state_dict": source.state_dict(),
            },
            path,
        )

        model, checkpoint = load_hybrid_checkpoint(path, torch.device("cpu"))

        self.assertEqual(checkpoint["cell_types"], cell_types)
        self.assertIsInstance(model, HybridMultiCellModel)
        for name, value in source.state_dict().items():
            torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)

    def test_rejects_conditioned_shared_head_checkpoint(self):
        cell_types = ["a", "b", "c", "d"]
        args = checkpoint_args(hybrid=False)
        source = ConditionedMultiCellModel(
            len(cell_types),
            inputlen=args["inputlen"],
            outputlen=args["outputlen"],
            filters=args["filters"],
            n_dil_layers=args["n_dil_layers"],
            embed_dim=args["embed_dim"],
            cond_mode=args["cond_mode"],
        )
        path = self.root / "conditioned.pt"
        torch.save(
            {
                "args": args,
                "cell_types": cell_types,
                "model_state_dict": source.state_dict(),
            },
            path,
        )

        with self.assertRaisesRegex(ValueError, "HybridMultiCellModel"):
            load_hybrid_checkpoint(path, torch.device("cpu"))


class CliContractTests(unittest.TestCase):
    def test_train_help_contains_required_inputs(self):
        parser = build_parser()
        train = parser._subparsers._group_actions[0].choices["train"]

        help_text = train.format_help()

        for option in (
            "--checkpoint",
            "--manifest",
            "--metadata",
            "--fold-json",
            "--train-calls",
            "--valid-calls",
            "--genome",
            "--out-dir",
        ):
            self.assertIn(option, help_text)

    def test_evaluate_help_contains_required_inputs(self):
        parser = build_parser()
        evaluate = parser._subparsers._group_actions[0].choices["evaluate"]

        help_text = evaluate.format_help()

        for option in (
            "--base-checkpoint",
            "--tuned-checkpoint",
            "--manifest",
            "--metadata",
            "--fold-json",
            "--calls",
            "--genome",
            "--out-json",
        ):
            self.assertIn(option, help_text)

    def test_train_modes_are_mutually_exclusive_and_defaults_are_approved(self):
        parser = build_parser()
        common = [
            "train",
            "--checkpoint",
            "base.pt",
            "--manifest",
            "manifest.tsv",
            "--metadata",
            "metadata.tsv",
            "--fold-json",
            "fold.json",
            "--train-calls",
            "train.parquet",
            "--valid-calls",
            "valid.parquet",
            "--genome",
            "genome.fa",
            "--out-dir",
            "out",
        ]

        args = parser.parse_args(common + ["--lambda-dev-grid", "0.25", "1", "4"])
        self.assertEqual(args.learning_rate, 1e-5)
        self.assertEqual(args.alpha, 1.0)
        self.assertEqual(args.pilot_steps, 2_000)
        self.assertEqual(args.full_steps, 10_000)
        self.assertEqual(args.warmup_steps, 100)
        self.assertEqual(args.min_learning_rate, 1e-6)
        self.assertEqual(args.grad_clip, 1.0)
        self.assertEqual(args.validate_every, 250)
        self.assertEqual(args.wandb_project, "jg_experiments")
        with self.assertRaises(SystemExit):
            parser.parse_args(
                common
                + [
                    "--lambda-dev-grid",
                    "1",
                    "--selected-pilot-json",
                    "selected.json",
                ]
            )


class ReplayLoaderTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.genome = self.root / "genome.fa"
        self.genome.write_text(">chr2\n" + "ACGT" * 500 + "\n")

    def tearDown(self):
        self.temporary_directory.cleanup()

    def _write_regions_and_tracks(self, cell_type: str) -> dict[str, object]:
        regions = self.root / f"{cell_type}.bed"
        regions.write_text("chr2\t900\t1100\tpeak\t1000\t.\t10\t10\t10\t100\n")
        paths = {}
        for track in ("acc", "nuc"):
            path = self.root / f"{cell_type}.{track}.bw"
            with pyBigWig.open(str(path), "w") as bigwig:
                bigwig.addHeader([("chr2", 2_000)])
                bigwig.addEntries(
                    ["chr2"],
                    [0],
                    ends=[2_000],
                    values=[1.0 if track == "acc" else 2.0],
                )
            paths[track] = str(path)
        return {
            "cell_type": cell_type,
            "peaks": str(regions),
            "nonpeaks": str(regions),
            "acc_bw": paths["acc"],
            "nuc_bw": paths["nuc"],
        }

    def test_tags_replay_rows_with_checkpoint_indices(self):
        cohort = pl.DataFrame(
            [
                self._write_regions_and_tracks("adult_b"),
                self._write_regions_and_tracks("adult_a"),
            ]
        )

        loader = build_adult_replay_loader(
            cohort,
            {"adult_a": 1, "adult_b": 4},
            {"chr2"},
            self.genome,
            inputlen=601,
            outputlen=502,
            batch_size=4,
            num_workers=0,
            shuffle=False,
            seed=7,
        )
        batch = next(iter(loader))

        self.assertEqual(len(batch), 6)
        self.assertEqual(batch[0].shape, (4, 601, 4))
        self.assertEqual(batch[3].shape, (4, 502))
        self.assertEqual(batch[5].tolist(), [4, 4, 1, 1])


class OptimizerStepTests(unittest.TestCase):
    def test_one_step_preserves_frozen_rows_and_accessibility(self):
        torch.manual_seed(19)
        model = tiny_hybrid()
        guard = configure_adult_profile_finetune(model, adult_indices=[1, 2, 3, 4])
        optimizer = guard.make_optimizer(lr=1e-2)
        sequence = torch.randn(601, 4)
        paired = PairedContextExample(
            sequence=sequence,
            raw_profiles=torch.rand(4, 503) * 10,
            checkpoint_indices=torch.tensor([1, 2, 3, 4]),
            observed_calls=torch.tensor([True, False, False, False]),
            deviation_slice=slice(1, 502),
            locus_id="chr2:1000",
            tissue_context="brain",
        )
        replay = (
            torch.randn(2, 601, 4),
            torch.rand(2, 503),
            torch.rand(2, 1),
            torch.rand(2, 503) * 10,
            torch.rand(2, 1),
            torch.tensor([1, 4]),
        )
        before_w = model.nuc_profile_w.detach().clone()
        with torch.no_grad():
            before_accessibility = model(sequence.unsqueeze(0), torch.tensor([1]))[:2]

        result = train_one_step(
            model,
            guard,
            optimizer,
            paired,
            replay,
            device=torch.device("cpu"),
            alpha=1.0,
            lambda_dev=1.0,
            grad_clip=1.0,
        )

        for value in (
            result.total_loss,
            result.replay_nll,
            result.differential_profile_nll,
            result.deviation_loss,
            result.gradient_norm,
        ):
            self.assertTrue(np.isfinite(value))
        self.assertFalse(torch.equal(before_w[1:5], model.nuc_profile_w[1:5]))
        torch.testing.assert_close(
            before_w[[0, 5]], model.nuc_profile_w[[0, 5]], rtol=0, atol=0
        )
        with torch.no_grad():
            after_accessibility = model(sequence.unsqueeze(0), torch.tensor([1]))[:2]
        for before, after in zip(
            before_accessibility, after_accessibility, strict=True
        ):
            torch.testing.assert_close(before, after, rtol=0, atol=0)
        guard.teardown()


class PreservationGateTests(unittest.TestCase):
    def setUp(self):
        self.base = {
            "nuc_profile_pearson": 0.50,
            "nuc_profile_nll": 100.0,
            "shift_direction_accuracy": 0.75,
        }

    def test_accepts_metrics_at_preservation_boundaries(self):
        tuned = {
            "nuc_profile_pearson": 0.49,
            "nuc_profile_nll": 102.0,
            "shift_direction_accuracy": 0.75,
            "accessibility_unchanged": True,
            "non_adult_unchanged": True,
        }
        self.assertTrue(passes_gates(self.base, tuned))

    def test_rejects_profile_drop_over_point_zero_one(self):
        tuned = dict(self.base, nuc_profile_pearson=0.489)
        self.assertFalse(passes_gates(self.base, tuned))

    def test_rejects_nll_increase_over_two_percent(self):
        tuned = dict(self.base, nuc_profile_nll=102.01)
        self.assertFalse(passes_gates(self.base, tuned))

    def test_rejects_shift_decline_or_invariant_failure(self):
        self.assertFalse(
            passes_gates(
                self.base,
                dict(self.base, shift_direction_accuracy=0.74),
            )
        )
        self.assertFalse(
            passes_gates(
                self.base,
                dict(self.base, accessibility_unchanged=False),
            )
        )

    def test_load_selected_lambda_requires_a_passing_selection(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "selected.json"
            path.write_text('{"lambda_dev": 4.0, "passes_gates": true}')
            self.assertEqual(load_selected_lambda(path), 4.0)
            path.write_text('{"lambda_dev": 4.0, "passes_gates": false}')
            with self.assertRaisesRegex(ValueError, "passing"):
                load_selected_lambda(path)


class OrchestrationContractTests(unittest.TestCase):
    def test_train_and_validation_artifacts_are_rejected_at_cli_boundary(self):
        valid = pl.DataFrame(
            {
                "locus_id": ["valid"],
                "chrom": ["chr8"],
                "center": [800],
                "lineage": ["brain"],
            }
        )
        train_with_test = pl.DataFrame(
            {
                "locus_id": ["leak"],
                "chrom": ["chr1"],
                "center": [100],
                "lineage": ["brain"],
            }
        )
        with self.assertRaisesRegex(ValueError, "forbidden.*chr1"):
            validate_call_splits(train_with_test, valid, {"chr2", "chr4"})

        train = train_with_test.with_columns(pl.lit("chr2").alias("chrom"))
        overlapping_valid = valid.with_columns(
            pl.lit("leak").alias("locus_id"),
            pl.lit(100).alias("center"),
            pl.lit("brain").alias("lineage"),
        )
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_call_splits(train, overlapping_valid, {"chr2", "chr4"})

    def test_fold_cannot_enable_test_chromosome_replay_gradients(self):
        train = pl.DataFrame({"chrom": ["chr2"]})
        valid = pl.DataFrame({"chrom": ["chr8"]})

        with self.assertRaisesRegex(ValueError, "fold.*chr1"):
            validate_call_splits(train, valid, {"chr1", "chr2"})

    def test_learning_rate_uses_exact_warmup_and_cosine_endpoints(self):
        self.assertAlmostEqual(
            learning_rate_for_step(1, 1_000, 100, 1e-5, 1e-6),
            1e-7,
        )
        self.assertEqual(
            learning_rate_for_step(100, 1_000, 100, 1e-5, 1e-6),
            1e-5,
        )
        self.assertAlmostEqual(
            learning_rate_for_step(1_000, 1_000, 100, 1e-5, 1e-6),
            1e-6,
        )

    def test_production_cardinalities_are_asserted_before_mapping(self):
        checkpoint_cell_types = [f"cell_{index}" for index in range(346)]
        adults = pl.DataFrame(
            {
                "cell_type": checkpoint_cell_types[:244],
                "lineage": ["adult"] * 244,
                "is_adult": [True] * 244,
                "is_cell_line": [False] * 244,
            }
        )

        assert_production_cardinalities(checkpoint_cell_types, adults)
        with self.assertRaisesRegex(ValueError, "346"):
            assert_production_cardinalities(checkpoint_cell_types[:-1], adults)
        with self.assertRaisesRegex(ValueError, "244"):
            assert_production_cardinalities(checkpoint_cell_types, adults.head(243))

    def test_cardinality_check_reads_string_booleans_from_raw_metadata(self):
        checkpoint_cell_types = [f"cell_{index}" for index in range(346)]
        metadata = pl.DataFrame(
            {
                "cell_type": checkpoint_cell_types[:244],
                "is_adult": ["true"] * 244,
                "is_cell_line": ["false"] * 244,
            }
        )

        assert_production_cardinalities(checkpoint_cell_types, metadata)
        extra = pl.concat(
            [
                metadata,
                pl.DataFrame(
                    {
                        "cell_type": ["extra_adult"],
                        "is_adult": ["true"],
                        "is_cell_line": ["false"],
                    }
                ),
            ]
        )
        with self.assertRaisesRegex(ValueError, "244"):
            assert_production_cardinalities(checkpoint_cell_types, extra)

    def test_aggregate_metrics_pools_auprc_across_all_target_rows(self):
        metrics = [
            DifferentialMetrics(0.2, 0.1, 0.5, 12.0, 1.0, 2),
            DifferentialMetrics(0.8, 0.7, 1.0, 2.0, 0.25, 4),
        ]
        ranking = [
            DifferentialRankingInputs((0.2, 0.1), (True, False)),
            DifferentialRankingInputs(
                (0.9, 0.8, 0.7, 0.6),
                (False, False, False, True),
            ),
        ]

        summary = aggregate_differential_metrics(metrics, ranking)

        self.assertAlmostEqual(summary["deviation_pearson"], 0.6)
        self.assertAlmostEqual(summary["average_precision"], 0.325)

    def test_wandb_failure_warns_and_returns_local_logger(self):
        with (
            patch("wandb.init", side_effect=RuntimeError("offline")),
            warnings.catch_warnings(record=True) as caught,
        ):
            warnings.simplefilter("always")
            logger = start_wandb_run(
                enabled=True,
                project="jg_experiments",
                name="test",
                config={"lambda_dev": 1.0},
            )

        logger.log({"loss": 1.0}, step=1)
        logger.finish()
        self.assertTrue(any("W&B" in str(warning.message) for warning in caught))

    def test_pilot_sweep_reuses_base_checkpoint_and_selects_best_passing_run(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = FineTuneConfig(
                checkpoint=root / "base.pt",
                manifest=root / "manifest.tsv",
                metadata=root / "metadata.tsv",
                fold_json=root / "fold.json",
                train_calls=root / "train.parquet",
                valid_calls=root / "valid.parquet",
                genome=root / "genome.fa",
                out_dir=root / "out",
            )
            observed_configs = []

            def fake_run(candidate):
                observed_configs.append(candidate)
                passing = candidate.lambda_dev != 0.25
                score = {0.25: 0.9, 1.0: 0.4, 4.0: 0.7}[candidate.lambda_dev]
                return RunResult(
                    lambda_dev=candidate.lambda_dev,
                    selected_step=250,
                    checkpoint_path=(
                        candidate.out_dir / f"{candidate.output_name}.pt"
                        if passing
                        else None
                    ),
                    report_path=candidate.out_dir / f"{candidate.output_name}.json",
                    validation={
                        "deviation_pearson": score,
                        "shift_direction_accuracy": 1.0,
                    },
                    passes_gates=passing,
                )

            with patch(
                "chrombpnet.multitask.fine_tune_differential_torch.run_single_finetune",
                side_effect=fake_run,
            ):
                result = run_pilot_sweep(config, [0.25, 1.0, 4.0])

            self.assertEqual(result.selected_lambda, 4.0)
            self.assertTrue(
                all(item.checkpoint == config.checkpoint for item in observed_configs)
            )
            self.assertEqual(
                [item.output_name for item in observed_configs],
                ["pilot_lambda_0.25", "pilot_lambda_1", "pilot_lambda_4"],
            )
            selection = json.loads(result.selection_path.read_text())
            self.assertEqual(selection["lambda_dev"], 4.0)
            self.assertTrue(selection["passes_gates"])

    def test_pilot_sweep_refuses_to_overwrite_existing_selection(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            out_dir = root / "out"
            out_dir.mkdir()
            (out_dir / "selected_pilot.json").write_text("{}")
            config = FineTuneConfig(
                checkpoint=root / "base.pt",
                manifest=root / "manifest.tsv",
                metadata=root / "metadata.tsv",
                fold_json=root / "fold.json",
                train_calls=root / "train.parquet",
                valid_calls=root / "valid.parquet",
                genome=root / "genome.fa",
                out_dir=out_dir,
            )

            with self.assertRaisesRegex(FileExistsError, "selected_pilot"):
                run_pilot_sweep(config, [1.0])


if __name__ == "__main__":
    unittest.main()
