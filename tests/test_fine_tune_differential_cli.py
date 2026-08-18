from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

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
    APPROVED_TEST_CHROMOSOMES,
    APPROVED_TRAIN_CHROMOSOMES,
    APPROVED_VALID_CHROMOSOMES,
    FineTuneConfig,
    RunResult,
    WandbRun,
    _prepare_run,
    aggregate_differential_metrics,
    assert_production_cardinalities,
    build_adult_replay_loader,
    build_parser,
    evaluate_checkpoints,
    learning_rate_for_step,
    load_hybrid_checkpoint,
    load_selected_lambda,
    passes_gates,
    run_pilot_sweep,
    run_single_finetune,
    start_wandb_run,
    train_one_step,
    validate_adult_manifest_coverage,
    validate_call_splits,
    validate_fold_boundaries,
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
        tuned = dict(
            self.base,
            nuc_profile_pearson=0.489,
            accessibility_unchanged=True,
            non_adult_unchanged=True,
        )
        self.assertFalse(passes_gates(self.base, tuned))

    def test_rejects_nll_increase_over_two_percent(self):
        tuned = dict(
            self.base,
            nuc_profile_nll=102.01,
            accessibility_unchanged=True,
            non_adult_unchanged=True,
        )
        self.assertFalse(passes_gates(self.base, tuned))

    def test_rejects_shift_decline_or_invariant_failure(self):
        self.assertFalse(
            passes_gates(
                self.base,
                dict(
                    self.base,
                    shift_direction_accuracy=0.74,
                    accessibility_unchanged=True,
                    non_adult_unchanged=True,
                ),
            )
        )
        self.assertFalse(
            passes_gates(
                self.base,
                dict(
                    self.base,
                    accessibility_unchanged=False,
                    non_adult_unchanged=True,
                ),
            )
        )

    def test_rejects_every_missing_undefined_or_nonfinite_gate_input(self):
        tuned = {
            **self.base,
            "accessibility_unchanged": True,
            "non_adult_unchanged": True,
        }
        metric_cases = [
            ("base", "nuc_profile_pearson"),
            ("base", "nuc_profile_nll"),
            ("base", "shift_direction_accuracy"),
            ("tuned", "nuc_profile_pearson"),
            ("tuned", "nuc_profile_nll"),
            ("tuned", "shift_direction_accuracy"),
        ]
        for side, key in metric_cases:
            for undefined in ("missing", None, float("nan"), True):
                with self.subTest(side=side, key=key, value=undefined):
                    candidate_base = dict(self.base)
                    candidate_tuned = dict(tuned)
                    target = candidate_base if side == "base" else candidate_tuned
                    if undefined == "missing":
                        target.pop(key)
                    else:
                        target[key] = undefined
                    self.assertFalse(passes_gates(candidate_base, candidate_tuned))
        for key in ("accessibility_unchanged", "non_adult_unchanged"):
            for undefined in ("missing", None, float("nan"), 1):
                with self.subTest(key=key, value=undefined):
                    candidate = dict(tuned)
                    if undefined == "missing":
                        candidate.pop(key)
                    else:
                        candidate[key] = undefined
                    self.assertFalse(passes_gates(self.base, candidate))

    def test_load_selected_lambda_requires_a_passing_selection(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint = Path(temporary_directory) / "base.pt"
            checkpoint.write_bytes(b"base checkpoint")
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            path = Path(temporary_directory) / "selected.json"
            selection = {
                "lambda_dev": 4.0,
                "passes_gates": True,
                "base_checkpoint": str(checkpoint.resolve()),
                "base_checkpoint_sha256": digest,
            }
            path.write_text(json.dumps(selection))
            self.assertEqual(
                load_selected_lambda(path, expected_checkpoint=checkpoint), 4.0
            )
            selection["passes_gates"] = False
            path.write_text(json.dumps(selection))
            with self.assertRaisesRegex(ValueError, "passing"):
                load_selected_lambda(path, expected_checkpoint=checkpoint)

    def test_selected_lambda_rejects_unapproved_or_mismatched_provenance(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            checkpoint = root / "base.pt"
            other = root / "other.pt"
            checkpoint.write_bytes(b"base")
            other.write_bytes(b"other")
            path = root / "selected.json"
            selection = {
                "lambda_dev": 2.0,
                "passes_gates": True,
                "base_checkpoint": str(checkpoint.resolve()),
                "base_checkpoint_sha256": hashlib.sha256(b"base").hexdigest(),
            }
            path.write_text(json.dumps(selection))
            with self.assertRaisesRegex(ValueError, "approved"):
                load_selected_lambda(path, expected_checkpoint=checkpoint)
            selection["lambda_dev"] = 1.0
            path.write_text(json.dumps(selection))
            with self.assertRaisesRegex(ValueError, "path"):
                load_selected_lambda(path, expected_checkpoint=other)
            del selection["base_checkpoint_sha256"]
            path.write_text(json.dumps(selection))
            with self.assertRaisesRegex(ValueError, "provenance"):
                load_selected_lambda(path, expected_checkpoint=checkpoint)
            with self.assertRaisesRegex(ValueError, "expected checkpoint"):
                load_selected_lambda(path)


class OrchestrationContractTests(unittest.TestCase):
    @staticmethod
    def approved_fold() -> dict[str, list[str]]:
        return {
            "train": sorted(APPROVED_TRAIN_CHROMOSOMES),
            "valid": sorted(APPROVED_VALID_CHROMOSOMES),
            "test": sorted(APPROVED_TEST_CHROMOSOMES),
        }

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

    def test_evaluate_rejects_invalid_fold_before_loading_data(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fold = self.approved_fold()
            fold["train"] = [*fold["train"], "chr1"]
            fold_path = root / "fold.json"
            fold_path.write_text(json.dumps(fold))
            with (
                patch(
                    "chrombpnet.multitask.fine_tune_differential_torch.load_hybrid_checkpoint"
                ) as load_checkpoint,
                self.assertRaisesRegex(ValueError, "fold role 'train'"),
            ):
                evaluate_checkpoints(
                    base_checkpoint_path=root / "base.pt",
                    tuned_checkpoint_path=root / "tuned.pt",
                    manifest_path=root / "manifest.tsv",
                    manifest_root=None,
                    metadata_path=root / "metadata.tsv",
                    fold_json=fold_path,
                    calls_path=root / "calls.parquet",
                    genome=root / "genome.fa",
                    out_json=root / "out.json",
                    device_name="cpu",
                    amp=False,
                )
            load_checkpoint.assert_not_called()

    def test_fold_roles_must_match_the_complete_approved_partition(self):
        fold = self.approved_fold()
        validate_fold_boundaries(fold)
        for role, replacement in (
            ("train", ["chr2"]),
            ("valid", ["chr8"]),
            ("test", ["chr3", "chr6"]),
        ):
            with self.subTest(role=role):
                invalid = {key: list(value) for key, value in fold.items()}
                invalid[role] = replacement
                with self.assertRaisesRegex(ValueError, f"fold role '{role}'"):
                    validate_fold_boundaries(invalid)

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

    def test_adult_manifest_must_exactly_cover_metadata_before_mapping(self):
        metadata = pl.DataFrame(
            {
                "cell_type": ["adult_a", "adult_b"],
                "is_adult": [True, True],
                "is_cell_line": [False, False],
            }
        )
        complete = pl.DataFrame({"cell_type": ["adult_a", "adult_b"]})
        validate_adult_manifest_coverage(complete, metadata)
        with self.assertRaisesRegex(ValueError, "missing=.*adult_b"):
            validate_adult_manifest_coverage(complete.head(1), metadata)
        duplicate = pl.DataFrame({"cell_type": ["adult_a", "adult_a"]})
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_adult_manifest_coverage(duplicate, metadata)

    def test_prepare_run_rejects_missing_or_duplicate_adult_rows_before_mapping(self):
        checkpoint_cell_types = [f"cell_{index}" for index in range(346)]
        metadata = pl.DataFrame(
            {
                "cell_type": checkpoint_cell_types,
                "lineage": ["brain"] * 346,
                "is_adult": [True] * 244 + [False] * 102,
                "is_cell_line": [False] * 346,
            }
        )
        complete = metadata.head(244).select("cell_type")
        config = FineTuneConfig(
            checkpoint=Path("base.pt"),
            manifest=Path("manifest.tsv"),
            metadata=Path("metadata.tsv"),
            fold_json=Path("fold.json"),
            train_calls=Path("train.parquet"),
            valid_calls=Path("valid.parquet"),
            genome=Path("genome.fa"),
            out_dir=Path("out"),
        )
        cases = (
            ("missing", complete.head(243)),
            (
                "duplicate",
                pl.concat([complete.head(243), complete.head(1)]),
            ),
        )
        for label, adult_manifest in cases:
            with (
                self.subTest(label=label),
                patch(
                    "chrombpnet.multitask.fine_tune_differential_torch.load_hybrid_checkpoint",
                    return_value=(
                        Mock(),
                        {"cell_types": checkpoint_cell_types, "args": {}},
                    ),
                ),
                patch(
                    "chrombpnet.multitask.fine_tune_differential_torch.read_manifest",
                    return_value=Mock(),
                ),
                patch(
                    "chrombpnet.multitask.fine_tune_differential_torch.read_metadata",
                    return_value=metadata,
                ),
                patch(
                    "chrombpnet.multitask.fine_tune_differential_torch.validate_adult_cohort",
                    return_value=adult_manifest,
                ),
                patch(
                    "chrombpnet.multitask.fine_tune_differential_torch.map_adult_checkpoint_indices"
                ) as mapper,
                self.assertRaisesRegex(ValueError, label),
            ):
                _prepare_run(config, torch.device("cpu"))
            mapper.assert_not_called()

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
            config.checkpoint.write_bytes(b"base checkpoint")
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
            self.assertEqual(
                selection["base_checkpoint"], str(config.checkpoint.resolve())
            )
            self.assertEqual(
                selection["base_checkpoint_sha256"],
                hashlib.sha256(b"base checkpoint").hexdigest(),
            )

    def test_pilot_sweep_requires_the_exact_approved_lambda_set(self):
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
            for values in (
                [0.25, 1.0],
                [0.25, 1.0, 4.0, 8.0],
                [0.25, 1.0, 4.0, 4.0],
                [0.25, 1.0, 2.0],
            ):
                with (
                    self.subTest(values=values),
                    self.assertRaisesRegex(ValueError, "exactly"),
                ):
                    run_pilot_sweep(config, values)

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
                run_pilot_sweep(config, [4.0, 0.25, 1.0])


class CleanupAndWandbTests(unittest.TestCase):
    def _config(self, root: Path) -> FineTuneConfig:
        paths = {
            name: root / name
            for name in (
                "base.pt",
                "manifest.tsv",
                "metadata.tsv",
                "fold.json",
                "train.parquet",
                "valid.parquet",
                "genome.fa",
            )
        }
        for path in paths.values():
            path.write_bytes(b"x")
        return FineTuneConfig(
            checkpoint=paths["base.pt"],
            manifest=paths["manifest.tsv"],
            metadata=paths["metadata.tsv"],
            fold_json=paths["fold.json"],
            train_calls=paths["train.parquet"],
            valid_calls=paths["valid.parquet"],
            genome=paths["genome.fa"],
            out_dir=root / "out",
            steps=1,
            validate_every=1,
            device="cpu",
        )

    def test_setup_failure_attempts_every_previously_acquired_cleanup(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = self._config(root)
            attempted = []

            def fake_prepare(_config, _device, cleanup):
                for label in (
                    "train paired",
                    "valid paired",
                    "train replay",
                    "valid replay",
                ):

                    def close(label=label):
                        attempted.append(label)
                        if label == "valid replay":
                            raise RuntimeError("close failed")

                    cleanup.callback(label, close)
                return Mock()

            with (
                patch(
                    "chrombpnet.multitask.fine_tune_differential_torch._prepare_run",
                    side_effect=fake_prepare,
                ),
                patch(
                    "chrombpnet.multitask.fine_tune_differential_torch.load_hybrid_checkpoint",
                    side_effect=RuntimeError("base scoring setup failed"),
                ),
                warnings.catch_warnings(record=True) as caught,
                self.assertRaisesRegex(RuntimeError, "base scoring setup"),
            ):
                warnings.simplefilter("always")
                run_single_finetune(config)

            self.assertCountEqual(
                attempted,
                ["train paired", "valid paired", "train replay", "valid replay"],
            )
            self.assertTrue(
                any("cleanup failed" in str(item.message) for item in caught)
            )

    def test_optimizer_failure_cleans_guard_and_all_datasets(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = self._config(root)
            attempted = []
            model = tiny_hybrid()
            checkpoint = {"cell_types": [f"cell_{index}" for index in range(6)]}
            prepared = SimpleNamespace(
                model=model,
                checkpoint=checkpoint,
                checkpoint_index={"cell_1": 1},
            )

            def fake_prepare(_config, _device, cleanup):
                for label in ("train", "valid", "replay", "validation"):
                    cleanup.callback(label, lambda label=label: attempted.append(label))
                return prepared

            guard = Mock()
            guard.make_optimizer.side_effect = RuntimeError("optimizer failed")
            guard.teardown.side_effect = lambda: attempted.append("guard")
            with (
                patch(
                    "chrombpnet.multitask.fine_tune_differential_torch._prepare_run",
                    side_effect=fake_prepare,
                ),
                patch(
                    "chrombpnet.multitask.fine_tune_differential_torch.load_hybrid_checkpoint",
                    return_value=(tiny_hybrid(), checkpoint),
                ),
                patch(
                    "chrombpnet.multitask.fine_tune_differential_torch.configure_adult_profile_finetune",
                    return_value=guard,
                ),
                self.assertRaisesRegex(RuntimeError, "optimizer failed"),
            ):
                run_single_finetune(config)
            self.assertCountEqual(
                attempted, ["train", "valid", "replay", "validation", "guard"]
            )

    def test_scoring_and_hash_failures_clean_every_acquired_resource(self):
        for failure_stage in ("score", "hash"):
            with (
                self.subTest(failure_stage=failure_stage),
                tempfile.TemporaryDirectory() as temporary_directory,
            ):
                root = Path(temporary_directory)
                config = self._config(root)
                attempted = []
                model = tiny_hybrid()
                checkpoint = {"cell_types": [f"cell_{index}" for index in range(6)]}
                prepared = SimpleNamespace(
                    model=model,
                    checkpoint=checkpoint,
                    checkpoint_index={"cell_1": 1},
                    train_dataset=Mock(),
                    valid_dataset=Mock(),
                    train_replay=[object()],
                )

                def fake_prepare(
                    _config,
                    _device,
                    cleanup,
                    *,
                    attempted=attempted,
                    prepared=prepared,
                ):
                    for label in ("train", "valid", "replay", "validation"):
                        cleanup.callback(
                            label, lambda label=label: attempted.append(label)
                        )
                    return prepared

                guard = Mock()
                guard.make_optimizer.return_value = Mock()
                guard.teardown.side_effect = lambda attempted=attempted: (
                    attempted.append("guard")
                )
                score_side_effect = (
                    RuntimeError("score failed")
                    if failure_stage == "score"
                    else {"shift_direction_accuracy": 0.5}
                )
                hash_side_effect = (
                    RuntimeError("hash failed") if failure_stage == "hash" else "digest"
                )
                with (
                    patch(
                        "chrombpnet.multitask.fine_tune_differential_torch._prepare_run",
                        side_effect=fake_prepare,
                    ),
                    patch(
                        "chrombpnet.multitask.fine_tune_differential_torch.load_hybrid_checkpoint",
                        return_value=(tiny_hybrid(), checkpoint),
                    ),
                    patch(
                        "chrombpnet.multitask.fine_tune_differential_torch.configure_adult_profile_finetune",
                        return_value=guard,
                    ),
                    patch(
                        "chrombpnet.multitask.fine_tune_differential_torch.ContextBalancedSampler",
                        return_value=[],
                    ),
                    patch(
                        "chrombpnet.multitask.fine_tune_differential_torch._score_differential",
                        side_effect=score_side_effect
                        if isinstance(score_side_effect, Exception)
                        else None,
                        return_value=score_side_effect
                        if isinstance(score_side_effect, dict)
                        else None,
                    ),
                    patch(
                        "chrombpnet.multitask.fine_tune_differential_torch._sha256",
                        side_effect=hash_side_effect
                        if isinstance(hash_side_effect, Exception)
                        else None,
                        return_value=hash_side_effect
                        if isinstance(hash_side_effect, str)
                        else None,
                    ),
                    self.assertRaisesRegex(RuntimeError, failure_stage),
                ):
                    run_single_finetune(config)
                self.assertCountEqual(
                    attempted,
                    ["train", "valid", "replay", "validation", "guard"],
                )

    def test_wandb_receives_final_selection_before_finish(self):
        fake_run = Mock()
        fake_run.config = Mock()
        logger = WandbRun(fake_run)
        selection = {
            "selected_step": 250,
            "lambda_dev": 1.0,
            "passes_gates": True,
            "selected_checkpoint": "/tmp/final.pt",
        }
        logger.log({f"selection/{key}": value for key, value in selection.items()})
        logger.update_config(selection)
        logger.finish()

        logged = fake_run.log.call_args.args[0]
        self.assertEqual(logged["selection/selected_step"], 250)
        self.assertEqual(logged["selection/lambda_dev"], 1.0)
        self.assertIs(logged["selection/passes_gates"], True)
        self.assertEqual(logged["selection/selected_checkpoint"], "/tmp/final.pt")
        fake_run.config.update.assert_called_once_with(selection, allow_val_change=True)
        fake_run.finish.assert_called_once_with()


class RealOrchestrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.cell_types = [f"cell_{index:03d}" for index in range(346)]
        self.adults = self.cell_types[:244]
        self.chromosomes = ("chr1", "chr2", "chr3", "chr6", "chr8", "chr20")
        self._write_corpus()

    def tearDown(self):
        self.temporary_directory.cleanup()

    def _write_bigwig(self, path: Path, shift: int = 0) -> None:
        positions = np.arange(2_000)
        signal = (
            1.0
            + 12.0 * np.exp(-((positions - (1_000 + shift)) ** 2) / (2 * 35**2))
            + 3.0 * np.exp(-((positions - 850) ** 2) / (2 * 50**2))
        ).astype(np.float64)
        with pyBigWig.open(str(path), "w") as bigwig:
            bigwig.addHeader([(chrom, 2_000) for chrom in self.chromosomes])
            for chrom in self.chromosomes:
                bigwig.addEntries(
                    [chrom] * len(positions),
                    positions.tolist(),
                    ends=(positions + 1).tolist(),
                    values=signal.tolist(),
                )

    def _calls(self, chrom: str) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "locus_id": [f"{chrom}:1000"] * 4,
                "chrom": [chrom] * 4,
                "center": [1_000] * 4,
                "lineage": ["brain"] * 4,
                "cell_type": self.adults[:4],
                "passes_depth": [True] * 4,
                "is_differential": [True, False, False, True],
            }
        )

    def _write_corpus(self) -> None:
        genome = self.root / "genome.fa"
        genome.write_text(
            "".join(f">{chrom}\n{('ACGT' * 500)}\n" for chrom in self.chromosomes)
        )
        peaks = self.root / "regions.bed"
        peaks.write_text(
            "".join(
                f"{chrom}\t900\t1100\tpeak\t1000\t.\t10\t10\t10\t100\n"
                for chrom in self.chromosomes
            )
        )
        acc = self.root / "acc.bw"
        self._write_bigwig(acc)
        nuc_paths = []
        for index, shift in enumerate((-30, -10, 10, 30)):
            path = self.root / f"nuc_{index}.bw"
            self._write_bigwig(path, shift)
            nuc_paths.append(path)
        base_nuc = self.root / "nuc_base.bw"
        self._write_bigwig(base_nuc)
        manifest = self.root / "manifest.tsv"
        manifest.write_text(
            "".join(
                "\t".join(
                    (
                        cell_type,
                        str(peaks),
                        str(peaks),
                        str(acc),
                        str(nuc_paths[index] if index < 4 else base_nuc),
                    )
                )
                + "\n"
                for index, cell_type in enumerate(self.adults)
            )
        )
        metadata = self.root / "metadata.tsv"
        metadata.write_text(
            "cell_type\tlineage\tis_adult\tis_cell_line\n"
            + "".join(
                f"{cell_type}\tbrain\t{index < 244}\tFalse\n"
                for index, cell_type in enumerate(self.cell_types)
            )
        )
        fold = {
            "train": sorted(APPROVED_TRAIN_CHROMOSOMES),
            "valid": sorted(APPROVED_VALID_CHROMOSOMES),
            "test": sorted(APPROVED_TEST_CHROMOSOMES),
        }
        (self.root / "fold.json").write_text(json.dumps(fold))
        self._calls("chr2").write_parquet(self.root / "train.parquet")
        self._calls("chr8").write_parquet(self.root / "valid.parquet")
        self._calls("chr3").write_parquet(self.root / "test.parquet")

        args = {
            "hybrid": True,
            "inputlen": 600,
            "outputlen": 502,
            "filters": 2,
            "n_dil_layers": 1,
            "embed_dim": 2,
            "cond_mode": "film",
            "profile_kernel_size": 75,
        }
        torch.manual_seed(9)
        model = HybridMultiCellModel(
            len(self.cell_types),
            args["inputlen"],
            args["outputlen"],
            args["filters"],
            args["n_dil_layers"],
            args["embed_dim"],
            cond_mode=args["cond_mode"],
            profile_kernel_size=args["profile_kernel_size"],
        )
        torch.save(
            {
                "args": args,
                "cell_types": self.cell_types,
                "model_state_dict": model.state_dict(),
            },
            self.root / "base.pt",
        )

    def test_real_training_restart_validation_checkpoint_cleanup_and_evaluation(self):
        config = FineTuneConfig(
            checkpoint=self.root / "base.pt",
            manifest=self.root / "manifest.tsv",
            metadata=self.root / "metadata.tsv",
            fold_json=self.root / "fold.json",
            train_calls=self.root / "train.parquet",
            valid_calls=self.root / "valid.parquet",
            genome=self.root / "genome.fa",
            out_dir=self.root / "run",
            device="cpu",
            lambda_dev=1.0,
            steps=3,
            learning_rate=0.0,
            min_learning_rate=0.0,
            warmup_steps=0,
            validate_every=2,
            replay_batch_size=488,
            validation_replay_examples=4,
            seed=17,
        )

        events = []

        class RecordingWandbRun:
            def log(self, values, *, step=None):
                events.append(("log", dict(values), step))

            def update_config(self, values):
                events.append(("config", dict(values)))

            def finish(self):
                events.append(("finish",))

        with patch(
            "chrombpnet.multitask.fine_tune_differential_torch.start_wandb_run",
            return_value=RecordingWandbRun(),
        ):
            result = run_single_finetune(config)

        self.assertTrue(result.passes_gates)
        self.assertEqual(result.selected_step, 2)
        self.assertTrue(result.checkpoint_path.is_file())
        run_report = json.loads(result.report_path.read_text())
        self.assertEqual(
            [entry["step"] for entry in run_report["validation_history"]], [2, 3]
        )
        self.assertTrue(np.isfinite(run_report["last_train_step"]["total_loss"]))
        self.assertTrue(run_report["passes_gates"])
        selection_log = next(
            event
            for event in events
            if event[0] == "log" and "selection/selected_step" in event[1]
        )
        selection_config = next(event for event in events if event[0] == "config")
        self.assertEqual(selection_log[1]["selection/selected_step"], 2)
        self.assertEqual(selection_log[1]["selection/lambda_dev"], 1.0)
        self.assertIs(selection_log[1]["selection/passes_gates"], True)
        self.assertEqual(
            selection_log[1]["selection/selected_checkpoint"],
            str(result.checkpoint_path.resolve()),
        )
        self.assertEqual(selection_config[1]["selected_step"], 2)
        self.assertEqual(events[-1], ("finish",))

        evaluation_path = self.root / "evaluation.json"
        report = evaluate_checkpoints(
            base_checkpoint_path=self.root / "base.pt",
            tuned_checkpoint_path=result.checkpoint_path,
            manifest_path=self.root / "manifest.tsv",
            manifest_root=None,
            metadata_path=self.root / "metadata.tsv",
            fold_json=self.root / "fold.json",
            calls_path=self.root / "test.parquet",
            genome=self.root / "genome.fa",
            out_json=evaluation_path,
            device_name="cpu",
            amp=False,
            replay_batch_size=4,
            replay_examples=4,
        )

        self.assertTrue(evaluation_path.is_file())
        self.assertEqual(report["chromosomes"], ["chr1", "chr3", "chr6"])
        self.assertEqual(report["base"]["differential"]["context_units"], 1)
        self.assertEqual(report["tuned"]["differential"]["target_locus_rows"], 4)
        self.assertTrue(report["accessibility_unchanged"])
        self.assertTrue(report["non_adult_unchanged"])


if __name__ == "__main__":
    unittest.main()
