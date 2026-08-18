from __future__ import annotations

import copy
import hashlib
import json
import math
import tempfile
import unittest
import warnings
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import polars as pl
import pyBigWig
import torch

import chrombpnet.multitask.fine_tune_differential_torch as fine_tune
from chrombpnet.multitask.differential_finetune import (
    configure_adult_profile_finetune,
)
from chrombpnet.multitask.differential_finetune_data import PairedContextExample
from chrombpnet.multitask.differential_finetune_metrics import (
    DifferentialEndpointInputs,
    DifferentialMetrics,
)
from chrombpnet.multitask.fine_tune_differential_torch import (
    APPROVED_TEST_CHROMOSOMES,
    APPROVED_TRAIN_CHROMOSOMES,
    APPROVED_VALID_CHROMOSOMES,
    FineTuneConfig,
    RunResult,
    WandbRun,
    _close_replay_loader,
    _DatasetCollection,
    _prepare_run,
    _save_finetuned_checkpoint,
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
    multinomial_nll,
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

        with patch.object(torch, "load", wraps=torch.load) as safe_load:
            model, checkpoint = load_hybrid_checkpoint(path, torch.device("cpu"))

        self.assertEqual(checkpoint["cell_types"], cell_types)
        self.assertIs(safe_load.call_args.kwargs["weights_only"], True)
        self.assertIsInstance(model, HybridMultiCellModel)
        for name, value in source.state_dict().items():
            torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)


class AtomicCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def _inputs(self, root: Path):
        base = root / "base.pt"
        base.write_bytes(b"base")
        model = tiny_hybrid()
        prepared = SimpleNamespace(
            checkpoint={
                "args": checkpoint_args(hybrid=True),
                "cell_types": [f"cell_{index}" for index in range(6)],
            }
        )
        config = FineTuneConfig(
            checkpoint=base,
            manifest=root / "manifest.tsv",
            metadata=root / "metadata.tsv",
            fold_json=root / "fold.json",
            train_calls=root / "train.parquet",
            valid_calls=root / "valid.parquet",
            genome=root / "genome.fa",
            out_dir=root,
            lambda_dev=4.0,
        )
        return model, prepared, config

    def test_checkpoint_save_flushes_and_atomically_replaces_destination(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            destination = root / "candidate.pt"
            destination.write_bytes(b"previous passing checkpoint")
            model, prepared, config = self._inputs(root)

            with (
                patch("os.fsync", wraps=__import__("os").fsync) as fsync,
                patch("os.replace", wraps=__import__("os").replace) as replace_file,
            ):
                _save_finetuned_checkpoint(
                    destination, model, prepared, config, step=250
                )

            checkpoint = torch.load(destination, map_location="cpu", weights_only=True)
            self.assertEqual(checkpoint["selected_step"], 250)
            self.assertGreaterEqual(fsync.call_count, 1)
            replace_file.assert_called_once()
            self.assertEqual(replace_file.call_args.args[1], destination)
            self.assertEqual(list(root.glob(".candidate.pt.*.tmp")), [])

    def test_failed_checkpoint_save_retains_previous_destination_and_cleans_temp(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            destination = root / "candidate.pt"
            previous = b"previous passing checkpoint"
            destination.write_bytes(previous)
            model, prepared, config = self._inputs(root)

            def fail_after_partial_write(_payload, handle):
                handle.write(b"partial")
                raise OSError("injected save failure")

            with (
                patch("torch.save", side_effect=fail_after_partial_write),
                patch("os.replace") as replace_file,
                self.assertRaisesRegex(OSError, "injected save failure"),
            ):
                _save_finetuned_checkpoint(
                    destination, model, prepared, config, step=250
                )

            self.assertEqual(destination.read_bytes(), previous)
            replace_file.assert_not_called()
            self.assertEqual(list(root.glob(".candidate.pt.*.tmp")), [])

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
            "--validation-report",
            "--manifest",
            "--metadata",
            "--fold-json",
            "--calls",
            "--genome",
            "--out-json",
        ):
            self.assertIn(option, help_text)

        with self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "evaluate",
                    "--base-checkpoint",
                    "base.pt",
                    "--tuned-checkpoint",
                    "tuned.pt",
                    "--manifest",
                    "manifest.tsv",
                    "--metadata",
                    "metadata.tsv",
                    "--fold-json",
                    "fold.json",
                    "--calls",
                    "calls.parquet",
                    "--genome",
                    "genome.fa",
                    "--out-json",
                    "evaluation.json",
                ]
            )

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

    def test_evaluate_command_fails_closed_when_evaluation_gate_fails(self):
        arguments = [
            "evaluate",
            "--base-checkpoint",
            "base.pt",
            "--tuned-checkpoint",
            "tuned.pt",
            "--validation-report",
            "final_validation.json",
            "--manifest",
            "manifest.tsv",
            "--metadata",
            "metadata.tsv",
            "--fold-json",
            "fold.json",
            "--calls",
            "calls.parquet",
            "--genome",
            "genome.fa",
            "--out-json",
            "evaluation.json",
        ]
        with patch.object(
            fine_tune,
            "evaluate_checkpoints",
            return_value={"passes_evaluation_gate": False},
        ):
            self.assertEqual(fine_tune.main(arguments), 2)


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
        before_b = model.nuc_profile_b.detach().clone()
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
        torch.testing.assert_close(before_b, model.nuc_profile_b, rtol=0, atol=0)
        with torch.no_grad():
            after_accessibility = model(sequence.unsqueeze(0), torch.tensor([1]))[:2]
        for before, after in zip(
            before_accessibility, after_accessibility, strict=True
        ):
            torch.testing.assert_close(before, after, rtol=0, atol=0)
        guard.teardown()


class AmpCompatibilityTests(unittest.TestCase):
    class AmpSensitiveProfileModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.autocast_states = []
            self.register_buffer(
                "profile_pattern", torch.log(torch.arange(1, 5, dtype=torch.float32))
            )

        def forward(self, sequence, _indices):
            self.autocast_states.append(torch.is_autocast_enabled(sequence.device.type))
            logits = (10_000.0 + self.profile_pattern).expand(sequence.shape[0], -1)
            if self.autocast_states[-1]:
                logits = logits.to(torch.bfloat16)
            counts = torch.zeros((sequence.shape[0], 1), device=sequence.device)
            return logits, counts, logits, counts

    def test_constant_profile_rows_return_none_without_warnings(self):
        logits = torch.ones(2, 4)
        observed = torch.ones(2, 4)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pearsons = fine_tune._row_pearsons(logits, observed)
        self.assertEqual(pearsons, [])
        self.assertEqual(caught, [])
        self.assertIsNone(fine_tune._centered_pearson(logits[0], observed[0]))

    def test_amp_profile_forwards_keep_production_scale_positional_signal(self):
        model = self.AmpSensitiveProfileModel()
        profiles = torch.tensor([[1.0, 2.0, 3.0, 4.0]]).expand(2, -1)
        loader = [
            (
                torch.zeros(2, 3, 4),
                profiles,
                torch.zeros(2, 1),
                profiles,
                torch.zeros(2, 1),
                torch.zeros(2, dtype=torch.long),
            )
        ]
        base, tuned, _ = fine_tune._score_ordinary_pair(
            model, model, loader, torch.device("cpu"), amp=True
        )
        self.assertEqual(model.autocast_states, [False, False])
        self.assertGreater(base["nuc_profile_pearson"], 0.99)
        self.assertTrue(math.isfinite(tuned["nuc_profile_pearson"]))

    def test_forward_helper_disables_enclosing_autocast(self):
        model = self.AmpSensitiveProfileModel()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            output = fine_tune._forward_model_float32(
                model, torch.zeros(1, 3, 4), torch.zeros(1, dtype=torch.long)
            )
        self.assertEqual(model.autocast_states, [False])
        self.assertEqual(output[2].dtype, torch.float32)

    def test_float32_profile_logits_keep_multinomial_loss_finite_and_sensitive(self):
        model = self.AmpSensitiveProfileModel()
        counts = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        with torch.autocast("cpu", dtype=torch.bfloat16):
            logits = fine_tune._forward_model_float32(
                model, torch.zeros(1, 3, 4), torch.zeros(1, dtype=torch.long)
            )[2]
        matched_loss = multinomial_nll(counts, logits)
        flattened_loss = multinomial_nll(counts, torch.full_like(logits, 10_000.0))
        self.assertTrue(torch.isfinite(matched_loss))
        self.assertTrue(torch.isfinite(flattened_loss))
        self.assertLess(matched_loss, flattened_loss)


class BasePearsonPreflightTests(unittest.TestCase):
    def test_undefined_base_pearson_aborts_before_optimizer_setup(self):
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
                device="cpu",
                steps=1,
                validate_every=1,
            )
            checkpoint = {"cell_types": ["adult_0"]}
            prepared = SimpleNamespace(
                checkpoint=checkpoint,
                checkpoint_index={"adult_0": 0},
                model=Mock(),
                valid_replay=Mock(),
            )
            with (
                patch(
                    "chrombpnet.multitask.fine_tune_differential_torch._prepare_run",
                    return_value=prepared,
                ),
                patch(
                    "chrombpnet.multitask.fine_tune_differential_torch.load_hybrid_checkpoint",
                    return_value=(Mock(), checkpoint),
                ),
                patch(
                    "chrombpnet.multitask.fine_tune_differential_torch._score_ordinary_pair",
                    return_value=(
                        {"nuc_profile_pearson": float("nan")},
                        {"nuc_profile_pearson": float("nan")},
                        True,
                    ),
                ),
                patch(
                    "chrombpnet.multitask.fine_tune_differential_torch.configure_adult_profile_finetune",
                    side_effect=RuntimeError("optimizer reached"),
                ) as configure,
                patch(
                    "chrombpnet.multitask.fine_tune_differential_torch._score_differential"
                ) as score_differential,
                self.assertRaisesRegex(ValueError, "base ordinary validation.*Pearson"),
            ):
                run_single_finetune(config)
            configure.assert_not_called()
            score_differential.assert_not_called()


class PreservationGateTests(unittest.TestCase):
    def setUp(self):
        counts = {
            name: {"expected": 4, "eligible": 4, "scored": 4}
            for name in (
                "deviation_pearson",
                "effect_spearman",
                "shift_direction_accuracy",
                "shift_mae_bp",
            )
        }
        counts["average_precision"] = {
            "expected": 4,
            "eligible": 4,
            "scored": 4,
            "positives": 1,
        }
        self.base = {
            "nuc_profile_pearson": 0.50,
            "nuc_profile_nll": 100.0,
            "shift_direction_accuracy": 0.75,
            "effect_spearman": 0.5,
            "shift_mae_bp": 4.0,
            "average_precision": 0.6,
            "deviation_pearson": 0.4,
            "counts": counts,
            "coverage_complete": True,
        }

    def test_accepts_metrics_at_preservation_boundaries(self):
        tuned = {
            "nuc_profile_pearson": 0.49,
            "nuc_profile_nll": 102.0,
            "shift_direction_accuracy": 0.75,
            "effect_spearman": 0.5,
            "shift_mae_bp": 4.0,
            "average_precision": 0.6,
            "deviation_pearson": 0.4,
            "counts": self.base["counts"],
            "coverage_complete": True,
            "accessibility_unchanged": True,
            "non_adult_unchanged": True,
            "profile_bias_unchanged": True,
        }
        self.assertTrue(passes_gates(self.base, tuned))

    def test_rejects_profile_drop_over_point_zero_one(self):
        tuned = dict(
            self.base,
            nuc_profile_pearson=0.489,
            accessibility_unchanged=True,
            non_adult_unchanged=True,
            profile_bias_unchanged=True,
        )
        self.assertFalse(passes_gates(self.base, tuned))

    def test_rejects_nll_increase_over_two_percent(self):
        tuned = dict(
            self.base,
            nuc_profile_nll=102.01,
            accessibility_unchanged=True,
            non_adult_unchanged=True,
            profile_bias_unchanged=True,
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
                    profile_bias_unchanged=True,
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
                    profile_bias_unchanged=True,
                ),
            )
        )

    def test_rejects_every_missing_undefined_or_nonfinite_gate_input(self):
        tuned = {
            **self.base,
            "accessibility_unchanged": True,
            "non_adult_unchanged": True,
            "profile_bias_unchanged": True,
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
        for key in (
            "accessibility_unchanged",
            "non_adult_unchanged",
            "profile_bias_unchanged",
        ):
            for undefined in ("missing", None, float("nan"), 1):
                with self.subTest(key=key, value=undefined):
                    candidate = dict(tuned)
                    if undefined == "missing":
                        candidate.pop(key)
                    else:
                        candidate[key] = undefined
                    self.assertFalse(passes_gates(self.base, candidate))

    def test_rejects_missing_incomplete_or_mismatched_endpoint_coverage(self):
        tuned = {
            **self.base,
            "accessibility_unchanged": True,
            "non_adult_unchanged": True,
            "profile_bias_unchanged": True,
        }
        for side, mutation in (
            ("base", {"coverage_complete": False}),
            ("tuned", {"coverage_complete": False}),
            ("base", {"counts": None}),
            ("tuned", {"counts": None}),
        ):
            with self.subTest(side=side, mutation=mutation):
                candidate_base = dict(self.base)
                candidate_tuned = dict(tuned)
                (candidate_base if side == "base" else candidate_tuned).update(mutation)
                self.assertFalse(passes_gates(candidate_base, candidate_tuned))

        mismatched = dict(tuned)
        mismatched_counts = {
            name: dict(values) for name, values in self.base["counts"].items()
        }
        mismatched_counts["shift_direction_accuracy"]["scored"] = 3
        mismatched["counts"] = mismatched_counts
        self.assertFalse(passes_gates(self.base, mismatched))

    def test_requires_all_ap_rows_but_allows_shift_eligibility_subset(self):
        base = copy.deepcopy(self.base)
        tuned = copy.deepcopy(self.base)
        tuned.update(
            accessibility_unchanged=True,
            non_adult_unchanged=True,
            profile_bias_unchanged=True,
        )
        for metrics in (base, tuned):
            for endpoint in ("shift_direction_accuracy", "shift_mae_bp"):
                metrics["counts"][endpoint].update(eligible=2, scored=2)

        self.assertTrue(passes_gates(base, tuned))

        for metrics in (base, tuned):
            metrics["counts"]["average_precision"].update(eligible=3, scored=3)
        self.assertFalse(passes_gates(base, tuned))

    def test_load_selected_lambda_requires_a_passing_selection(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint = Path(temporary_directory) / "base.pt"
            checkpoint.write_bytes(b"base checkpoint")
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            path = Path(temporary_directory) / "selected.json"
            selection = {
                "schema_version": 2,
                "analysis": "differential_nucleosome_pilot_selection",
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
                "schema_version": 2,
                "analysis": "differential_nucleosome_pilot_selection",
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
                    "chrombpnet.multitask.fine_tune_differential_torch.load_passing_validation_report",
                    return_value={},
                ),
                patch(
                    "chrombpnet.multitask.fine_tune_differential_torch.load_hybrid_checkpoint"
                ) as load_checkpoint,
                self.assertRaisesRegex(ValueError, "fold role 'train'"),
            ):
                evaluate_checkpoints(
                    base_checkpoint_path=root / "base.pt",
                    tuned_checkpoint_path=root / "tuned.pt",
                    validation_report_path=root / "validation.json",
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

    def test_aggregate_metrics_pool_all_endpoint_rows_once(self):
        metrics = [
            DifferentialMetrics.from_endpoint_inputs(
                DifferentialEndpointInputs(
                    deviation_pearsons=(0.1, 0.3),
                    deviation_eligible=(True, True),
                    predicted_effects=(1.0, 2.0),
                    observed_effects=(1.0, 2.0),
                    predicted_shifts=(12.0, -99.0),
                    observed_shifts=(12.0, 5.0),
                    predicted_robust_z=(0.2, 0.1),
                    observed_calls=(True, False),
                )
            ),
            DifferentialMetrics.from_endpoint_inputs(
                DifferentialEndpointInputs(
                    deviation_pearsons=(0.5, 0.7, 0.9, 1.0),
                    deviation_eligible=(True, True, True, True),
                    predicted_effects=(6.0, 5.0, 4.0, 3.0),
                    observed_effects=(3.0, 4.0, 5.0, 6.0),
                    predicted_shifts=(12.0, 0.0, 3.0, 4.0),
                    observed_shifts=(-12.0, 0.0, float("nan"), 8.0),
                    predicted_robust_z=(0.9, 0.8, 0.7, 0.6),
                    observed_calls=(False, False, False, True),
                )
            ),
        ]

        summary = aggregate_differential_metrics(metrics)

        self.assertAlmostEqual(summary["deviation_pearson"], 0.5833333333333334)
        self.assertAlmostEqual(summary["effect_spearman"], 0.4285714285714286)
        self.assertAlmostEqual(summary["shift_direction_accuracy"], 0.5)
        self.assertAlmostEqual(summary["shift_mae_bp"], 12.0)
        self.assertAlmostEqual(summary["average_precision"], 0.325)
        self.assertEqual(
            summary["counts"]["effect_spearman"],
            {"expected": 6, "eligible": 6, "scored": 6},
        )
        self.assertEqual(
            summary["counts"]["shift_direction_accuracy"],
            {"expected": 6, "eligible": 2, "scored": 2},
        )
        self.assertEqual(
            summary["counts"]["shift_mae_bp"],
            {"expected": 6, "eligible": 2, "scored": 2},
        )
        self.assertTrue(summary["coverage_complete"])

    def test_aggregate_metrics_records_partial_prediction_coverage(self):
        metrics = [
            DifferentialMetrics.from_endpoint_inputs(
                DifferentialEndpointInputs(
                    deviation_pearsons=(0.2, float("nan"), 0.4, 0.6),
                    deviation_eligible=(True, True, True, True),
                    predicted_effects=(1.0, float("nan"), 3.0, 4.0),
                    observed_effects=(1.0, 2.0, 3.0, 4.0),
                    predicted_shifts=(12.0, float("nan"), 99.0, 0.0),
                    observed_shifts=(12.0, -15.0, 5.0, float("nan")),
                    predicted_robust_z=(1.0, float("nan"), 0.5, 0.0),
                    observed_calls=(True, False, False, False),
                )
            )
        ]

        summary = aggregate_differential_metrics(metrics)

        self.assertFalse(summary["coverage_complete"])
        self.assertEqual(
            summary["counts"]["deviation_pearson"],
            {"expected": 4, "eligible": 4, "scored": 3},
        )
        self.assertEqual(
            summary["counts"]["effect_spearman"],
            {"expected": 4, "eligible": 4, "scored": 3},
        )
        self.assertEqual(
            summary["counts"]["shift_direction_accuracy"],
            {"expected": 4, "eligible": 2, "scored": 1},
        )
        self.assertEqual(
            summary["counts"]["average_precision"],
            {"expected": 4, "eligible": 4, "scored": 3, "positives": 1},
        )

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
                candidate.out_dir.mkdir(parents=True, exist_ok=True)
                report_path = candidate.out_dir / f"{candidate.output_name}.json"
                report_path.write_text(
                    json.dumps(
                        {
                            "validation": {"deviation_pearson": score},
                            "comparison": {"candidate_lambda": candidate.lambda_dev},
                            "gate_checks": {"synthetic_pass": passing},
                        }
                    )
                    + "\n"
                )
                checkpoint_path = candidate.out_dir / f"{candidate.output_name}.pt"
                if passing:
                    checkpoint_path.write_bytes(b"checkpoint")
                return RunResult(
                    lambda_dev=candidate.lambda_dev,
                    selected_step=250,
                    checkpoint_path=(checkpoint_path if passing else None),
                    report_path=report_path,
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
            self.assertEqual(selection["selected_candidate_index"], 2)
            self.assertEqual(
                selection["checkpoint_sha256"],
                hashlib.sha256(b"checkpoint").hexdigest(),
            )
            selected_candidate = selection["candidates"][2]
            self.assertEqual(
                selected_candidate["gate_checks"], {"synthetic_pass": True}
            )
            self.assertEqual(selection["selected_report"], selected_candidate["report"])
            self.assertEqual(
                selection["selected_report_sha256"],
                selected_candidate["report_sha256"],
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


class ValidationReportBindingTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.base = self.root / "base.pt"
        self.tuned = self.root / "tuned.pt"
        self.pilot_checkpoint = self.root / "pilot.pt"
        self.pilot_report = self.root / "pilot_validation.json"
        self.selection = self.root / "selected_pilot.json"
        self.report = self.root / "final_validation.json"
        self.base.write_bytes(b"base checkpoint")
        self.tuned.write_bytes(b"tuned checkpoint")
        self.pilot_checkpoint.write_bytes(b"pilot checkpoint")
        self._write_pilot_evidence()

    def tearDown(self):
        self.temporary_directory.cleanup()

    @staticmethod
    def _counts() -> dict[str, dict[str, int]]:
        counts = {
            name: {"expected": 4, "eligible": 4, "scored": 4}
            for name in (
                "deviation_pearson",
                "effect_spearman",
                "shift_direction_accuracy",
                "shift_mae_bp",
            )
        }
        counts["average_precision"] = {
            "expected": 4,
            "eligible": 4,
            "scored": 4,
            "positives": 1,
        }
        return counts

    def _differential(self) -> dict[str, object]:
        return {
            "deviation_pearson": 0.4,
            "effect_spearman": 0.5,
            "shift_direction_accuracy": 0.75,
            "shift_mae_bp": 4.0,
            "average_precision": 0.6,
            "target_locus_rows": 4,
            "context_units": 1,
            "counts": self._counts(),
            "coverage_complete": True,
        }

    @staticmethod
    def _ordinary() -> dict[str, float]:
        return {
            "nuc_profile_pearson": 0.5,
            "nuc_profile_nll": 100.0,
            "nuc_count_mse": 1.0,
            "acc_profile_pearson": 0.6,
            "acc_profile_nll": 80.0,
            "acc_count_mse": 2.0,
        }

    @staticmethod
    def _gate_checks() -> dict[str, bool]:
        return {
            "differential_metrics_finite": True,
            "differential_coverage_matches": True,
            "nuc_profile_pearson_preserved": True,
            "nuc_profile_nll_preserved": True,
            "shift_direction_preserved": True,
            "accessibility_unchanged": True,
            "non_adult_unchanged": True,
            "profile_bias_unchanged": True,
        }

    def _evidence(
        self,
        *,
        step: int,
        lambda_dev: float,
        protocol: dict[str, object],
    ) -> dict[str, object]:
        differential = self._differential()
        ordinary = self._ordinary()
        validation = {
            **differential,
            **ordinary,
            "accessibility_unchanged": True,
            "non_adult_unchanged": True,
            "profile_bias_unchanged": True,
        }
        comparison = {
            "base": {
                "differential": copy.deepcopy(differential),
                "ordinary": dict(ordinary),
            },
            "tuned": {
                "differential": copy.deepcopy(differential),
                "ordinary": dict(ordinary),
            },
        }
        gate_checks = self._gate_checks()
        history = {
            "step": step,
            "lambda_dev": lambda_dev,
            "run_protocol": copy.deepcopy(protocol),
            "validation": copy.deepcopy(validation),
            "comparison": copy.deepcopy(comparison),
            "gate_checks": dict(gate_checks),
            "passes_gates": True,
        }
        return {
            "validation": validation,
            "comparison": comparison,
            "gate_checks": gate_checks,
            "validation_history": [history],
        }

    def _candidate_report_payload(self) -> dict[str, object]:
        protocol = {"kind": "pilot_sweep_candidate"}
        return {
            "schema_version": 2,
            "analysis": "differential_nucleosome_validation",
            "split": "validation",
            "status": "passing",
            "passes_gates": True,
            "base_checkpoint": str(self.base.resolve()),
            "base_checkpoint_sha256": hashlib.sha256(
                self.base.read_bytes()
            ).hexdigest(),
            "selected_checkpoint": str(self.pilot_checkpoint.resolve()),
            "selected_checkpoint_sha256": hashlib.sha256(
                self.pilot_checkpoint.read_bytes()
            ).hexdigest(),
            "selected_step": 2_000,
            "lambda_dev": 4.0,
            "run_protocol": protocol,
            **self._evidence(step=2_000, lambda_dev=4.0, protocol=protocol),
        }

    def _selection_payload(self) -> dict[str, object]:
        candidate_report_hash = hashlib.sha256(
            self.pilot_report.read_bytes()
        ).hexdigest()
        pilot_checkpoint_hash = hashlib.sha256(
            self.pilot_checkpoint.read_bytes()
        ).hexdigest()
        candidate_report = json.loads(self.pilot_report.read_text())
        candidate = {
            "lambda_dev": 4.0,
            "passes_gates": True,
            "selected_step": 2_000,
            "checkpoint": str(self.pilot_checkpoint.resolve()),
            "checkpoint_sha256": pilot_checkpoint_hash,
            "report": str(self.pilot_report.resolve()),
            "report_sha256": candidate_report_hash,
            "validation": candidate_report["validation"],
            "comparison": candidate_report["comparison"],
            "gate_checks": candidate_report["gate_checks"],
        }
        return {
            "schema_version": 2,
            "analysis": "differential_nucleosome_pilot_selection",
            "passes_gates": True,
            "lambda_dev": 4.0,
            "selected_step": 2_000,
            "selected_candidate_index": 0,
            "checkpoint": str(self.pilot_checkpoint.resolve()),
            "checkpoint_sha256": pilot_checkpoint_hash,
            "selected_report": str(self.pilot_report.resolve()),
            "selected_report_sha256": candidate_report_hash,
            "base_checkpoint": str(self.base.resolve()),
            "base_checkpoint_sha256": hashlib.sha256(
                self.base.read_bytes()
            ).hexdigest(),
            "candidates": [candidate],
        }

    def _write_pilot_evidence(self) -> None:
        self.pilot_report.write_text(
            json.dumps(self._candidate_report_payload()) + "\n"
        )
        self.selection.write_text(json.dumps(self._selection_payload()) + "\n")

    def _payload(self) -> dict[str, object]:
        protocol = {
            "kind": "selected_pilot_restart",
            "selected_pilot_json": str(self.selection.resolve()),
            "selected_pilot_json_sha256": hashlib.sha256(
                self.selection.read_bytes()
            ).hexdigest(),
        }
        return {
            "schema_version": 2,
            "analysis": "differential_nucleosome_validation",
            "split": "validation",
            "status": "passing",
            "passes_gates": True,
            "base_checkpoint": str(self.base.resolve()),
            "base_checkpoint_sha256": hashlib.sha256(
                self.base.read_bytes()
            ).hexdigest(),
            "selected_checkpoint": str(self.tuned.resolve()),
            "selected_checkpoint_sha256": hashlib.sha256(
                self.tuned.read_bytes()
            ).hexdigest(),
            "selected_step": 9_500,
            "lambda_dev": 4.0,
            "run_protocol": protocol,
            **self._evidence(step=9_500, lambda_dev=4.0, protocol=protocol),
        }

    def _assert_rejected_before_data(
        self, payload: dict[str, object], message: str
    ) -> None:
        self.report.write_text(json.dumps(payload) + "\n")
        with (
            patch(
                "chrombpnet.multitask.fine_tune_differential_torch.load_hybrid_checkpoint"
            ) as load_checkpoint,
            patch(
                "chrombpnet.multitask.fine_tune_differential_torch.read_manifest"
            ) as load_manifest,
            patch(
                "chrombpnet.multitask.fine_tune_differential_torch.pl.read_parquet"
            ) as load_calls,
            patch(
                "chrombpnet.multitask.fine_tune_differential_torch.PairedContextDataset"
            ) as open_context_tracks,
            patch(
                "chrombpnet.multitask.fine_tune_differential_torch.build_adult_replay_loader"
            ) as open_replay_tracks,
            self.assertRaisesRegex((TypeError, ValueError), message),
        ):
            self._evaluate()
        load_checkpoint.assert_not_called()
        load_manifest.assert_not_called()
        load_calls.assert_not_called()
        open_context_tracks.assert_not_called()
        open_replay_tracks.assert_not_called()

    def _evaluate(self):
        return evaluate_checkpoints(
            base_checkpoint_path=self.base,
            tuned_checkpoint_path=self.tuned,
            validation_report_path=self.report,
            manifest_path=self.root / "manifest.tsv",
            manifest_root=None,
            metadata_path=self.root / "metadata.tsv",
            fold_json=self.root / "fold.json",
            calls_path=self.root / "calls.parquet",
            genome=self.root / "genome.fa",
            out_json=self.root / "evaluation.json",
            device_name="cpu",
            amp=False,
        )

    def test_complete_hash_bound_recomputed_evidence_is_accepted(self):
        self.report.write_text(json.dumps(self._payload()) + "\n")

        loaded = fine_tune.load_passing_validation_report(
            self.report,
            base_checkpoint_path=self.base,
            tuned_checkpoint_path=self.tuned,
        )

        self.assertEqual(loaded["selected_step"], 9_500)
        self.assertEqual(loaded["gate_checks"], self._gate_checks())

    def test_missing_malformed_or_failing_report_aborts_before_any_data_load(self):
        cases = (
            ("missing", None, "validation report"),
            ("malformed", "{not json", "validation report"),
            (
                "failing",
                json.dumps({**self._payload(), "passes_gates": False}),
                "passing",
            ),
        )
        for label, contents, message in cases:
            with self.subTest(label=label):
                self.report.unlink(missing_ok=True)
                if contents is not None:
                    self.report.write_text(contents)
                with (
                    patch(
                        "chrombpnet.multitask.fine_tune_differential_torch.load_hybrid_checkpoint"
                    ) as load_checkpoint,
                    patch(
                        "chrombpnet.multitask.fine_tune_differential_torch.read_manifest"
                    ) as load_manifest,
                    patch(
                        "chrombpnet.multitask.fine_tune_differential_torch.pl.read_parquet"
                    ) as load_calls,
                    self.assertRaisesRegex((FileNotFoundError, ValueError), message),
                ):
                    self._evaluate()
                load_checkpoint.assert_not_called()
                load_manifest.assert_not_called()
                load_calls.assert_not_called()

    def test_mismatched_or_incomplete_binding_aborts_before_any_data_load(self):
        mutations = (
            ("schema", {"schema_version": 1}, "schema"),
            ("split", {"split": "test"}, "split"),
            ("base_path", {"base_checkpoint": "/wrong/base.pt"}, "base checkpoint"),
            ("base_hash", {"base_checkpoint_sha256": "wrong"}, "base checkpoint"),
            ("tuned_path", {"selected_checkpoint": "/wrong/tuned.pt"}, "checkpoint"),
            ("tuned_hash", {"selected_checkpoint_sha256": "wrong"}, "checkpoint"),
            ("step", {"selected_step": None}, "selected step"),
            ("lambda", {"lambda_dev": 2.0}, "lambda"),
            ("protocol", {"run_protocol": None}, "protocol"),
        )
        for label, mutation, message in mutations:
            with self.subTest(label=label):
                self.report.write_text(json.dumps({**self._payload(), **mutation}))
                with (
                    patch(
                        "chrombpnet.multitask.fine_tune_differential_torch.load_hybrid_checkpoint"
                    ) as load_checkpoint,
                    patch(
                        "chrombpnet.multitask.fine_tune_differential_torch.read_manifest"
                    ) as load_manifest,
                    patch(
                        "chrombpnet.multitask.fine_tune_differential_torch.pl.read_parquet"
                    ) as load_calls,
                    self.assertRaisesRegex((TypeError, ValueError), message),
                ):
                    self._evaluate()
                load_checkpoint.assert_not_called()
                load_manifest.assert_not_called()
                load_calls.assert_not_called()

    def test_metadata_only_or_missing_evidence_is_rejected_before_data_load(self):
        complete = self._payload()
        metadata_only = {
            key: value
            for key, value in complete.items()
            if key
            not in {
                "validation",
                "comparison",
                "gate_checks",
                "validation_history",
            }
        }
        missing_history = copy.deepcopy(complete)
        missing_history.pop("validation_history")
        missing_metrics = copy.deepcopy(complete)
        missing_metrics["validation_history"][0]["comparison"]["base"][
            "differential"
        ].pop("effect_spearman")
        missing_counts = copy.deepcopy(complete)
        missing_counts["validation_history"][0]["comparison"]["tuned"]["differential"][
            "counts"
        ].pop("shift_mae_bp")
        for label, payload in (
            ("metadata_only", metadata_only),
            ("missing_history", missing_history),
            ("missing_metrics", missing_metrics),
            ("missing_counts", missing_counts),
        ):
            with self.subTest(label=label):
                self._assert_rejected_before_data(
                    payload, "evidence|history|comparison"
                )

    def test_selected_step_must_have_exactly_one_complete_history_record(self):
        duplicate = self._payload()
        duplicate["validation_history"].append(
            copy.deepcopy(duplicate["validation_history"][0])
        )
        missing = self._payload()
        missing["validation_history"][0]["step"] = 9_250
        for label, payload in (("duplicate", duplicate), ("missing", missing)):
            with self.subTest(label=label):
                self._assert_rejected_before_data(payload, "selected step|history")

    def test_asserted_gate_status_must_equal_recomputed_gates(self):
        stored_gate_mismatch = self._payload()
        stored_gate_mismatch["validation_history"][0]["gate_checks"][
            "shift_direction_preserved"
        ] = False
        aggregate_mismatch = self._payload()
        aggregate_mismatch["validation_history"][0]["passes_gates"] = False
        for label, payload in (
            ("stored_gate", stored_gate_mismatch),
            ("aggregate", aggregate_mismatch),
        ):
            with self.subTest(label=label):
                self._assert_rejected_before_data(payload, "gate")

    def test_top_level_metrics_and_counts_must_match_selected_history(self):
        metric_mismatch = self._payload()
        for metrics in (
            metric_mismatch["validation"],
            metric_mismatch["comparison"]["tuned"]["differential"],
        ):
            metrics["effect_spearman"] = 0.1
        count_mismatch = self._payload()
        for metrics in (
            count_mismatch["validation"],
            count_mismatch["comparison"]["base"]["differential"],
            count_mismatch["comparison"]["tuned"]["differential"],
        ):
            metrics["counts"]["average_precision"]["positives"] = 2
        for label, payload in (
            ("metric", metric_mismatch),
            ("count", count_mismatch),
        ):
            with self.subTest(label=label):
                self._assert_rejected_before_data(payload, "selected.*match|counts")

    def test_ap_eligibility_below_expected_is_rejected_before_data_load(self):
        payload = self._payload()
        for evidence in (payload, payload["validation_history"][0]):
            for metrics in (
                evidence["validation"],
                evidence["comparison"]["base"]["differential"],
                evidence["comparison"]["tuned"]["differential"],
            ):
                metrics["counts"]["average_precision"].update(eligible=3, scored=3)

        self._assert_rejected_before_data(
            payload, "average_precision.*eligible|AP.*eligible"
        )

    def test_forged_selected_pilot_candidate_is_rejected_before_data_load(self):
        forged_report = self._candidate_report_payload()
        forged_report["validation_history"][0]["gate_checks"][
            "shift_direction_preserved"
        ] = False
        self.pilot_report.write_text(json.dumps(forged_report) + "\n")
        forged_selection = self._selection_payload()
        self.selection.write_text(json.dumps(forged_selection) + "\n")
        self._assert_rejected_before_data(self._payload(), "pilot.*gate|candidate")

    def test_selected_lambda_must_match_final_report_and_candidate(self):
        mismatched_selection = self._selection_payload()
        mismatched_selection["lambda_dev"] = 1.0
        self.selection.write_text(json.dumps(mismatched_selection) + "\n")
        self._assert_rejected_before_data(self._payload(), "pilot.*lambda")

        mismatched_candidate = self._selection_payload()
        mismatched_candidate["candidates"][0]["lambda_dev"] = 1.0
        self.selection.write_text(json.dumps(mismatched_candidate) + "\n")
        self._assert_rejected_before_data(self._payload(), "pilot.*lambda|candidate")


class CleanupAndWandbTests(unittest.TestCase):
    def _call_evaluation(
        self,
        root: Path,
        context_dataset,
        *,
        replay_loader=None,
        replay_error: Exception | None = None,
        score_error: Exception | None = None,
        replay_close=None,
    ):
        fold = {
            "train": sorted(APPROVED_TRAIN_CHROMOSOMES),
            "valid": sorted(APPROVED_VALID_CHROMOSOMES),
            "test": sorted(APPROVED_TEST_CHROMOSOMES),
        }
        fold_path = root / "fold.json"
        fold_path.write_text(json.dumps(fold))
        checkpoint = {
            "cell_types": ["adult_a"],
            "args": {"inputlen": 600, "outputlen": 502},
        }
        tuned_checkpoint = {
            **checkpoint,
            "base_checkpoint_sha256": "base-hash",
            "selected_step": 250,
            "lambda_dev": 1.0,
            "run_protocol": {"kind": "selected_pilot_restart"},
        }
        adult_manifest = pl.DataFrame(
            {"cell_type": ["adult_a"], "nuc_bw": ["adult_a.bw"]}
        )
        replay_kwargs = (
            {"side_effect": replay_error}
            if replay_error is not None
            else {"return_value": replay_loader}
        )
        with ExitStack() as stack:
            stack.enter_context(
                patch(
                    "chrombpnet.multitask.fine_tune_differential_torch.load_passing_validation_report",
                    return_value={
                        "selected_step": 250,
                        "lambda_dev": 1.0,
                        "run_protocol": {"kind": "selected_pilot_restart"},
                    },
                )
            )
            stack.enter_context(
                patch(
                    "chrombpnet.multitask.fine_tune_differential_torch.load_hybrid_checkpoint",
                    side_effect=[
                        (Mock(), checkpoint),
                        (Mock(), tuned_checkpoint),
                    ],
                )
            )
            stack.enter_context(
                patch(
                    "chrombpnet.multitask.fine_tune_differential_torch._sha256",
                    return_value="base-hash",
                )
            )
            stack.enter_context(
                patch(
                    "chrombpnet.multitask.fine_tune_differential_torch.read_manifest",
                    return_value=Mock(),
                )
            )
            stack.enter_context(
                patch(
                    "chrombpnet.multitask.fine_tune_differential_torch.read_metadata",
                    return_value=Mock(),
                )
            )
            stack.enter_context(
                patch(
                    "chrombpnet.multitask.fine_tune_differential_torch.validate_adult_cohort",
                    return_value=adult_manifest,
                )
            )
            stack.enter_context(
                patch(
                    "chrombpnet.multitask.fine_tune_differential_torch.assert_production_cardinalities"
                )
            )
            stack.enter_context(
                patch(
                    "chrombpnet.multitask.fine_tune_differential_torch.validate_adult_manifest_coverage"
                )
            )
            stack.enter_context(
                patch(
                    "chrombpnet.multitask.fine_tune_differential_torch.map_adult_checkpoint_indices",
                    return_value={"adult_a": 0},
                )
            )
            stack.enter_context(
                patch(
                    "chrombpnet.multitask.fine_tune_differential_torch.pl.read_parquet",
                    return_value=pl.DataFrame({"chrom": ["chr3"]}),
                )
            )
            stack.enter_context(
                patch(
                    "chrombpnet.multitask.fine_tune_differential_torch.validate_artifact_chromosomes"
                )
            )
            stack.enter_context(
                patch(
                    "chrombpnet.multitask.fine_tune_differential_torch.build_context_units",
                    return_value=[Mock()],
                )
            )
            stack.enter_context(
                patch(
                    "chrombpnet.multitask.fine_tune_differential_torch.PairedContextDataset",
                    return_value=context_dataset,
                )
            )
            stack.enter_context(
                patch(
                    "chrombpnet.multitask.fine_tune_differential_torch.build_adult_replay_loader",
                    **replay_kwargs,
                )
            )
            if score_error is not None:
                stack.enter_context(
                    patch(
                        "chrombpnet.multitask.fine_tune_differential_torch._score_differential",
                        side_effect=score_error,
                    )
                )
            if replay_close is not None:
                stack.enter_context(
                    patch(
                        "chrombpnet.multitask.fine_tune_differential_torch._close_replay_loader",
                        side_effect=replay_close,
                    )
                )
            return evaluate_checkpoints(
                base_checkpoint_path=root / "base.pt",
                tuned_checkpoint_path=root / "tuned.pt",
                validation_report_path=root / "validation.json",
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

    def test_replay_cleanup_attempts_every_handle_after_first_close_failure(self):
        attempted = []

        class CloseSpy:
            def __init__(self, label: str, fail: bool = False):
                self.label = label
                self.fail = fail

            def close(self):
                attempted.append(self.label)
                if self.fail:
                    raise RuntimeError(f"{self.label} failed")

        sources = []
        labels = []
        for source_index in range(3):
            handles = {}
            for attribute in ("_genome", "_acc_bw", "_nuc_bw"):
                label = f"{source_index}:{attribute}"
                labels.append(label)
                handles[attribute] = CloseSpy(label, fail=not sources and not handles)
            sources.append(SimpleNamespace(**handles))
        collection = _DatasetCollection([])
        collection.datasets = [SimpleNamespace(dataset=source) for source in sources]
        loader = SimpleNamespace(dataset=collection)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("error")
            _close_replay_loader(loader)

        self.assertEqual(attempted, labels)
        for source in sources:
            for attribute in ("_genome", "_acc_bw", "_nuc_bw"):
                self.assertIsNone(getattr(source, attribute))
        self.assertTrue(any("replay handle" in str(item.message) for item in caught))

    def test_evaluation_cleans_context_when_replay_setup_fails(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            context_dataset = Mock()
            with self.assertRaisesRegex(RuntimeError, "replay setup failed"):
                self._call_evaluation(
                    Path(temporary_directory),
                    context_dataset,
                    replay_error=RuntimeError("replay setup failed"),
                )
            context_dataset.close.assert_called_once_with()

    def test_evaluation_cleanup_failure_does_not_skip_other_resource(self):
        for failing_resource in ("context", "replay"):
            with (
                self.subTest(failing_resource=failing_resource),
                tempfile.TemporaryDirectory() as temporary_directory,
            ):
                attempted = []

                def close_context(
                    attempted=attempted,
                    failing_resource=failing_resource,
                ):
                    attempted.append("context")
                    if failing_resource == "context":
                        raise RuntimeError("context close failed")

                def close_replay(
                    _loader,
                    attempted=attempted,
                    failing_resource=failing_resource,
                ):
                    attempted.append("replay")
                    if failing_resource == "replay":
                        raise RuntimeError("replay close failed")

                context_dataset = Mock()
                context_dataset.close.side_effect = close_context
                with (
                    warnings.catch_warnings(record=True) as caught,
                    self.assertRaisesRegex(RuntimeError, "score failed"),
                ):
                    warnings.simplefilter("error")
                    self._call_evaluation(
                        Path(temporary_directory),
                        context_dataset,
                        replay_loader=Mock(),
                        score_error=RuntimeError("score failed"),
                        replay_close=close_replay,
                    )

                self.assertCountEqual(attempted, ["context", "replay"])
                self.assertTrue(
                    any("cleanup failed" in str(item.message) for item in caught)
                )

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
                valid_replay=Mock(),
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
                    "chrombpnet.multitask.fine_tune_differential_torch._score_ordinary_pair",
                    return_value=({"nuc_profile_pearson": 0.5}, {}, True),
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
                    valid_replay=Mock(),
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
                        "chrombpnet.multitask.fine_tune_differential_torch._score_ordinary_pair",
                        return_value=({"nuc_profile_pearson": 0.5}, {}, True),
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
        counts = {
            name: {"expected": 4, "eligible": 4, "scored": 4}
            for name in (
                "deviation_pearson",
                "effect_spearman",
                "shift_direction_accuracy",
                "shift_mae_bp",
            )
        }
        counts["average_precision"] = {
            "expected": 4,
            "eligible": 4,
            "scored": 4,
            "positives": 1,
        }
        differential = {
            "deviation_pearson": 0.4,
            "effect_spearman": 0.5,
            "shift_direction_accuracy": 0.75,
            "shift_mae_bp": 4.0,
            "average_precision": 0.6,
            "target_locus_rows": 4,
            "context_units": 1,
            "counts": counts,
            "coverage_complete": True,
        }
        ordinary = {
            "nuc_profile_pearson": 0.5,
            "nuc_profile_nll": 100.0,
            "nuc_count_mse": 1.0,
            "acc_profile_pearson": 0.6,
            "acc_profile_nll": 80.0,
            "acc_count_mse": 2.0,
        }
        validation = {
            **differential,
            **ordinary,
            "accessibility_unchanged": True,
            "non_adult_unchanged": True,
            "profile_bias_unchanged": True,
        }
        comparison = {
            side: {
                "differential": copy.deepcopy(differential),
                "ordinary": dict(ordinary),
            }
            for side in ("base", "tuned")
        }
        stored_gate_checks = {
            "differential_metrics_finite": True,
            "differential_coverage_matches": True,
            "nuc_profile_pearson_preserved": True,
            "nuc_profile_nll_preserved": True,
            "shift_direction_preserved": True,
            "accessibility_unchanged": True,
            "non_adult_unchanged": True,
            "profile_bias_unchanged": True,
        }
        pilot_protocol = {"kind": "pilot_sweep_candidate"}
        pilot_checkpoint = self.root / "pilot.pt"
        pilot_checkpoint.write_bytes((self.root / "base.pt").read_bytes())
        pilot_report = self.root / "pilot_validation.json"
        pilot_report.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "analysis": "differential_nucleosome_validation",
                    "split": "validation",
                    "status": "passing",
                    "passes_gates": True,
                    "base_checkpoint": str((self.root / "base.pt").resolve()),
                    "base_checkpoint_sha256": hashlib.sha256(
                        (self.root / "base.pt").read_bytes()
                    ).hexdigest(),
                    "selected_checkpoint": str(pilot_checkpoint.resolve()),
                    "selected_checkpoint_sha256": hashlib.sha256(
                        pilot_checkpoint.read_bytes()
                    ).hexdigest(),
                    "selected_step": 2,
                    "lambda_dev": 1.0,
                    "run_protocol": pilot_protocol,
                    "validation": validation,
                    "comparison": comparison,
                    "gate_checks": stored_gate_checks,
                    "validation_history": [
                        {
                            "step": 2,
                            "lambda_dev": 1.0,
                            "run_protocol": pilot_protocol,
                            "validation": validation,
                            "comparison": comparison,
                            "gate_checks": stored_gate_checks,
                            "passes_gates": True,
                        }
                    ],
                }
            )
            + "\n"
        )
        pilot_checkpoint_hash = hashlib.sha256(
            pilot_checkpoint.read_bytes()
        ).hexdigest()
        pilot_report_hash = hashlib.sha256(pilot_report.read_bytes()).hexdigest()
        candidate = {
            "lambda_dev": 1.0,
            "passes_gates": True,
            "selected_step": 2,
            "checkpoint": str(pilot_checkpoint.resolve()),
            "checkpoint_sha256": pilot_checkpoint_hash,
            "report": str(pilot_report.resolve()),
            "report_sha256": pilot_report_hash,
            "validation": validation,
            "comparison": comparison,
            "gate_checks": stored_gate_checks,
        }
        selection = {
            "schema_version": 2,
            "analysis": "differential_nucleosome_pilot_selection",
            "passes_gates": True,
            "lambda_dev": 1.0,
            "selected_step": 2,
            "selected_candidate_index": 0,
            "checkpoint": str(pilot_checkpoint.resolve()),
            "checkpoint_sha256": pilot_checkpoint_hash,
            "selected_report": str(pilot_report.resolve()),
            "selected_report_sha256": pilot_report_hash,
            "base_checkpoint": str((self.root / "base.pt").resolve()),
            "base_checkpoint_sha256": hashlib.sha256(
                (self.root / "base.pt").read_bytes()
            ).hexdigest(),
            "candidates": [candidate],
        }
        (self.root / "selected_pilot.json").write_text(json.dumps(selection) + "\n")

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
            run_protocol="selected_pilot_restart",
            selected_pilot_json=self.root / "selected_pilot.json",
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
        selected_record = next(
            entry
            for entry in run_report["validation_history"]
            if entry["step"] == run_report["selected_step"]
        )
        self.assertEqual(selected_record["validation"], run_report["validation"])
        self.assertEqual(selected_record["comparison"], run_report["comparison"])
        self.assertEqual(selected_record["gate_checks"], run_report["gate_checks"])
        self.assertTrue(all(run_report["gate_checks"].values()))
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
            validation_report_path=result.report_path,
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
        self.assertEqual(report["evaluation_kind"], "corrective_re_evaluation")
        self.assertEqual(report["chromosomes"], ["chr3"])
        self.assertEqual(report["base"]["differential"]["context_units"], 1)
        self.assertEqual(report["tuned"]["differential"]["target_locus_rows"], 4)
        self.assertTrue(report["accessibility_unchanged"])
        self.assertTrue(report["non_adult_unchanged"])
        self.assertTrue(report["profile_bias_unchanged"])
        self.assertTrue(report["passes_evaluation_gate"])


if __name__ == "__main__":
    unittest.main()
