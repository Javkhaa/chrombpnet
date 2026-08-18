"""CPU-safe end-to-end smoke test for differential nucleosome fine-tuning."""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from chrombpnet.multitask.differential_finetune import (
    configure_adult_profile_finetune,
)
from chrombpnet.multitask.differential_finetune_data import (
    ContextBalancedSampler,
    ContextUnit,
    PairedContextExample,
)
from chrombpnet.multitask.differential_finetune_metrics import (
    DifferentialMetrics,
    score_context_predictions,
)
from chrombpnet.multitask.fine_tune_differential_torch import (
    load_hybrid_checkpoint,
    train_one_step,
)
from chrombpnet.training.models.multitask_nucleosome_torch import (
    HybridMultiCellModel,
)
from chrombpnet.training.utils.one_hot import dna_to_one_hot

CELL_TYPES = ("fetal", "adult_target", "adult_b", "adult_c", "adult_d", "line")
ADULT_INDICES = torch.tensor([1, 2, 3, 4])
INPUT_LENGTH = 601
OUTPUT_LENGTH = 503
DEVIATION_SLICE = slice(1, 502)


def _sequence(*, shifted: bool, variant: int) -> torch.Tensor:
    bases = np.array(list("ACGT"), dtype="U1")
    rng = np.random.default_rng(100 + variant % 2)
    sequence = rng.choice(bases, size=INPUT_LENGTH)
    flank_rng = np.random.default_rng(100 + variant)
    sequence[:20] = flank_rng.choice(bases, size=20)
    sequence[-20:] = flank_rng.choice(bases, size=20)
    if shifted:
        sequence[INPUT_LENGTH // 2 - 2 : INPUT_LENGTH // 2 + 3] = list("AAAAA")
    else:
        sequence[INPUT_LENGTH // 2 - 2 : INPUT_LENGTH // 2 + 3] = list("CCCCC")
    return torch.from_numpy(dna_to_one_hot(["".join(sequence)])[0].astype(np.float32))


def _gaussian(center: float) -> torch.Tensor:
    positions = torch.arange(OUTPUT_LENGTH, dtype=torch.float32)
    profile = torch.exp(-0.5 * ((positions - center) / 5.0) ** 2)
    return 100.0 * (profile + 1e-5)


def _paired_example(
    *, shifted: bool, variant: int, locus_id: str
) -> PairedContextExample:
    center = OUTPUT_LENGTH // 2
    profiles = torch.stack(
        (
            _gaussian(center - (12 if shifted else 0)),
            _gaussian(center),
            _gaussian(center),
            _gaussian(center),
        )
    )
    return PairedContextExample(
        sequence=_sequence(shifted=shifted, variant=variant),
        raw_profiles=profiles,
        checkpoint_indices=ADULT_INDICES,
        observed_calls=torch.tensor([shifted, False, False, False]),
        deviation_slice=DEVIATION_SLICE,
        locus_id=locus_id,
        tissue_context="adult_brain",
    )


def _mean_defined(values: list[float]) -> float:
    values_array = np.asarray(values, dtype=np.float64)
    return float(values_array[np.isfinite(values_array)].mean())


def _metrics(
    model: HybridMultiCellModel, examples: list[PairedContextExample]
) -> DifferentialMetrics:
    scored = []
    model.eval()
    with torch.no_grad():
        for example in examples:
            sequence = example.sequence.unsqueeze(0).expand(4, -1, -1)
            logits = model(sequence, example.checkpoint_indices)[2].numpy()
            scored.append(
                score_context_predictions(
                    logits,
                    example.raw_profiles.numpy(),
                    example.observed_calls.numpy(),
                    example.deviation_slice,
                )
            )
    return DifferentialMetrics(
        deviation_pearson=_mean_defined(
            [metric.deviation_pearson for metric in scored]
        ),
        effect_spearman=_mean_defined([metric.effect_spearman for metric in scored]),
        shift_direction_accuracy=_mean_defined(
            [metric.shift_direction_accuracy for metric in scored]
        ),
        shift_mae_bp=_mean_defined([metric.shift_mae_bp for metric in scored]),
        average_precision=_mean_defined(
            [metric.average_precision for metric in scored]
        ),
        target_locus_rows=sum(metric.target_locus_rows for metric in scored),
    )


def _replay(example: PairedContextExample):
    return (
        example.sequence.unsqueeze(0).expand(4, -1, -1),
        torch.zeros(4, OUTPUT_LENGTH),
        torch.zeros(4, 1),
        example.raw_profiles,
        torch.zeros(4, 1),
        example.checkpoint_indices,
    )


class DifferentialFineTuneSmokeTest(unittest.TestCase):
    def test_synthetic_adult_profile_finetuning_learns_without_changing_frozen_state(
        self,
    ):
        torch.manual_seed(0)
        train_examples = [
            _paired_example(shifted=False, variant=0, locus_id="train:plain"),
            _paired_example(shifted=True, variant=1, locus_id="train:shifted"),
        ]
        validation_examples = [
            _paired_example(shifted=False, variant=2, locus_id="valid:plain"),
            _paired_example(shifted=True, variant=3, locus_id="valid:shifted"),
        ]
        units = [
            ContextUnit(
                locus_id=example.locus_id,
                chrom="chr8",
                center=1_000 + index,
                tissue_context=example.tissue_context,
                cell_types=CELL_TYPES[1:5],
                checkpoint_indices=tuple(ADULT_INDICES.tolist()),
                observed_calls=tuple(example.observed_calls.tolist()),
            )
            for index, example in enumerate(train_examples)
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = Path(temporary_directory) / "tiny_hybrid.pt"
            model = HybridMultiCellModel(
                n_cell_types=len(CELL_TYPES),
                inputlen=INPUT_LENGTH,
                outputlen=OUTPUT_LENGTH,
                filters=4,
                n_dil_layers=1,
                embed_dim=2,
                profile_kernel_size=3,
            )
            torch.save(
                {
                    "args": {
                        "hybrid": True,
                        "inputlen": INPUT_LENGTH,
                        "outputlen": OUTPUT_LENGTH,
                        "filters": 4,
                        "n_dil_layers": 1,
                        "embed_dim": 2,
                        "cond_mode": "film",
                        "profile_kernel_size": 3,
                    },
                    "cell_types": list(CELL_TYPES),
                    "model_state_dict": model.state_dict(),
                },
                checkpoint_path,
            )
            tuned_model, _ = load_hybrid_checkpoint(
                checkpoint_path, torch.device("cpu")
            )

        base_model = copy.deepcopy(tuned_model)
        base_metrics = _metrics(base_model, validation_examples)
        with torch.no_grad():
            base_accessibility = base_model(
                validation_examples[1].sequence.unsqueeze(0), torch.tensor([1])
            )[0].clone()
            base_nonadult_rows = base_model.nuc_profile_w[[0, 5]].clone()

        guard = configure_adult_profile_finetune(tuned_model, ADULT_INDICES.tolist())
        optimizer = guard.make_optimizer(lr=1e-2)
        sampler = ContextBalancedSampler(units, num_samples=80, seed=0)
        for index in sampler:
            example = train_examples[index]
            result = train_one_step(
                tuned_model,
                guard,
                optimizer,
                example,
                _replay(example),
                device=torch.device("cpu"),
                alpha=1.0,
                lambda_dev=4.0,
                grad_clip=100.0,
            )
            self.assertTrue(np.isfinite(result.total_loss))
        guard.teardown()

        tuned_metrics = _metrics(tuned_model, validation_examples)
        with torch.no_grad():
            tuned_accessibility = tuned_model(
                validation_examples[1].sequence.unsqueeze(0), torch.tensor([1])
            )[0]
            tuned_nonadult_rows = tuned_model.nuc_profile_w[[0, 5]]

        print(
            "DIFFERENTIAL_FINETUNE_METRICS "
            f"base_deviation_pearson={base_metrics.deviation_pearson:.6f} "
            f"tuned_deviation_pearson={tuned_metrics.deviation_pearson:.6f} "
            f"base_shift_direction_accuracy={base_metrics.shift_direction_accuracy:.6f} "
            f"tuned_shift_direction_accuracy={tuned_metrics.shift_direction_accuracy:.6f}"
        )

        self.assertGreater(
            tuned_metrics.deviation_pearson, base_metrics.deviation_pearson
        )
        self.assertGreaterEqual(
            tuned_metrics.shift_direction_accuracy,
            base_metrics.shift_direction_accuracy,
        )
        torch.testing.assert_close(
            base_accessibility, tuned_accessibility, rtol=0, atol=0
        )
        torch.testing.assert_close(
            base_nonadult_rows, tuned_nonadult_rows, rtol=0, atol=0
        )


if __name__ == "__main__":
    result = unittest.main(exit=False)
    if result.result.wasSuccessful():
        print("DIFFERENTIAL_FINETUNE_SMOKE_OK")
