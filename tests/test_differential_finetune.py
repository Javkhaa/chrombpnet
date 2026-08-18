from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from chrombpnet.multitask.differential_finetune import (
    configure_adult_profile_finetune,
    leave_one_out_deviation,
    paired_deviation_loss,
)
from chrombpnet.training.models.multitask_nucleosome_torch import HybridMultiCellModel


def shifted_four_cell_profiles(shift: int) -> torch.Tensor:
    positions = torch.arange(201, dtype=torch.float32)
    centers = torch.tensor([100 + shift, 100, 100 - shift, 100], dtype=torch.float32)
    profiles = torch.exp(
        -0.5 * ((positions.unsqueeze(0) - centers.unsqueeze(1)) / 4.0) ** 2
    )
    profiles = profiles + 1e-6
    return profiles / profiles.sum(dim=-1, keepdim=True)


class DifferentialObjectiveTests(unittest.TestCase):
    def test_leave_one_out_deviation(self):
        profiles = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.5, 0.5]])

        deviation = leave_one_out_deviation(profiles)

        expected_first = profiles[0] - profiles[1:].mean(0)
        torch.testing.assert_close(deviation[0], expected_first)

    def test_correct_shift_has_lower_deviation_loss(self):
        observed = shifted_four_cell_profiles(shift=12)
        correct = observed.clone()
        collapsed = observed.mean(0, keepdim=True).repeat(4, 1)

        self.assertLess(
            paired_deviation_loss(
                correct.log(), observed, slice(0, observed.shape[-1])
            ).item(),
            paired_deviation_loss(
                collapsed.log(), observed, slice(0, observed.shape[-1])
            ).item(),
        )


class AdultProfileHeadGuardTests(unittest.TestCase):
    def test_updates_only_adult_profile_rows(self):
        torch.manual_seed(7)
        model = HybridMultiCellModel(
            n_cell_types=6,
            inputlen=128,
            outputlen=20,
            filters=4,
            n_dil_layers=1,
            embed_dim=2,
            profile_kernel_size=3,
        )
        guard = configure_adult_profile_finetune(model, adult_indices=[1, 4])
        before_w = model.nuc_profile_w.detach().clone()
        before_b = model.nuc_profile_b.detach().clone()
        optimizer = guard.make_optimizer(lr=0.01)

        parameter_ids = {
            id(parameter) for parameter in optimizer.param_groups[0]["params"]
        }
        self.assertEqual(
            parameter_ids, {id(model.nuc_profile_w), id(model.nuc_profile_b)}
        )
        self.assertEqual(optimizer.param_groups[0]["weight_decay"], 0.0)
        self.assertTrue(
            all(
                not parameter.requires_grad
                for name, parameter in model.named_parameters()
                if name not in {"nuc_profile_w", "nuc_profile_b"}
            )
        )

        sequence = torch.randn(2, 128, 4)
        _, _, logits, _ = model(sequence, torch.tensor([1, 4]))
        loss = F.cross_entropy(logits, torch.tensor([5, 15]))
        loss.backward()
        torch.testing.assert_close(
            model.nuc_profile_w.grad[[0, 2, 3, 5]],
            torch.zeros_like(model.nuc_profile_w.grad[[0, 2, 3, 5]]),
            rtol=0,
            atol=0,
        )

        optimizer.step()
        guard.assert_unchanged()

        self.assertFalse(torch.equal(before_w[1], model.nuc_profile_w[1]))
        self.assertFalse(torch.equal(before_w[4], model.nuc_profile_w[4]))
        torch.testing.assert_close(
            before_w[[0, 2, 3, 5]], model.nuc_profile_w[[0, 2, 3, 5]], rtol=0, atol=0
        )
        torch.testing.assert_close(
            before_b[[0, 2, 3, 5]], model.nuc_profile_b[[0, 2, 3, 5]], rtol=0, atol=0
        )

    def test_teardown_and_reconfiguration_replace_prior_gradient_masks(self):
        torch.manual_seed(11)
        model = HybridMultiCellModel(
            n_cell_types=6,
            inputlen=128,
            outputlen=20,
            filters=4,
            n_dil_layers=1,
            embed_dim=2,
            profile_kernel_size=3,
        )
        first_guard = configure_adult_profile_finetune(model, adult_indices=[1, 4])
        first_guard.teardown()
        first_guard.teardown()

        configure_adult_profile_finetune(model, adult_indices=[1, 4])
        second_guard = configure_adult_profile_finetune(model, adult_indices=[2, 5])
        sequence = torch.randn(2, 128, 4)
        _, _, logits, _ = model(sequence, torch.tensor([2, 5]))
        F.cross_entropy(logits, torch.tensor([5, 15])).backward()

        self.assertGreater(torch.count_nonzero(model.nuc_profile_w.grad[2]).item(), 0)
        self.assertGreater(torch.count_nonzero(model.nuc_profile_w.grad[5]).item(), 0)
        torch.testing.assert_close(
            model.nuc_profile_w.grad[[1, 4]],
            torch.zeros_like(model.nuc_profile_w.grad[[1, 4]]),
            rtol=0,
            atol=0,
        )
        second_guard.teardown()
        second_guard.teardown()
