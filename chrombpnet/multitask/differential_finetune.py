"""Objectives and parameter guards for adult differential-nucleosome fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from chrombpnet.training.models.multitask_nucleosome_torch import multinomial_nll

_MODEL_GUARD_ATTRIBUTE = "_chrombpnet_differential_finetune_guard"


def smooth_profiles(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Apply zero-padded Gaussian smoothing across the profile axis."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    radius = max(1, round(4.0 * sigma))
    positions = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    kernel = torch.exp(-0.5 * (positions / sigma) ** 2)
    kernel = kernel / kernel.sum()
    return F.conv1d(x.unsqueeze(1), kernel.reshape(1, 1, -1), padding=radius).squeeze(1)


def normalize_profiles(x: torch.Tensor) -> torch.Tensor:
    """Normalize each profile independently, leaving empty profiles empty."""
    return x / x.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def sum_preserving_smooth(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Smooth profiles while retaining their original total observed counts."""
    smoothed = smooth_profiles(x.float(), sigma)
    return smoothed * (
        x.sum(dim=-1, keepdim=True) / smoothed.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    )


def leave_one_out_deviation(probabilities: torch.Tensor) -> torch.Tensor:
    """Return each cell's profile relative to the mean of all other cells."""
    if probabilities.shape[0] < 2:
        raise ValueError("leave-one-out deviation requires at least two profiles")
    return probabilities - (probabilities.sum(dim=0, keepdim=True) - probabilities) / (
        probabilities.shape[0] - 1
    )


def paired_deviation_loss(
    predicted: torch.Tensor,
    observed: torch.Tensor,
    center_slice: slice,
    sigma: float = 20.0,
) -> torch.Tensor:
    """Penalize disagreement between predicted and observed leave-one-out deviations."""
    pred = torch.softmax(predicted, dim=-1)[:, center_slice]
    obs = observed.float()[:, center_slice]
    pred = normalize_profiles(smooth_profiles(pred, sigma))
    obs = normalize_profiles(smooth_profiles(obs, sigma))
    pred_dev = leave_one_out_deviation(pred)
    obs_dev = leave_one_out_deviation(obs)
    pred_centered = pred_dev - pred_dev.mean(-1, keepdim=True)
    obs_centered = obs_dev - obs_dev.mean(-1, keepdim=True)
    denominator = pred_centered.norm(dim=-1) * obs_centered.norm(dim=-1)
    correlation = (pred_centered * obs_centered).sum(-1) / denominator.clamp_min(1e-8)
    return 1.0 - correlation.mean()


@dataclass(frozen=True)
class DifferentialLoss:
    total: torch.Tensor
    profile_nll: torch.Tensor
    deviation: torch.Tensor


def differential_loss(
    logits: torch.Tensor,
    observed_raw: torch.Tensor,
    center_slice: slice,
    *,
    alpha: float,
    lambda_dev: float,
) -> DifferentialLoss:
    """Combine full-output profile likelihood with the central deviation objective."""
    profile_target = sum_preserving_smooth(observed_raw, sigma=6.0)
    profile_nll = multinomial_nll(profile_target, logits)
    deviation = paired_deviation_loss(logits, observed_raw, center_slice, sigma=20.0)
    return DifferentialLoss(
        alpha * profile_nll + lambda_dev * deviation, profile_nll, deviation
    )


@dataclass
class AdultProfileHeadGuard:
    """Protect frozen nucleosome-profile rows during adult-only fine-tuning."""

    profile_w: torch.nn.Parameter
    profile_b: torch.nn.Parameter
    non_adult_indices: torch.Tensor
    frozen_w: torch.Tensor
    frozen_b: torch.Tensor
    gradient_hooks: tuple[torch.utils.hooks.RemovableHandle, ...]
    model: object
    _teardown_complete: bool = False

    def make_optimizer(self, lr: float) -> torch.optim.Optimizer:
        """Create the only permitted optimizer for this fine-tuning phase."""
        return torch.optim.AdamW(
            [self.profile_w],
            lr=lr,
            weight_decay=0.0,
        )

    def assert_unchanged(self) -> None:
        """Raise if an optimizer step changed a frozen profile row."""
        if not torch.equal(
            self.frozen_w, self.profile_w.detach()[self.non_adult_indices]
        ):
            raise AssertionError("frozen nuc_profile_w rows changed")
        if not torch.equal(self.frozen_b, self.profile_b.detach()):
            raise AssertionError("frozen nuc_profile_b changed")

    def teardown(self) -> None:
        """Remove this guard's gradient hooks exactly once."""
        if self._teardown_complete:
            return
        for handle in self.gradient_hooks:
            handle.remove()
        self._teardown_complete = True
        if getattr(self.model, _MODEL_GUARD_ATTRIBUTE, None) is self:
            delattr(self.model, _MODEL_GUARD_ATTRIBUTE)

    def remove(self) -> None:
        """Alias for :meth:`teardown` for hook-handle terminology."""
        self.teardown()


def _row_gradient_mask(row_mask: torch.Tensor):
    def mask(gradient: torch.Tensor) -> torch.Tensor:
        shape = (row_mask.shape[0],) + (1,) * (gradient.ndim - 1)
        return gradient * row_mask.to(
            device=gradient.device, dtype=gradient.dtype
        ).reshape(shape)

    return mask


def configure_adult_profile_finetune(model, adult_indices) -> AdultProfileHeadGuard:
    """Freeze a hybrid checkpoint except for adult nucleosome-profile rows."""
    previous_guard = getattr(model, _MODEL_GUARD_ATTRIBUTE, None)
    if isinstance(previous_guard, AdultProfileHeadGuard):
        previous_guard.teardown()
    profile_w = model.nuc_profile_w
    profile_b = model.nuc_profile_b
    if profile_w.shape[0] != profile_b.shape[0]:
        raise ValueError(
            "nucleosome profile weights and biases must have matching cell rows"
        )

    adult_indices = torch.as_tensor(
        tuple(adult_indices), dtype=torch.long, device=profile_w.device
    )
    if adult_indices.ndim != 1 or adult_indices.numel() == 0:
        raise ValueError("adult_indices must contain at least one row")
    if adult_indices.unique().numel() != adult_indices.numel():
        raise ValueError("adult_indices must be unique")
    if adult_indices.min() < 0 or adult_indices.max() >= profile_w.shape[0]:
        raise ValueError("adult_indices contain an out-of-range row")

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    profile_w.requires_grad_(True)

    adult_mask = torch.zeros(
        profile_w.shape[0], dtype=torch.bool, device=profile_w.device
    )
    adult_mask[adult_indices] = True
    non_adult_indices = torch.arange(profile_w.shape[0], device=profile_w.device)[
        ~adult_mask
    ]
    gradient_hooks = (profile_w.register_hook(_row_gradient_mask(adult_mask)),)
    guard = AdultProfileHeadGuard(
        model=model,
        profile_w=profile_w,
        profile_b=profile_b,
        non_adult_indices=non_adult_indices,
        frozen_w=profile_w.detach()[non_adult_indices].clone(),
        frozen_b=profile_b.detach().clone(),
        gradient_hooks=gradient_hooks,
    )
    setattr(model, _MODEL_GUARD_ATTRIBUTE, guard)
    return guard
