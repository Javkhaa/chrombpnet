"""PyTorch multi-task ChromBPNet-style model.

Shared dilated convolution trunk with two BPNet-style profile/count heads:
accessibility cut-site signal and nucleosome dyad signal.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ProfileCountHead(nn.Module):
    def __init__(self, filters: int, outputlen: int, profile_kernel_size: int = 75):
        super().__init__()
        self.outputlen = outputlen
        self.profile = nn.Conv1d(filters, 1, kernel_size=profile_kernel_size, padding=0)
        self.count = nn.Linear(filters, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.profile(x).squeeze(1)
        crop = (logits.shape[-1] - self.outputlen) // 2
        if crop < 0 or (logits.shape[-1] - self.outputlen) % 2 != 0:
            raise ValueError(f"Cannot center-crop profile length {logits.shape[-1]} to {self.outputlen}")
        if crop:
            logits = logits[:, crop:-crop]
        pooled = x.mean(dim=-1)
        logcount = self.count(pooled)
        return logits, logcount


class MultiTaskNucleosomeModel(nn.Module):
    def __init__(self, inputlen: int = 2114, outputlen: int = 1000,
                 filters: int = 512, n_dil_layers: int = 8):
        super().__init__()
        self.inputlen = inputlen
        self.outputlen = outputlen
        self.first = nn.Conv1d(4, filters, kernel_size=21, padding=0)
        self.dilated = nn.ModuleList([
            nn.Conv1d(filters, filters, kernel_size=3, dilation=2 ** i, padding=0)
            for i in range(1, n_dil_layers + 1)
        ])
        self.accessibility = ProfileCountHead(filters, outputlen)
        self.nucleosome = ProfileCountHead(filters, outputlen)

    @staticmethod
    def _center_crop(x: torch.Tensor, width: int) -> torch.Tensor:
        diff = x.shape[-1] - width
        if diff < 0 or diff % 2 != 0:
            raise ValueError(f"Cannot center-crop length {x.shape[-1]} to {width}")
        crop = diff // 2
        return x[..., crop:-crop] if crop else x

    def forward(self, seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Input may arrive as NHWC-style one-hot from NumPy: B x L x 4.
        if seq.shape[1] != 4:
            seq = seq.transpose(1, 2)
        x = F.relu(self.first(seq))
        for conv in self.dilated:
            conv_x = F.relu(conv(x))
            x = conv_x + self._center_crop(x, conv_x.shape[-1])
        acc_profile, acc_count = self.accessibility(x)
        nuc_profile, nuc_count = self.nucleosome(x)
        return acc_profile, acc_count, nuc_profile, nuc_count


class MultiCellMultiTaskModel(nn.Module):
    """Shared dilated-conv trunk with one (accessibility, nucleosome) head pair
    per cell type. The trunk is cell-type-agnostic so it can later be reused with
    a conditioning embedding for zero-shot transfer to unseen cell types.

    forward(seq, ct_idx) returns per-sample outputs aligned to the batch, so the
    existing ``multitask_loss`` applies unchanged.
    """

    def __init__(self, n_cell_types: int, inputlen: int = 2114, outputlen: int = 1000,
                 filters: int = 512, n_dil_layers: int = 8):
        super().__init__()
        self.inputlen = inputlen
        self.outputlen = outputlen
        self.n_cell_types = n_cell_types
        self.first = nn.Conv1d(4, filters, kernel_size=21, padding=0)
        self.dilated = nn.ModuleList([
            nn.Conv1d(filters, filters, kernel_size=3, dilation=2 ** i, padding=0)
            for i in range(1, n_dil_layers + 1)
        ])
        self.accessibility = nn.ModuleList([ProfileCountHead(filters, outputlen) for _ in range(n_cell_types)])
        self.nucleosome = nn.ModuleList([ProfileCountHead(filters, outputlen) for _ in range(n_cell_types)])

    @staticmethod
    def _center_crop(x: torch.Tensor, width: int) -> torch.Tensor:
        diff = x.shape[-1] - width
        if diff < 0 or diff % 2 != 0:
            raise ValueError(f"Cannot center-crop length {x.shape[-1]} to {width}")
        crop = diff // 2
        return x[..., crop:-crop] if crop else x

    def trunk(self, seq: torch.Tensor) -> torch.Tensor:
        if seq.shape[1] != 4:
            seq = seq.transpose(1, 2)
        x = F.relu(self.first(seq))
        for conv in self.dilated:
            conv_x = F.relu(conv(x))
            x = conv_x + self._center_crop(x, conv_x.shape[-1])
        return x

    def forward(self, seq: torch.Tensor, ct_idx: torch.Tensor):
        x = self.trunk(seq)                                  # (B, filters, L')
        B = x.shape[0]
        acc_profile = x.new_zeros((B, self.outputlen))
        acc_count = x.new_zeros((B, 1))
        nuc_profile = x.new_zeros((B, self.outputlen))
        nuc_count = x.new_zeros((B, 1))
        # Route each cell type's rows to its own head pair (one trunk pass total).
        for ct in torch.unique(ct_idx):
            m = ct_idx == ct
            xc = x[m]
            ap, ac = self.accessibility[int(ct)](xc)
            npf, nc = self.nucleosome[int(ct)](xc)
            acc_profile[m] = ap
            acc_count[m] = ac
            nuc_profile[m] = npf
            nuc_count[m] = nc
        return acc_profile, acc_count, nuc_profile, nuc_count


def multinomial_nll(true_counts: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """Mean multinomial negative log likelihood for profile-count targets."""
    true_counts = true_counts.float()
    logits = logits.float()
    total = true_counts.sum(dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    log_factorial = torch.lgamma(total + 1) - torch.lgamma(true_counts + 1).sum(dim=-1)
    log_prob = log_factorial + (true_counts * log_probs).sum(dim=-1)
    return -log_prob.mean()


def multitask_loss(outputs, targets, counts_loss_weight=1.0,
                   nucleosome_profile_weight=1.0, nucleosome_counts_weight=1.0,
                   return_components=False):
    acc_profile, acc_count, nuc_profile, nuc_count = outputs
    acc, acc_lc, nuc, nuc_lc = targets
    acc_profile_nll = multinomial_nll(acc, acc_profile)
    acc_count_mse = F.mse_loss(acc_count, acc_lc)
    nuc_profile_nll = multinomial_nll(nuc, nuc_profile)
    nuc_count_mse = F.mse_loss(nuc_count, nuc_lc)
    total = (acc_profile_nll
             + counts_loss_weight * acc_count_mse
             + nucleosome_profile_weight * nuc_profile_nll
             + nucleosome_counts_weight * nuc_count_mse)
    if return_components:
        # raw (unweighted) per-head terms, detached for logging
        comps = {
            'acc_profile_nll': acc_profile_nll.detach(),
            'acc_count_mse': acc_count_mse.detach(),
            'nuc_profile_nll': nuc_profile_nll.detach(),
            'nuc_count_mse': nuc_count_mse.detach(),
        }
        return total, comps
    return total
