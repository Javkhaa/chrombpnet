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
                 filters: int = 512, n_dil_layers: int = 8, profile_kernel_size: int = 75):
        super().__init__()
        self.inputlen = inputlen
        self.outputlen = outputlen
        self.n_cell_types = n_cell_types
        self.filters = filters
        self.pks = profile_kernel_size
        self.first = nn.Conv1d(4, filters, kernel_size=21, padding=0)
        self.dilated = nn.ModuleList([
            nn.Conv1d(filters, filters, kernel_size=3, dilation=2 ** i, padding=0)
            for i in range(1, n_dil_layers + 1)
        ])
        # Per-cell-type heads as STACKED parameters (equivalent to n_cell_types
        # ProfileCountHead modules) so a batch spanning many cell types runs as one
        # grouped conv + a gather -- no Python loop over unique cell types and no
        # int(ct) GPU->CPU syncs (critical when n_cell_types is large).
        self.acc_profile_w = nn.Parameter(torch.empty(n_cell_types, filters, profile_kernel_size))
        self.nuc_profile_w = nn.Parameter(torch.empty(n_cell_types, filters, profile_kernel_size))
        self.acc_profile_b = nn.Parameter(torch.zeros(n_cell_types))
        self.nuc_profile_b = nn.Parameter(torch.zeros(n_cell_types))
        self.acc_count_w = nn.Parameter(torch.empty(n_cell_types, filters))
        self.nuc_count_w = nn.Parameter(torch.empty(n_cell_types, filters))
        self.acc_count_b = nn.Parameter(torch.zeros(n_cell_types))
        self.nuc_count_b = nn.Parameter(torch.zeros(n_cell_types))
        import math
        for w in (self.acc_profile_w, self.nuc_profile_w, self.acc_count_w, self.nuc_count_w):
            nn.init.kaiming_uniform_(w, a=math.sqrt(5))

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
        B, Fdim, L = x.shape
        xr = x.reshape(1, B * Fdim, L)                       # for per-sample grouped conv
        pooled = x.mean(dim=-1)                              # (B, filters)

        def _head(prof_w, prof_b, cnt_w, cnt_b):
            # profile: each sample convolved with ITS cell type's filter via grouped conv
            logits = F.conv1d(xr, prof_w[ct_idx], bias=prof_b[ct_idx], groups=B).reshape(B, -1)
            diff = logits.shape[-1] - self.outputlen
            if diff < 0 or diff % 2 != 0:
                raise ValueError(f"Cannot center-crop profile length {logits.shape[-1]} to {self.outputlen}")
            crop = diff // 2
            if crop:
                logits = logits[:, crop:-crop]
            # count: per-sample linear on the pooled trunk features
            logcount = (pooled * cnt_w[ct_idx]).sum(-1, keepdim=True) + cnt_b[ct_idx].unsqueeze(-1)
            return logits, logcount

        acc_profile, acc_count = _head(self.acc_profile_w, self.acc_profile_b, self.acc_count_w, self.acc_count_b)
        nuc_profile, nuc_count = _head(self.nuc_profile_w, self.nuc_profile_b, self.nuc_count_w, self.nuc_count_b)
        return acc_profile, acc_count, nuc_profile, nuc_count


class ConditionedMultiCellModel(nn.Module):
    """Cell-type CONDITIONED trunk (FiLM on a per-cell-type embedding) with SHARED
    (accessibility, nucleosome) heads. Cell type enters as an input that modulates
    the sequence representation throughout the trunk, rather than only selecting an
    output head. Scales to any number of cell types (one embedding vector each) and
    is a step toward zero-shot transfer to unseen cell types.

    FiLM is initialized to identity (zero-init generators + the ``1 + gamma`` form),
    so the model starts ~unconditioned and learns modulation. The forward has no
    data-dependent control flow, so it is torch.compile-friendly as a whole.
    """

    def __init__(self, n_cell_types: int, inputlen: int = 2114, outputlen: int = 1000,
                 filters: int = 256, n_dil_layers: int = 8, embed_dim: int = 32):
        super().__init__()
        self.inputlen = inputlen
        self.outputlen = outputlen
        self.n_cell_types = n_cell_types
        self.embed_dim = embed_dim
        self.cell_emb = nn.Embedding(n_cell_types, embed_dim)
        self.first = nn.Conv1d(4, filters, kernel_size=21, padding=0)
        self.dilated = nn.ModuleList([
            nn.Conv1d(filters, filters, kernel_size=3, dilation=2 ** i, padding=0)
            for i in range(1, n_dil_layers + 1)
        ])
        # One FiLM generator per conditioned layer (after first conv + each dilated).
        self.film = nn.ModuleList([nn.Linear(embed_dim, 2 * filters) for _ in range(n_dil_layers + 1)])
        for lin in self.film:                      # identity init: gamma=0 (-> 1+gamma=1), beta=0
            nn.init.zeros_(lin.weight); nn.init.zeros_(lin.bias)
        self.accessibility = ProfileCountHead(filters, outputlen)
        self.nucleosome = ProfileCountHead(filters, outputlen)

    @staticmethod
    def _center_crop(x: torch.Tensor, width: int) -> torch.Tensor:
        diff = x.shape[-1] - width
        if diff < 0 or diff % 2 != 0:
            raise ValueError(f"Cannot center-crop length {x.shape[-1]} to {width}")
        crop = diff // 2
        return x[..., crop:-crop] if crop else x

    def _film_mod(self, x: torch.Tensor, e: torch.Tensor, i: int) -> torch.Tensor:
        gamma, beta = self.film[i](e).chunk(2, dim=-1)   # (B, filters) each
        return (1.0 + gamma).unsqueeze(-1) * x + beta.unsqueeze(-1)

    def forward(self, seq: torch.Tensor, ct_idx: torch.Tensor):
        if seq.shape[1] != 4:
            seq = seq.transpose(1, 2)
        e = self.cell_emb(ct_idx)                        # (B, embed_dim)
        x = F.relu(self.first(seq))
        x = self._film_mod(x, e, 0)
        for j, conv in enumerate(self.dilated):
            conv_x = F.relu(conv(x))
            x = conv_x + self._center_crop(x, conv_x.shape[-1])
            x = self._film_mod(x, e, j + 1)
        acc_profile, acc_count = self.accessibility(x)
        nuc_profile, nuc_count = self.nucleosome(x)
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
