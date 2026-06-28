#!/usr/bin/env python3
"""Small PyTorch smoke test for the multi-task nucleosome model."""
import torch

from chrombpnet.training.models.multitask_nucleosome_torch import (
    MultiTaskNucleosomeModel,
    multitask_loss,
)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultiTaskNucleosomeModel(inputlen=2114, outputlen=1000, filters=32, n_dil_layers=8).to(device)
    seq = torch.randn(4, 2114, 4, device=device)
    outs = model(seq)
    assert [tuple(o.shape) for o in outs] == [(4, 1000), (4, 1), (4, 1000), (4, 1)]
    targets = (
        torch.poisson(torch.ones(4, 1000, device=device)),
        torch.ones(4, 1, device=device),
        torch.poisson(torch.ones(4, 1000, device=device)),
        torch.ones(4, 1, device=device),
    )
    loss = multitask_loss(outs, targets)
    loss.backward()
    assert torch.isfinite(loss)
    print('TORCH_SMOKE_OK', device, float(loss.detach()))


if __name__ == '__main__':
    main()
