"""Two lightweight baseline MLPs for 28x28 grayscale image classification."""
from __future__ import annotations

import torch
from torch import nn


class SmallMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(784, 64), nn.ReLU(), nn.Linear(64, 10))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LargeMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(),
                                 nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 10))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_model(name: str) -> nn.Module:
    if name == 'small':
        return SmallMLP()
    if name == 'large':
        return LargeMLP()
    raise ValueError(f'Unknown model: {name}')
