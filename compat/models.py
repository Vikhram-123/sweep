"""Compatibility models on frozen 512-d FashionCLIP embeddings.

Every model maps an item into the space where it is compared with an item of another type:
    model(x, own_type, other_type) -> z
and pair distance is ||z_a - z_b||. Model A ignores the types; Model B (type-aware) will use them.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class PairwiseMLP(nn.Module):
    """Model A: one shared MLP for every item, regardless of type."""

    def __init__(self, in_dim: int = 512, hidden: int = 256, out_dim: int = 128, dropout: float = 0.1,
                 normalize: bool = True, **_):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, out_dim))
        self.normalize = normalize

    def forward(self, x: torch.Tensor, own_type: torch.Tensor | None = None,
                other_type: torch.Tensor | None = None) -> torch.Tensor:
        z = self.net(x)
        return F.normalize(z, dim=-1) if self.normalize else z


def build_model(cfg: dict, n_types: int) -> nn.Module:
    if cfg["model"] == "pairwise":
        return PairwiseMLP(**cfg)
    raise ValueError(f"unknown model {cfg['model']!r}")
