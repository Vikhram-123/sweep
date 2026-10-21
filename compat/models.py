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


class TypeAwareMLP(nn.Module):
    """Model B (Vasileva et al. 2018): a shared general embedding, then a learned projection per
    unordered type pair into that pair's compatibility space.

    projection="diagonal": z = normalize(g(x) * mask[pair])   (128 params per pair, the paper's default)
    projection="full":     z = normalize(W[pair] @ g(x))      (128x128 per pair, paper Table 7b ablation)
    Projections start at the identity, so training starts from Model A's geometry.
    """

    def __init__(self, n_types: int, in_dim: int = 512, hidden: int = 256, out_dim: int = 128,
                 dropout: float = 0.1, normalize: bool = True, projection: str = "diagonal", **_):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, out_dim))
        self.normalize = normalize
        self.projection = projection
        iu = torch.triu_indices(n_types, n_types)
        pair = torch.empty(n_types, n_types, dtype=torch.long)
        pair[iu[0], iu[1]] = torch.arange(iu.shape[1])
        pair[iu[1], iu[0]] = torch.arange(iu.shape[1])
        self.register_buffer("pair_index", pair)  # symmetric: (u, v) and (v, u) share a space
        n_pairs = iu.shape[1]
        if projection == "diagonal":
            self.masks = nn.Parameter(torch.ones(n_pairs, out_dim))
        elif projection == "full":
            self.mats = nn.Parameter(torch.eye(out_dim).repeat(n_pairs, 1, 1))
        else:
            raise ValueError(f"unknown projection {projection!r}")

    def general(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def forward(self, x: torch.Tensor, own_type: torch.Tensor, other_type: torch.Tensor) -> torch.Tensor:
        g = self.net(x)
        pair = self.pair_index[own_type, other_type]
        if self.projection == "diagonal":
            z = g * self.masks[pair]
        else:
            z = torch.bmm(self.mats[pair], g.unsqueeze(-1)).squeeze(-1)
        return F.normalize(z, dim=-1) if self.normalize else z


def build_model(cfg: dict, n_types: int) -> nn.Module:
    if cfg["model"] == "pairwise":
        return PairwiseMLP(**cfg)
    if cfg["model"] == "type_aware":
        return TypeAwareMLP(n_types=n_types, **cfg)
    raise ValueError(f"unknown model {cfg['model']!r}")
