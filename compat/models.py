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
            # One matmul per type pair present. Gathering a matrix per row (bmm on mats[pair]) would
            # allocate rows x 128 x 128 floats -- 4 GB for a 65k-pair evaluation chunk.
            z = torch.empty_like(g)
            for p in pair.unique():
                rows = pair == p
                z[rows] = g[rows] @ self.mats[p].T
        return F.normalize(z, dim=-1) if self.normalize else z


class CSAMLP(nn.Module):
    """Model C, after CSA-Net (Lin et al. 2020): a shared general embedding x, k shared learnable
    masks, and per-category-pair attention weights over them:
        w = softmax(MLP([onehot(own_type), onehot(other_type)]))      (k weights)
        z = normalize( sum_i w_i * (x * m_i) )
    Unlike Model B's one mask per type pair, all pairs share the same k subspaces and only learn how
    to mix them, so rare pairs borrow strength from common ones.
    """

    def __init__(self, n_types: int, in_dim: int = 512, hidden: int = 256, out_dim: int = 128,
                 dropout: float = 0.1, normalize: bool = True, n_subspaces: int = 5, **_):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, out_dim))
        self.normalize = normalize
        self.n_types = n_types
        # Masks start near 1 (slightly perturbed so the subspaces can differentiate).
        self.masks = nn.Parameter(1.0 + 0.1 * torch.randn(n_subspaces, out_dim))
        self.attn = nn.Sequential(nn.Linear(2 * n_types, 64), nn.ReLU(), nn.Linear(64, n_subspaces))

    def forward(self, x: torch.Tensor, own_type: torch.Tensor, other_type: torch.Tensor) -> torch.Tensor:
        g = self.net(x)
        c = torch.cat([F.one_hot(own_type, self.n_types), F.one_hot(other_type, self.n_types)], -1).float()
        w = torch.softmax(self.attn(c), dim=-1)                       # (B, k)
        z = torch.einsum("bk,kd,bd->bd", w, self.masks, g)              # sum_i w_i (g * m_i)
        return F.normalize(z, dim=-1) if self.normalize else z


def build_model(cfg: dict, n_types: int) -> nn.Module:
    if cfg["model"] == "csa":
        return CSAMLP(n_types=n_types, **cfg)
    if cfg["model"] == "pairwise":
        return PairwiseMLP(**cfg)
    if cfg["model"] == "type_aware":
        return TypeAwareMLP(n_types=n_types, **cfg)
    raise ValueError(f"unknown model {cfg['model']!r}")
