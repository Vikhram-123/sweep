"""Training losses for the compatibility models."""
from __future__ import annotations

import torch


def triplet_loss(anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor,
                 margin: float) -> torch.Tensor:
    """Triplet margin loss.

    anchor, positive, negative: (batch, dim) embeddings, already projected into the space
    where they are compared (for Model B: the anchor/positive type-pair space).
    margin: how much closer the positive must be than the negative before a triplet stops
    contributing.

    Returns a scalar tensor (the mean loss over the batch).
    """
    d_pos = (anchor - positive).norm(dim=-1)
    d_neg = (anchor - negative).norm(dim=-1)
    return torch.relu(d_pos - d_neg + margin).mean()
