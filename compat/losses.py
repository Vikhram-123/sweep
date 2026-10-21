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
    raise NotImplementedError("your turn -- see the hints in the chat")
