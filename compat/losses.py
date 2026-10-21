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


def squared_triplet_loss(anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor,
                         margin: float) -> torch.Tensor:
    """Same hinge as triplet_loss, on squared Euclidean distances (FaceNet-style).

    Used only by the squared-vs-plain distance experiment (config distance: sq), where evaluation
    also averages squared distances so training and testing measure the same quantity.
    """
    d_pos = ((anchor - positive) ** 2).sum(dim=-1)
    d_neg = ((anchor - negative) ** 2).sum(dim=-1)
    return torch.relu(d_pos - d_neg + margin).mean()


def info_nce_loss(anchor: torch.Tensor, positive: torch.Tensor, valid_neg: torch.Tensor,
                  logit_scale: torch.Tensor) -> torch.Tensor:
    """InfoNCE with in-batch negatives.

    anchor, positive: (B, d) L2-normalised embeddings; row i of `positive` is anchor i's true partner.
    valid_neg: (B, B) bool. valid_neg[i, j] is True when positive[j] may serve as a negative for
        anchor i (same category as positive[i], from a different outfit, not the same item).
        The diagonal is False: positive[i] is the answer, not a negative.
    logit_scale: scalar tensor = 1 / temperature (learned; the training loop keeps it positive).

    Returns the mean over the batch of the cross-entropy of picking positive[i] for anchor i among
    {positive[i]} + its valid negatives.
    """
    raise NotImplementedError("your turn -- see the hints in the chat")
