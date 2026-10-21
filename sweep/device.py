"""Torch device auto-detection: mps (Apple Silicon) > cuda > cpu."""

from __future__ import annotations

import os

# Some ops used by Grounding DINO / SAM are not implemented on MPS yet; let
# PyTorch fall back to CPU for those instead of crashing. Must be set before
# torch is imported.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
# PyTorch's default MPS limit (1.7x the recommended working set) lets the GPU
# allocate more than physical RAM on Apple Silicon, which swaps the whole Mac
# to a crawl. Cap it at the recommended working set so a spike fails fast.
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "1.0")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.8")


def get_device(prefer: str | None = None) -> str:
    """Return the best available torch device string.

    `prefer` (or the SWEEP_DEVICE env var) forces a specific device.
    """
    import torch

    forced = prefer or os.environ.get("SWEEP_DEVICE")
    if forced:
        return forced
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def align_weights(model, device: str):
    """Copy CPU weights out of the memory-mapped checkpoint into aligned buffers.

    transformers maps safetensors files zero-copy, so on CPU each weight lives at its byte offset
    in the file -- for these checkpoints 2 bytes off float32 alignment. Apple's BLAS then returns
    NaNs (batched matmul) or dies with SIGBUS (batch-1 matrix-vector). Moving to mps/cuda already
    copies, so this only does work on CPU.
    """
    if device == "cpu":
        for t in list(model.parameters()) + list(model.buffers()):
            if t.data_ptr() % 64:
                t.data = t.data.clone()
    return model


def free_memory() -> None:
    """Return cached GPU blocks to the system between pipeline stages."""
    import sys

    torch = sys.modules.get("torch")
    if torch is None:
        return
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()
