"""Load each model once per process and reuse it (CLI, Streamlit server, Colab)."""

from __future__ import annotations

import logging
from functools import lru_cache

from . import device as _device  # noqa: F401  (sets MPS fallback env before torch loads)

log = logging.getLogger(__name__)


@lru_cache(maxsize=None)
def get_detector(device: str | None = None, box_threshold: float = 0.3):
    from .detect import GroundingDinoDetector

    log.info("loading Grounding DINO")
    return GroundingDinoDetector(device=device, box_threshold=box_threshold)


@lru_cache(maxsize=None)
def get_segmenter(device: str | None = None):
    from .segment import SamSegmenter

    log.info("loading SAM")
    return SamSegmenter(device=device)


@lru_cache(maxsize=None)
def get_embedder(device: str | None = None):
    from .embed import FashionClipEmbedder

    log.info("loading FashionCLIP")
    return FashionClipEmbedder(device=device)
