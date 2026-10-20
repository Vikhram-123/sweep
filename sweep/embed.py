"""FashionCLIP image/text embeddings and zero-shot classification."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Embedder(Protocol):
    def embed_images(self, images: list[np.ndarray]) -> np.ndarray: ...
    def embed_texts(self, texts: list[str]) -> np.ndarray: ...


def l2norm(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


def zero_shot(image_emb: np.ndarray, text_emb: np.ndarray, temperature: float = 100.0) -> np.ndarray:
    """Softmax over labels for one (d,) image embedding vs (m, d) text embeddings."""
    logits = temperature * (l2norm(text_emb) @ l2norm(image_emb))
    e = np.exp(logits - logits.max())
    return e / e.sum()


def _features(out):
    # transformers v5 may return a ModelOutput instead of a bare tensor.
    if hasattr(out, "pooler_output") and not hasattr(out, "shape"):
        return out.pooler_output
    return out


class FashionClipEmbedder:
    def __init__(self, model_id: str = "patrickjohncyh/fashion-clip", device: str | None = None):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        from .device import get_device

        self.torch = torch
        self.device = device or get_device()
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model = CLIPModel.from_pretrained(model_id).to(self.device).eval()
        self._text_cache: dict[str, np.ndarray] = {}

    def embed_images(self, images: list[np.ndarray], batch_size: int = 32) -> np.ndarray:
        from PIL import Image

        out = []
        for i in range(0, len(images), batch_size):
            batch = [Image.fromarray(im) for im in images[i:i + batch_size]]
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
            with self.torch.inference_mode():
                feats = _features(self.model.get_image_features(**inputs))
            out.append(feats.float().cpu().numpy())
        return l2norm(np.concatenate(out)) if out else np.zeros((0, 512), np.float32)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        missing = [t for t in texts if t not in self._text_cache]
        if missing:
            inputs = self.processor(text=missing, return_tensors="pt", padding=True).to(self.device)
            with self.torch.inference_mode():
                feats = l2norm(_features(self.model.get_text_features(**inputs)).float().cpu().numpy())
            self._text_cache.update(zip(missing, feats))
        return np.stack([self._text_cache[t] for t in texts])
