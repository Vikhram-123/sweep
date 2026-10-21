"""Learned outfit compatibility (pure numpy; weights exported from compat/ by compat.export).

Model A from the compat-model experiments: a 2-layer MLP maps each item's FashionCLIP embedding
into a space where items that are worn together are close. Outfit score = mean pairwise
-distance. Trained on Polyvore Outfits (disjoint): test AUC 0.915, FITB 67.0%.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

DEFAULT_PATH = Path(__file__).with_name("compat_model.npz")


class CompatModel:
    def __init__(self, params):
        self.w1, self.b1 = np.asarray(params["w1"], np.float32), np.asarray(params["b1"], np.float32)
        self.w2, self.b2 = np.asarray(params["w2"], np.float32), np.asarray(params["b2"], np.float32)
        self.normalize = bool(params["normalize"])
        self.score_mean = float(params["score_mean"]) if "score_mean" in params else 0.0
        self.score_std = float(params["score_std"]) if "score_std" in params else 1.0
        self.rules_visual_std = float(params["rules_visual_std"]) if "rules_visual_std" in params else 1.0
        self.meta = json.loads(str(params["meta"])) if "meta" in params else {}

    @classmethod
    def load(cls, path: str | Path = DEFAULT_PATH) -> "CompatModel":
        with np.load(path) as d:
            return cls({k: d[k] for k in d.files})

    def embed(self, x: np.ndarray) -> np.ndarray:
        """(n, 512) FashionCLIP embeddings -> (n, d) compatibility-space points."""
        h = np.maximum(np.asarray(x, np.float32) @ self.w1.T + self.b1, 0.0)
        z = h @ self.w2.T + self.b2
        if self.normalize:
            z = z / (np.linalg.norm(z, axis=-1, keepdims=True) + 1e-12)
        return z

    def pair_distances(self, x: np.ndarray) -> np.ndarray:
        z = self.embed(x)
        return np.linalg.norm(z[:, None, :] - z[None, :, :], axis=-1)

    def outfit_score(self, x: np.ndarray) -> float:
        """Mean pairwise -distance (higher = more compatible); 0.0 for fewer than 2 items."""
        if len(x) < 2:
            return 0.0
        d = self.pair_distances(x)
        return float(-d[np.triu_indices(len(x), 1)].mean())

    def calibrated(self, score: float) -> float:
        """Model score on the scale of the rule-based visual terms it replaces (z-score * rules std)."""
        return (score - self.score_mean) / self.score_std * self.rules_visual_std


@lru_cache(maxsize=1)
def load_default() -> CompatModel | None:
    return CompatModel.load() if DEFAULT_PATH.exists() else None
