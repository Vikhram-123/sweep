"""Tag an item: dominant color, subtype + pattern (zero-shot), formality/warmth."""

from __future__ import annotations

from collections import Counter

import numpy as np

from .colors import dominant_colors
from .embed import Embedder, zero_shot
from .taxonomy import (LABEL_SUBTYPE_HINT, PATTERNS, SUBTYPES, normalize_label, pattern_prompt,
                       subtype_attrs, subtype_prompt)

HINT_BOOST = 0.15  # added to the detector-suggested subtype's probability

CATEGORY_PROMPTS = {
    "top": "a photo of a shirt or top",
    "bottom": "a photo of pants, shorts or a skirt",
    "outerwear": "a photo of a jacket or coat",
    "dress": "a photo of a dress",
    "shoes": "a photo of shoes",
}


def choose_category(detector_shares: dict[str, float], item_emb: np.ndarray, embedder: Embedder,
                    clip_weight: float = 0.5) -> str:
    """Blend the detector's vote with a FashionCLIP zero-shot over categories.

    The detector alone confuses coats with dresses and shirts in some frames;
    FashionCLIP looks at the best views of the whole item.
    """
    cats = list(CATEGORY_PROMPTS)
    clip_p = zero_shot(item_emb, embedder.embed_texts([CATEGORY_PROMPTS[c] for c in cats]))
    score = {c: (1 - clip_weight) * detector_shares.get(c, 0.0) + clip_weight * float(p) for c, p in zip(cats, clip_p)}
    return max(score, key=score.get)


def tag_color(rgba: np.ndarray) -> dict:
    """Dominant color record plus an optional secondary color (>= 25% share)."""
    cols = dominant_colors(rgba, k=3)
    main = dict(cols[0])
    secondary = next((c for c in cols[1:] if c["share"] >= 0.25 and c["name"] != main["name"]), None)
    main["secondary"] = secondary["name"] if secondary else None
    return main


def tag_item(category: str, item_emb: np.ndarray, best_rgba: np.ndarray, labels: list[str],
             embedder: Embedder) -> dict:
    subtypes = list(SUBTYPES[category])
    sub_p = zero_shot(item_emb, embedder.embed_texts([subtype_prompt(category, s) for s in subtypes]))
    top_label = Counter(normalize_label(l) for l in labels).most_common(1)[0][0] if labels else ""
    hint = LABEL_SUBTYPE_HINT.get(top_label)
    if hint in subtypes:
        sub_p = sub_p.copy()
        sub_p[subtypes.index(hint)] += HINT_BOOST
    subtype = subtypes[int(np.argmax(sub_p))]

    pat_p = zero_shot(item_emb, embedder.embed_texts([pattern_prompt(category, p) for p in PATTERNS]))
    pattern = PATTERNS[int(np.argmax(pat_p))]

    formality, warmth = subtype_attrs(category, subtype)
    return {
        "subtype": subtype,
        "subtype_confidence": round(float(sub_p.max()), 3),
        "pattern": pattern,
        "pattern_confidence": round(float(pat_p.max()), 3),
        "color": tag_color(best_rgba),
        "formality": formality,
        "warmth": warmth,
    }
