"""Categories, detector label mapping, subtypes with formality/warmth, patterns."""

from __future__ import annotations

import re

CATEGORIES = ["top", "bottom", "outerwear", "dress", "shoes"]

DETECTION_PROMPT = (
    "shirt. t-shirt. sweater. hoodie. jacket. coat. pants. jeans. "
    "shorts. skirt. dress. shoes. sneakers."
)

# Checked in order, so "t-shirt" wins over "shirt" and "sneakers" over "shoes".
_LABEL_KEYWORDS: list[tuple[str, str]] = [
    ("t-shirt", "top"), ("tshirt", "top"), ("sweater", "top"), ("hoodie", "top"),
    ("shirt", "top"),
    ("jacket", "outerwear"), ("coat", "outerwear"),
    ("jeans", "bottom"), ("pants", "bottom"), ("shorts", "bottom"), ("skirt", "bottom"),
    ("dress", "dress"),
    ("sneakers", "shoes"), ("shoes", "shoes"), ("shoe", "shoes"),
]


def normalize_label(label: str) -> str:
    """Grounding DINO decodes 't-shirt' as 't - shirt'; undo tokenizer spacing."""
    return re.sub(r"\s*-\s*", "-", label.lower()).strip(" .")


def label_to_category(label: str) -> str | None:
    norm = normalize_label(label)
    for kw, cat in _LABEL_KEYWORDS:
        if kw in norm:
            return cat
    return None


# Extra dedupe distance between categories. Grounding DINO often calls the same
# coat "jacket" in one frame and "dress" or "shirt" in the next, so upper-body
# categories get a small soft penalty; physically impossible pairs (top vs
# bottom, shoes vs anything) are absent here and get the hard penalty.
_SOFT_CATEGORY_DISTANCE = {
    frozenset(("top", "outerwear")): 0.08,
    frozenset(("top", "dress")): 0.12,
    frozenset(("outerwear", "dress")): 0.12,
    frozenset(("bottom", "dress")): 0.15,  # skirt vs dress
}


def category_distance(a: str, b: str, hard: float = 10.0) -> float:
    if a == b:
        return 0.0
    return _SOFT_CATEGORY_DISTANCE.get(frozenset((a, b)), hard)


# subtype -> (formality 1-5, warmth 1-5), per category.
SUBTYPES: dict[str, dict[str, tuple[int, int]]] = {
    "top": {
        "t-shirt": (1, 2), "tank top": (1, 1), "long-sleeve t-shirt": (1, 3),
        "polo shirt": (2, 2), "button-up shirt": (3, 2), "dress shirt": (4, 2),
        "blouse": (3, 2), "sweater": (3, 4), "cardigan": (3, 3),
        "hoodie": (1, 4), "sweatshirt": (1, 4), "turtleneck": (3, 4),
        "flannel shirt": (2, 3),
    },
    "bottom": {
        "jeans": (2, 3), "chinos": (3, 3), "dress pants": (4, 3),
        "shorts": (1, 1), "sweatpants": (1, 3), "cargo pants": (1, 3),
        "skirt": (3, 2), "leggings": (1, 2),
    },
    "outerwear": {
        "denim jacket": (2, 3), "leather jacket": (2, 3), "bomber jacket": (2, 3),
        "blazer": (4, 2), "puffer jacket": (1, 5), "parka": (1, 5),
        "trench coat": (4, 3), "wool coat": (4, 5), "rain jacket": (1, 2),
        "fleece jacket": (1, 4), "vest": (2, 2),
    },
    "dress": {
        "casual dress": (2, 2), "sundress": (2, 1), "shirt dress": (3, 2),
        "maxi dress": (3, 2), "cocktail dress": (4, 2), "evening gown": (5, 2),
        "sweater dress": (2, 4),
    },
    "shoes": {
        "sneakers": (1, 2), "running shoes": (1, 2), "loafers": (3, 2),
        "dress shoes": (5, 2), "boots": (3, 4), "chelsea boots": (3, 3),
        "sandals": (1, 1), "heels": (4, 1), "flats": (3, 2), "slides": (1, 1),
    },
}

PATTERNS = ["solid", "striped", "plaid", "checked", "floral", "polka dot",
            "graphic print", "camouflage", "animal print"]

# Detector label -> subtype prior, used to break near-ties in zero-shot.
LABEL_SUBTYPE_HINT = {
    "t-shirt": "t-shirt", "sweater": "sweater", "hoodie": "hoodie", "shirt": "button-up shirt",
    "jeans": "jeans", "shorts": "shorts", "skirt": "skirt", "sneakers": "sneakers",
}


def subtype_attrs(category: str, subtype: str) -> tuple[int, int]:
    """(formality, warmth) for a subtype; mid values if unknown."""
    return SUBTYPES.get(category, {}).get(subtype, (3, 3))


def subtype_prompt(category: str, subtype: str) -> str:
    return f"a photo of {subtype}"


def pattern_prompt(category: str, pattern: str) -> str:
    noun = {"top": "top", "bottom": "pants", "outerwear": "jacket", "dress": "dress", "shoes": "shoes"}[category]
    return f"a photo of {'a plain solid color' if pattern == 'solid' else 'a ' + pattern} {noun}"
