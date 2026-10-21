"""Rule-based outfit generation (pure numpy, no torch).

An outfit is top+bottom+shoes or dress+shoes, with optional outerwear. Each
rule adds/subtracts points and leaves a short reason, so the UI can explain
why an outfit was picked. Feedback (thumbs up/down) nudges scores through
item-pair affinities, so liking one outfit also lifts outfits that share
pairs with it.

Visual harmony comes from one of two scorers: the hand-written color/pattern/contrast rules
("rules"), or the learned compatibility model in sweep/compat.py ("model"). Occasion, weather,
formality and feedback rules apply either way.
"""

from __future__ import annotations

import itertools
import math
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from .colors import hue_deg
from .compat import CompatModel, load_default

# Target mean formality (1-5) per occasion.
OCCASIONS: dict[str, float] = {
    "lounge": 1.2, "casual": 1.8, "smart casual": 2.7, "date night": 3.2, "work": 3.5, "formal": 4.6,
}
# Target outfit warmth (1-5) per weather.
WEATHER: dict[str, float] = {"hot": 1.3, "warm": 2.0, "mild": 2.8, "cool": 3.6, "cold": 4.5}

W_FORMALITY = 0.6
W_WARMTH = 0.5
FEEDBACK_PAIR = 0.3
FEEDBACK_EXACT = 0.6


@dataclass
class Outfit:
    items: list[dict]
    score: float
    reasons: list[str]
    breakdown: dict[str, float] = field(default_factory=dict)

    @property
    def ids(self) -> list[str]:
        return [it["id"] for it in self.items]

    def to_dict(self) -> dict:
        return {"items": self.ids, "score": round(self.score, 3), "reasons": self.reasons,
                "breakdown": {k: round(v, 3) for k, v in self.breakdown.items()}}


def hue_diff(a: dict, b: dict) -> float:
    d = abs(hue_deg(a["color"]["rgb"]) - hue_deg(b["color"]["rgb"])) % 360
    return min(d, 360 - d)


def outfit_warmth(items: list[dict]) -> float:
    by = {it["category"]: it for it in items}
    if "dress" in by:
        core = by["dress"]["warmth"]
    else:
        core = np.mean([by[c]["warmth"] for c in ("top", "bottom") if c in by] or [3])
    w = 0.85 * core + 0.15 * by["shoes"]["warmth"] if "shoes" in by else core
    if "outerwear" in by:
        w += 0.6 * (by["outerwear"]["warmth"] - 1)
    return float(w)


def _label(it: dict) -> str:
    return f"{it['color']['name']} {it['subtype']}"


class ModelHarmony:
    """Learned visual compatibility for a closet: each item is mapped into the model's space once,
    so scoring an outfit is just pairwise distances."""

    def __init__(self, model: CompatModel, embeddings: dict[str, np.ndarray]):
        self.model = model
        ids = [i for i, v in embeddings.items() if v is not None]
        z = model.embed(np.stack([embeddings[i] for i in ids])) if ids else np.zeros((0, 1))
        self.z = dict(zip(ids, z))

    def score(self, items: list[dict]) -> tuple[float, list[tuple[float, str]]]:
        its = [it for it in items if it["id"] in self.z]
        if len(its) < 2:
            return 0.0, []
        pairs = [(float(np.linalg.norm(self.z[a["id"]] - self.z[b["id"]])), a, b)
                 for a, b in itertools.combinations(its, 2)]
        term = self.model.calibrated(-float(np.mean([d for d, _, _ in pairs])))
        notes = []
        if term >= 0.15:
            notes.append((term, "pieces look like they belong together"))
        elif term <= -0.15:
            notes.append((term, "pieces don't quite go together"))
        best, worst = min(pairs, key=lambda p: p[0]), max(pairs, key=lambda p: p[0])
        if len(pairs) > 1 and best[0] < worst[0]:
            notes.append((0.1, f"{_label(best[1])} and {_label(best[2])} pair especially well"))
            if term < 0:
                notes.append((-0.2, f"{_label(worst[1])} is the odd one out with {_label(worst[2])}"))
        return term, notes


class FeedbackModel:
    """Votes -> pair affinities and exact-outfit scores."""

    def __init__(self, feedback: list[dict] | None = None):
        self.pairs: dict[frozenset, float] = defaultdict(float)
        self.exact: dict[frozenset, float] = defaultdict(float)
        for fb in feedback or []:
            ids = fb["items"]
            self.exact[frozenset(ids)] += fb["vote"]
            for a, b in itertools.combinations(ids, 2):
                self.pairs[frozenset((a, b))] += fb["vote"]

    def bonus(self, ids: list[str]) -> float:
        pairs = [self.pairs.get(frozenset(p), 0.0) for p in itertools.combinations(ids, 2)]
        pair_term = FEEDBACK_PAIR * float(np.mean(np.tanh(pairs))) if pairs else 0.0
        return pair_term + FEEDBACK_EXACT * math.tanh(self.exact.get(frozenset(ids), 0.0))


def _rule_harmony(items: list[dict], by: dict, notes: list, br: dict) -> None:
    # --- color harmony -------------------------------------------------------
    color = 0.0
    accents = [it for it in items if not it["color"]["neutral"]]
    bold = [it for it in accents if it["color"].get("bold")]
    if not accents:
        color += 0.15
        notes.append((0.15, "all neutrals, easy to wear"))
    elif len(bold) == 1 and len(accents) == 1:
        color += 0.3
        notes.append((0.3, f"{bold[0]['color']['name']} {bold[0]['subtype']} as the one statement color"))
    elif len(accents) == 1:
        color += 0.2
        notes.append((0.2, f"soft {accents[0]['color']['name']} accent on neutrals"))
    for a, b in itertools.combinations(accents, 2):
        hd = hue_diff(a, b)
        both_bold = a["color"].get("bold") and b["color"].get("bold")
        if a["color"]["name"] == b["color"]["name"] or hd < 40:
            color += 0.1
            notes.append((0.1, f"{a['color']['name']} and {b['color']['name']} are analogous"))
        elif 150 <= hd <= 210:
            delta = -0.1 if both_bold else 0.05
            color += delta
            notes.append((delta, f"complementary {a['color']['name']} and {b['color']['name']}"))
        else:
            delta = -0.4 if both_bold else -0.15
            color += delta
            notes.append((delta, f"{a['color']['name']} and {b['color']['name']} clash"))
    if len(bold) > 2:
        delta = -0.5 * (len(bold) - 2)
        color += delta
        notes.append((delta, f"{len(bold)} bold colors is a lot"))
    br["color"] = color

    # --- patterns ------------------------------------------------------------
    patterned = [it for it in items if it.get("pattern", "solid") != "solid"]
    pat = 0.0
    if len(patterned) == 1:
        pat = 0.15
        notes.append((pat, f"{patterned[0]['pattern']} {patterned[0]['subtype']} is the only pattern"))
    elif len(patterned) >= 2:
        pat = -0.5 * (len(patterned) - 1)
        notes.append((pat, "competing patterns: " + " + ".join(it["pattern"] for it in patterned)))
    br["pattern"] = pat

    # --- light/dark contrast between top and bottom ------------------------------
    con = 0.0
    upper = by.get("top")
    if upper is not None and "bottom" in by:
        dl = abs(upper["color"]["lab"][0] - by["bottom"]["color"]["lab"][0])
        if dl >= 20:
            con = 0.3 * min(dl, 50) / 50
            notes.append((con, "good light/dark contrast"))
        elif dl < 8 and upper["color"]["name"] != by["bottom"]["color"]["name"]:
            con = -0.1
            notes.append((con, "top and bottom are similar but not matching"))
    br["contrast"] = con


def score_outfit(items: list[dict], occasion: str | None = "casual", weather: str | None = "mild",
                 feedback: FeedbackModel | None = None,
                 harmony: ModelHarmony | None = None) -> tuple[float, list[tuple[float, str]], dict]:
    """Return (score, [(delta, reason)], breakdown). occasion/weather None skips that rule.

    With `harmony`, the learned model replaces the color, pattern and contrast rules.
    """
    notes: list[tuple[float, str]] = []
    br: dict[str, float] = {}
    by = {it["category"]: it for it in items}

    if harmony is not None:
        br["model"], model_notes = harmony.score(items)
        notes.extend(model_notes)
    else:
        _rule_harmony(items, by, notes, br)

    # --- formality -----------------------------------------------------------------
    f = np.array([it["formality"] for it in items], float)
    spread = float(f.max() - f.min())
    form = -0.3 * max(0.0, spread - 1.5)
    if spread > 1.5:
        lo, hi = min(items, key=lambda it: it["formality"]), max(items, key=lambda it: it["formality"])
        notes.append((form, f"{hi['subtype']} is dressier than {lo['subtype']}"))
    if occasion is not None:
        gap = abs(float(f.mean()) - OCCASIONS[occasion])
        form -= W_FORMALITY * gap
        if gap < 0.5:
            notes.append((0.2, f"right level of dressy for {occasion}"))
        elif gap > 1.2:
            notes.append((-W_FORMALITY * gap, f"too {'dressy' if f.mean() > OCCASIONS[occasion] else 'casual'} for {occasion}"))
    br["formality"] = form

    # --- warmth --------------------------------------------------------------------
    warm = 0.0
    if weather is not None:
        w = outfit_warmth(items)
        gap = w - WEATHER[weather]
        warm = -W_WARMTH * abs(gap)
        if abs(gap) < 0.5:
            notes.append((0.2, f"right warmth for {weather} weather"))
        elif abs(gap) > 1.0:
            notes.append((warm, f"too {'warm' if gap > 0 else 'light'} for {weather} weather"))
    br["warmth"] = warm

    # --- feedback ------------------------------------------------------------------
    fbk = feedback.bonus([it["id"] for it in items]) if feedback else 0.0
    if abs(fbk) >= 0.05:
        notes.append((fbk, "you liked similar outfits" if fbk > 0 else "you disliked similar outfits"))
    br["feedback"] = fbk

    return float(sum(br.values())), notes, br


def _reasons(notes: list[tuple[float, str]], k: int = 3) -> list[str]:
    pos = [n for d, n in sorted(notes, key=lambda x: -x[0]) if d > 0][:k]
    neg = [n for d, n in sorted(notes, key=lambda x: x[0]) if d <= -0.2][:1]
    return pos + [f"but {n}" for n in neg]


def generate_outfits(
    items: list[dict],
    occasion: str = "casual",
    weather: str = "mild",
    n: int = 5,
    must_include: str | None = None,
    feedback: list[dict] | None = None,
    allow_outerwear: bool = True,
    beam: int = 400,
    diversity: float = 0.35,
    scorer: str = "rules",
    embeddings: dict[str, np.ndarray] | None = None,
    model: CompatModel | None = None,
) -> list[Outfit]:
    """Top-n diverse outfits for the occasion/weather.

    scorer="model" scores visual harmony with the learned compatibility model; it needs each
    item's FashionCLIP embedding (`embeddings`, id -> vector) and uses the bundled weights unless
    `model` is given.
    """
    fb = FeedbackModel(feedback)
    harmony = None
    if scorer == "model":
        model = model or load_default()
        if model is None or not embeddings:
            raise ValueError("model scorer needs sweep/compat_model.npz and item embeddings")
        harmony = ModelHarmony(model, embeddings)
    elif scorer != "rules":
        raise ValueError(f"unknown scorer {scorer!r}")
    must = next((it for it in items if it["id"] == must_include), None) if must_include else None
    pool = defaultdict(list)
    for it in items:
        pool[it["category"]].append(it)
    if must is not None:
        pool[must["category"]] = [must]
        if must["category"] == "dress":
            pool["top"], pool["bottom"] = [], []
        elif must["category"] in ("top", "bottom"):
            pool["dress"] = []

    def partial_score(combo):
        return score_outfit(combo, occasion, None, fb, harmony)[0]

    def prune(combos):
        combos.sort(key=partial_score, reverse=True)
        return combos[:beam]

    cores = [[t, b] for t in pool["top"] for b in pool["bottom"]] + [[d] for d in pool["dress"]]
    cores = prune(cores)
    shoes = pool["shoes"] or [None]
    with_shoes = prune([c + ([s] if s else []) for c in cores for s in shoes])
    outers = [None] + (pool["outerwear"] if allow_outerwear else [])
    if must is not None and must["category"] == "outerwear":
        outers = [must]
    full = [c + ([o] if o else []) for c in with_shoes for o in outers]

    scored = []
    for combo in full:
        s, notes, br = score_outfit(combo, occasion, weather, fb, harmony)
        scored.append(Outfit(items=combo, score=s, reasons=_reasons(notes), breakdown=br))
    scored.sort(key=lambda o: o.score, reverse=True)

    # Greedy diverse selection: penalise sharing items with already-picked outfits.
    chosen: list[Outfit] = []
    remaining = scored[: max(beam, 4 * n)]
    while remaining and len(chosen) < n:
        def adjusted(o):
            if not chosen:
                return o.score
            shared = max(len((set(o.ids) & set(c.ids)) - {must_include}) for c in chosen)
            return o.score - diversity * shared
        best = max(remaining, key=adjusted)
        chosen.append(best)
        remaining.remove(best)
    return chosen
