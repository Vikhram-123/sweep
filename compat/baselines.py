"""Baseline outfit scorers on the disjoint test set.

    python -m compat.baselines

- random:            uniform noise (sanity check: AUC ~0.5, FITB ~25%)
- rules:             sweep/outfits.py score_outfit, occasion/weather off, all items
- rules_clothing:    same, accessories dropped (Sweep's taxonomy has no accessories)
- fashionclip_cosine: mean pairwise cosine similarity of FashionCLIP embeddings
"""
from __future__ import annotations

import numpy as np

from sweep.outfits import score_outfit
from sweep.taxonomy import PATTERNS, SUBTYPES, pattern_prompt, subtype_attrs, subtype_prompt

from .data import DATA_DIR, item_category, load_split
from .embed_items import load_colors, load_embeddings
from .evaluate import evaluate

# Polyvore semantic category -> Sweep category (None = accessory, outside Sweep's taxonomy).
TO_SWEEP = {"tops": "top", "bottoms": "bottom", "all-body": "dress", "outerwear": "outerwear", "shoes": "shoes"}
PROMPTS_PATH = DATA_DIR / "emb" / "text_prompts.npz"


def prompt_embeddings() -> dict[str, np.ndarray]:
    """FashionCLIP text embeddings of the app's subtype/pattern prompts (computed once, cached)."""
    if PROMPTS_PATH.exists():
        d = np.load(PROMPTS_PATH)
        return {k: d[k] for k in d.files}
    from sweep.embed import FashionClipEmbedder
    emb = FashionClipEmbedder()
    out = {}
    for cat, subs in SUBTYPES.items():
        out[f"subtype/{cat}"] = emb.embed_texts([subtype_prompt(cat, s) for s in subs])
        out[f"pattern/{cat}"] = emb.embed_texts([pattern_prompt(cat, p) for p in PATTERNS])
    np.savez(PROMPTS_PATH, **out)
    return out


def build_rule_items(item_ids, embs, colors) -> dict[str, dict]:
    """Sweep-style item dicts (category, subtype, pattern, formality, color) for Polyvore items."""
    prompts = prompt_embeddings()
    items = {}
    for iid in item_ids:
        cat = TO_SWEEP.get(item_category(iid))
        it = {"id": iid, "category": cat or "accessory", "color": colors[iid],
              "subtype": "accessory", "pattern": "solid", "formality": None, "warmth": 3}
        if cat:
            e = embs[iid]
            subs = list(SUBTYPES[cat])
            it["subtype"] = subs[int(np.argmax(prompts[f"subtype/{cat}"] @ e))]
            it["pattern"] = PATTERNS[int(np.argmax(prompts[f"pattern/{cat}"] @ e))]
            it["formality"], it["warmth"] = subtype_attrs(cat, it["subtype"])
        items[iid] = it
    return items


def rules_scorer(items: dict[str, dict], clothing_only: bool):
    def score(ids):
        its = [items[i] for i in ids]
        clothes = [it for it in its if it["category"] != "accessory"]
        if clothing_only:
            its = clothes
        if not its:
            return 0.0
        # Accessories have no formality in Sweep's table: give them the outfit's median so they
        # don't affect the formality-spread rule.
        f_fill = float(np.median([it["formality"] for it in clothes])) if clothes else 3.0
        its = [it if it["formality"] is not None else {**it, "formality": f_fill} for it in its]
        return score_outfit(its, occasion=None, weather=None)[0]
    return score


def cosine_scorer(embs: dict[str, np.ndarray]):
    def score(ids):
        if len(ids) < 2:
            return 0.0
        e = np.stack([embs[i] for i in ids])
        sim = e @ e.T
        iu = np.triu_indices(len(ids), 1)
        return float(sim[iu].mean())
    return score


def main(argv=None) -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="fashionclip",
                    help="fashionclip runs every baseline; another encoder (clip, resnet18, text) runs "
                         "only its cosine baseline (clip -> openai_clip_cosine)")
    args = ap.parse_args(argv)
    test = load_split("test")
    if args.encoder != "fashionclip":
        name = {"clip": "openai_clip_cosine"}.get(args.encoder, f"{args.encoder}_cosine")
        evaluate(name, cosine_scorer(load_embeddings(["test"], args.encoder)), test,
                 extra={"notes": f"mean pairwise cosine of frozen {args.encoder} embeddings"})
        return
    rng = np.random.default_rng(0)
    evaluate("random", lambda ids: float(rng.random()), test)

    embs = load_embeddings(["test"])
    colors = load_colors(["test"])
    items = build_rule_items(test.item_ids, embs, colors)
    evaluate("rules", rules_scorer(items, clothing_only=False), test)
    evaluate("rules_clothing", rules_scorer(items, clothing_only=True), test)
    evaluate("fashionclip_cosine", cosine_scorer(embs), test,
             extra={"notes": "mean pairwise cosine of frozen fashionclip embeddings"})


if __name__ == "__main__":
    main()
