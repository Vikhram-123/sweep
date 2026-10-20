import sys

from sweep.outfits import generate_outfits, outfit_warmth, score_outfit

from .conftest import make_item

CLOSET = [
    make_item("tee_white", "top", "t-shirt", "white"),
    make_item("tee_red", "top", "t-shirt", "red"),
    make_item("shirt_plaid", "top", "flannel shirt", "green", pattern="plaid"),
    make_item("dress_shirt", "top", "dress shirt", "light blue"),
    make_item("sweater_navy", "top", "sweater", "navy"),
    make_item("jeans", "bottom", "jeans", "denim"),
    make_item("chinos", "bottom", "chinos", "khaki"),
    make_item("trousers", "bottom", "dress pants", "charcoal"),
    make_item("shorts_floral", "bottom", "shorts", "orange", pattern="floral"),
    make_item("sneakers", "shoes", "sneakers", "white"),
    make_item("oxfords", "shoes", "dress shoes", "brown"),
    make_item("sandals", "shoes", "sandals", "tan"),
    make_item("blazer", "outerwear", "blazer", "navy"),
    make_item("puffer", "outerwear", "puffer jacket", "black"),
    make_item("sundress", "dress", "sundress", "yellow"),
]
BY_ID = {it["id"]: it for it in CLOSET}


def outfit(*ids):
    return [BY_ID[i] for i in ids]


def test_no_torch_import():
    import subprocess
    code = "import sweep.outfits, sweep.closet, sys; assert 'torch' not in sys.modules and 'cv2' not in sys.modules"
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


def test_neutrals_beat_clashing_bold_colors():
    calm, _, _ = score_outfit(outfit("tee_red", "jeans", "sneakers"))
    clash, notes, _ = score_outfit(outfit("tee_red", "shorts_floral", "sneakers"))
    assert calm > clash


def test_clashing_patterns_penalized():
    _, notes, br = score_outfit(outfit("shirt_plaid", "shorts_floral", "sandals"))
    assert br["pattern"] < 0 and any("competing patterns" in n for _, n in notes)
    _, _, br1 = score_outfit(outfit("shirt_plaid", "chinos", "sneakers"))
    assert br1["pattern"] > 0


def test_light_dark_contrast_rewarded():
    _, _, contrast = score_outfit(outfit("tee_white", "trousers", "sneakers"))
    _, _, flat = score_outfit(outfit("sweater_navy", "trousers", "oxfords"))
    assert contrast["contrast"] > flat["contrast"]


def test_formality_matches_occasion():
    work = outfit("dress_shirt", "trousers", "oxfords")
    lazy = outfit("tee_white", "jeans", "sneakers")
    assert score_outfit(work, "formal")[0] > score_outfit(lazy, "formal")[0]
    assert score_outfit(lazy, "casual")[0] > score_outfit(work, "casual")[0]
    # Mixing a dress shirt with sandals is inconsistent regardless of target.
    assert score_outfit(outfit("dress_shirt", "trousers", "sandals"), None)[2]["formality"] < 0


def test_warmth_vs_weather():
    assert outfit_warmth(outfit("tee_white", "jeans", "sneakers", "puffer")) > outfit_warmth(
        outfit("tee_white", "jeans", "sneakers"))
    top = generate_outfits(CLOSET, "casual", "cold", n=1)[0]
    assert "puffer" in top.ids or "sweater_navy" in top.ids
    hot = generate_outfits(CLOSET, "casual", "hot", n=3)
    assert all("puffer" not in o.ids for o in hot)


def test_generate_valid_diverse_and_must_include():
    outs = generate_outfits(CLOSET, "smart casual", "mild", n=5)
    assert len(outs) == 5
    for o in outs:
        cats = [it["category"] for it in o.items]
        assert cats.count("shoes") == 1
        assert ("dress" in cats) != ("top" in cats and "bottom" in cats)
        assert o.reasons
    assert len({tuple(o.ids) for o in outs}) == 5
    first = set(outs[0].ids)
    assert any(len(first & set(o.ids)) <= 1 for o in outs[1:])

    for must in ("blazer", "sundress", "shorts_floral"):
        got = generate_outfits(CLOSET, "casual", "warm", n=3, must_include=must)
        assert got and all(must in o.ids for o in got)


def test_feedback_nudges_scores():
    base = generate_outfits(CLOSET, "casual", "mild", n=3)
    disliked = base[0]
    fb = [{"items": disliked.ids, "vote": -1}]
    after = generate_outfits(CLOSET, "casual", "mild", n=3, feedback=fb)
    assert after[0].ids != disliked.ids

    target = outfit("tee_red", "chinos", "oxfords")  # unusual combo
    s0 = score_outfit(target, "casual", "mild")[0]
    from sweep.outfits import FeedbackModel
    s1 = score_outfit(target, "casual", "mild", FeedbackModel([{"items": [i["id"] for i in target], "vote": 1}]))[0]
    assert s1 > s0
    # Pair affinity generalizes: liking red tee + chinos lifts it with other shoes too.
    other = outfit("tee_red", "chinos", "sneakers")
    fbm = FeedbackModel([{"items": [i["id"] for i in target], "vote": 1}])
    assert score_outfit(other, "casual", "mild", fbm)[0] > score_outfit(other, "casual", "mild")[0]


def test_small_closets():
    assert generate_outfits([], "casual", "mild") == []
    only = [make_item("t", "top", "t-shirt", "white"), make_item("b", "bottom", "jeans", "denim")]
    got = generate_outfits(only, "casual", "mild")
    assert len(got) == 1 and set(got[0].ids) == {"t", "b"}
