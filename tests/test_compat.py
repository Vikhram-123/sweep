import json

import numpy as np
import pytest

from sweep.compat import DEFAULT_PATH, CompatModel
from sweep.outfits import ModelHarmony, generate_outfits, score_outfit

from .test_outfits import BY_ID, CLOSET, outfit

D = 8


def toy_model(**calib):
    """Identity MLP: an item's compatibility point is just its normalised (ReLU'd) embedding."""
    eye = np.eye(D, dtype=np.float32)
    return CompatModel({"w1": eye, "b1": np.zeros(D), "w2": eye, "b2": np.zeros(D), "normalize": True,
                        "score_mean": calib.get("mean", -1.0), "score_std": calib.get("std", 0.5),
                        "rules_visual_std": calib.get("rules_std", 0.4)})


def onehot(k, noise=0.0, seed=0):
    v = np.zeros(D, np.float32)
    v[k] = 1.0
    return v + noise * np.random.default_rng(seed).random(D).astype(np.float32)


def test_forward_matches_manual_computation():
    rng = np.random.default_rng(0)
    p = {"w1": rng.normal(size=(6, D)), "b1": rng.normal(size=6), "w2": rng.normal(size=(4, 6)),
         "b2": rng.normal(size=4), "normalize": True}
    m = CompatModel(p)
    x = rng.normal(size=(3, D)).astype(np.float32)
    h = np.maximum(x @ p["w1"].T + p["b1"], 0) @ p["w2"].T + p["b2"]
    np.testing.assert_allclose(m.embed(x), h / np.linalg.norm(h, axis=1, keepdims=True), rtol=1e-5)
    d = np.linalg.norm(m.embed(x)[0] - m.embed(x)[1])
    assert m.pair_distances(x)[0, 1] == pytest.approx(d, rel=1e-5)


def test_forward_matches_torch():
    torch = pytest.importorskip("torch")
    net = torch.nn.Sequential(torch.nn.Linear(D, 16), torch.nn.ReLU(), torch.nn.Dropout(0.1), torch.nn.Linear(16, 4)).eval()
    m = CompatModel({"w1": net[0].weight.detach().numpy(), "b1": net[0].bias.detach().numpy(),
                     "w2": net[3].weight.detach().numpy(), "b2": net[3].bias.detach().numpy(), "normalize": True})
    x = np.random.default_rng(1).normal(size=(10, D)).astype(np.float32)
    with torch.no_grad():
        ref = torch.nn.functional.normalize(net(torch.from_numpy(x)), dim=-1).numpy()
    np.testing.assert_allclose(m.embed(x), ref, atol=1e-6)


def test_outfit_score_properties():
    m = toy_model()
    same = np.stack([onehot(0), onehot(0, 0.1), onehot(0, 0.1, 2)])
    mixed = np.stack([onehot(0), onehot(3), onehot(5)])
    assert m.outfit_score(same) > m.outfit_score(mixed)
    assert m.outfit_score(mixed) == pytest.approx(m.outfit_score(mixed[::-1]))  # order doesn't matter
    assert m.outfit_score(mixed[:1]) == 0.0
    assert m.calibrated(m.score_mean) == pytest.approx(0.0)
    assert m.calibrated(m.score_mean + m.score_std) == pytest.approx(m.rules_visual_std)


def test_bundled_weights_load():
    if not DEFAULT_PATH.exists():
        pytest.skip("no exported model")
    m = CompatModel.load()
    assert m.w1.shape[1] == 512 and m.normalize
    assert m.embed(np.ones((2, 512), np.float32)).shape == (2, m.w2.shape[0])
    assert m.meta["test"]["compat_auc"] > 0.85 and m.score_std > 0 and m.rules_visual_std > 0
    json.dumps(m.meta)


def _embeddings(groups):
    """Items in the same group get (nearly) the same embedding, i.e. the model thinks they go together."""
    return {iid: onehot(g, 0.05, seed=k) for k, (iid, g) in enumerate(groups.items())}


def test_model_replaces_visual_rules_but_keeps_occasion_and_weather():
    emb = _embeddings({"tee_red": 0, "shorts_floral": 0, "sneakers": 0})
    harmony = ModelHarmony(toy_model(), emb)
    combo = outfit("tee_red", "shorts_floral", "sneakers")
    _, _, rules_br = score_outfit(combo, "formal", "cold")
    _, notes, model_br = score_outfit(combo, "formal", "cold", harmony=harmony)
    assert {"color", "pattern", "contrast"} <= set(rules_br) and not {"color", "pattern", "contrast"} & set(model_br)
    assert model_br["model"] > 0  # all three items sit together in model space
    for k in ("formality", "warmth", "feedback"):
        assert model_br[k] == pytest.approx(rules_br[k])
    assert any("belong together" in n for _, n in notes)


def test_generate_follows_the_model():
    # The model says jeans/white tee/sneakers belong together and everything else is unrelated.
    groups = {it["id"]: 1 + k % (D - 1) for k, it in enumerate(CLOSET)}
    groups.update({"tee_white": 0, "jeans": 0, "sneakers": 0})
    emb = _embeddings(groups)
    best = generate_outfits(CLOSET, "casual", "mild", n=1, scorer="model", embeddings=emb,
                            model=toy_model(), allow_outerwear=False)[0]
    assert set(best.ids) == {"tee_white", "jeans", "sneakers"}
    assert "model" in best.breakdown


def test_items_without_embeddings_are_skipped_and_errors_are_clear():
    emb = {"jeans": onehot(0), "sneakers": onehot(0)}
    term, _ = ModelHarmony(toy_model(), emb).score(outfit("tee_red", "jeans", "sneakers"))
    assert term > 0  # scored on the two items it knows
    assert ModelHarmony(toy_model(), {"jeans": onehot(0)}).score(outfit("tee_red", "jeans"))[0] == 0.0
    with pytest.raises(ValueError):
        generate_outfits(CLOSET, scorer="model", embeddings={}, model=toy_model())
    with pytest.raises(ValueError):
        generate_outfits(CLOSET, scorer="vibes")
