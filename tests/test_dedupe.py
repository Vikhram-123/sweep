import json

import numpy as np

from sweep.closet import Closet
from sweep.dedupe import best_index, cluster, distance_matrix
from sweep.detect import RawDetection, filter_detections
from sweep.scan import ScanConfig, run_scan
from sweep.taxonomy import label_to_category

from .conftest import FakeDetector, FakeEmbedder, FakeSegmenter


def noisy_views(rng, base, n, scale=0.03):
    v = base + rng.normal(0, scale, (n, len(base)))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_label_mapping_handles_dino_tokenization():
    assert label_to_category("t - shirt") == "top"
    assert label_to_category("shirt") == "top"
    assert label_to_category("jeans") == "bottom"
    assert label_to_category("coat jacket") == "outerwear"
    assert label_to_category("sneakers") == "shoes"
    assert label_to_category("dress") == "dress"
    assert label_to_category("hanger") is None


def test_filter_detections_size_and_nms():
    raw = [
        RawDetection((0, 0, 5, 5), 0.9, "shirt"),          # tiny
        RawDetection((0, 0, 100, 100), 0.9, "coat"),       # whole frame
        RawDetection((10, 10, 50, 60), 0.6, "jacket"),
        RawDetection((11, 11, 50, 61), 0.5, "coat"),       # same garment, other label
        RawDetection((60, 10, 90, 40), 0.4, "lamp"),       # unmapped label
        RawDetection((60, 50, 90, 90), 0.4, "sneakers"),
    ]
    kept = filter_detections(raw, (100, 100))
    assert [(d.label, c) for d, c in kept] == [("jacket", "outerwear"), ("sneakers", "shoes")]


def test_distance_penalties():
    emb = np.array([[1.0, 0], [1.0, 0], [1.0, 0]])
    lab = np.zeros((3, 3))
    d = distance_matrix(emb, lab, ["top", "top", "bottom"], [0, 0, 1])
    assert d[0, 1] >= 10      # same frame
    assert d[0, 2] >= 10      # different category
    assert np.allclose(d, d.T) and np.all(np.diag(d) == 0)


def test_cluster_recovers_items_across_frames():
    rng = np.random.default_rng(0)
    bases = rng.normal(size=(3, 32))
    bases /= np.linalg.norm(bases, axis=1, keepdims=True)
    emb, lab, cats, frames, truth = [], [], [], [], []
    for k, (cat, color) in enumerate([("top", (50, 60, 40)), ("top", (30, -5, -30)), ("bottom", (40, 0, -20))]):
        emb.append(noisy_views(rng, bases[k], 6))
        lab += [np.array(color) + rng.normal(0, 2, 3) for _ in range(6)]
        cats += [cat] * 6
        frames += list(range(k, k + 6))  # items overlap in frames 2..5
        truth += [k] * 6
    labels = cluster(distance_matrix(np.vstack(emb), np.array(lab), cats, frames), threshold=0.25)
    assert len(set(labels)) == 3
    for k in range(3):
        assert len(set(labels[np.array(truth) == k])) == 1


def test_identical_items_in_same_frame_never_merge():
    # Two identical black tees hanging side by side in every frame.
    emb = np.tile([[1.0, 0.0]], (6, 1))
    labels = cluster(distance_matrix(emb, np.zeros((6, 3)), ["top"] * 6, [0, 0, 1, 1, 2, 2]),
                     threshold=0.25, linkage="complete")
    assert len(set(labels)) == 2


def test_best_crop_prefers_sharp_large_and_not_cut_off():
    g = np.arange(4)
    sharp = np.array([100, 300, 300, 290])
    area = np.array([0.2, 0.05, 0.3, 0.29])
    score = np.array([0.9, 0.9, 0.9, 0.85])
    edge = np.array([False, False, True, False])  # #2 is biggest but cut off by the frame
    assert best_index(g, sharp, area, score, edge) == 3


def test_end_to_end_scan_with_fakes_and_append(closet_video, tmp_path):
    out = tmp_path / "closet"
    cfg = ScanConfig(threshold=0.25, min_detections=1)
    fakes = dict(detector=FakeDetector(), segmenter=FakeSegmenter(), embedder=FakeEmbedder())
    res = run_scan(closet_video, out, cfg, **fakes, progress=lambda f, m: None)

    items = Closet(out).load_items()
    got = sorted((it["category"], it["color"]["name"]) for it in items)
    assert got == sorted([("top", "red"), ("bottom", "denim"), ("outerwear", "forest green"),
                          ("shoes", "white"), ("dress", "pink")])
    assert res.n_detections > len(items)  # each garment seen in several frames
    for it in items:
        assert (out / it["image"]).exists() and (out / it["thumb"]).exists()
        assert it["n_detections"] >= 2
        assert 1 <= it["formality"] <= 5 and 1 <= it["warmth"] <= 5
    dbg = out / "debug" / res.scan_id
    for sub in ("frames", "detections", "masks", "clusters"):
        assert any((dbg / sub).iterdir()), sub
    assert json.loads((dbg / "summary.json").read_text())["n_frames"] == res.n_frames

    # Second scan of the same closet appends nothing new.
    res2 = run_scan(closet_video, out, cfg, **fakes, progress=lambda f, m: None)
    assert res2.added == [] and len(set(res2.matched)) == 5
    assert len(Closet(out).load_items()) == 5


def test_detector_label_noise_between_coat_and_dress_still_merges():
    # Same coat, labelled "outerwear" in 3 frames and "dress"/"top" in others.
    rng = np.random.default_rng(1)
    base = rng.normal(size=32)
    emb = noisy_views(rng, base / np.linalg.norm(base), 6)
    cats = ["outerwear", "outerwear", "dress", "outerwear", "top", "dress"]
    labels = cluster(distance_matrix(emb, np.zeros((6, 3)), cats, list(range(6))), threshold=0.25)
    assert len(set(labels)) == 1
    # ...but a top and a bottom never merge, however similar they look.
    labels = cluster(distance_matrix(emb[:2], np.zeros((2, 3)), ["top", "bottom"], [0, 1]), threshold=0.25)
    assert len(set(labels)) == 2


def test_vote_and_partial_cluster_rules():
    from sweep.dedupe import is_partial_cluster, vote_category

    assert vote_category(["dress", "outerwear", "outerwear"], [0.9, 0.4, 0.4]) == "dress"
    assert vote_category(["dress", "outerwear", "outerwear"], [0.5, 0.4, 0.4]) == "outerwear"
    assert is_partial_cluster([True, True])            # sliver entering the frame
    assert not is_partial_cluster([True, False])       # also seen whole
    assert is_partial_cluster([True] * 5)               # only ever seen as a strip


def test_sliver_vs_tall_item():
    from sweep.detect import is_sliver

    shape = (1000, 600)
    assert is_sliver((0, 100, 80, 900), shape)           # thin strip at the left edge
    assert is_sliver((540, 100, 600, 900), shape)        # ...or the right edge
    assert not is_sliver((150, 0, 450, 980), shape)      # long dress touching top & bottom
    assert not is_sliver((200, 880, 330, 1000), shape)   # shoes on the floor at the bottom edge
    assert not is_sliver((200, 300, 400, 600), shape)    # fully visible


def test_part_of_garment_box_is_suppressed():
    raw = [
        RawDetection((100, 50, 300, 450), 0.6, "hoodie"),
        RawDetection((120, 150, 290, 440), 0.5, "t-shirt"),   # hoodie body without the hood
        RawDetection((130, 300, 280, 440), 0.7, "jeans"),     # different group: kept
    ]
    kept = [d.label for d, _ in filter_detections(raw, (1000, 1000))]
    assert sorted(kept) == ["hoodie", "jeans"]


def test_merge_lookalikes_ignores_labels_but_not_frames():
    from sweep.dedupe import merge_lookalikes

    rng = np.random.default_rng(2)
    base = rng.normal(size=32)
    emb = noisy_views(rng, base / np.linalg.norm(base), 5, scale=0.01)
    lab = np.zeros((5, 3))
    # Clusters 0 and 1 are the same pink shorts (one mislabelled as a top).
    merged = merge_lookalikes(np.array([0, 0, 0, 1, 1]), emb, lab, [0, 1, 2, 3, 4])
    assert len(set(merged)) == 1
    # Same look but sharing a frame: two identical items, keep apart.
    kept = merge_lookalikes(np.array([0, 0, 0, 1, 1]), emb, lab, [0, 1, 2, 2, 4])
    assert len(set(kept)) == 2
