"""Group detections of the same garment across frames, then pick the best crop.

Distance between two detections:
    cosine(embedding) + color_weight * deltaE(dominant LAB) / 100
    + category distance (small for detector-confusable pairs like
      jacket/dress, `penalty` for impossible ones like top/bottom)
    + penalty if both come from the same frame
The penalty acts as a cannot-link constraint: two garments visible in the
same frame are by definition different items.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .taxonomy import category_distance


def distance_matrix(
    embeddings: np.ndarray,
    colors_lab: np.ndarray,
    categories: list[str],
    frames: list[int],
    color_weight: float = 0.5,
    penalty: float = 10.0,
) -> np.ndarray:
    emb = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9)
    cos = 1.0 - emb @ emb.T
    lab = np.asarray(colors_lab, float)
    col = np.linalg.norm(lab[:, None, :] - lab[None, :, :], axis=-1) / 100.0
    d = cos + color_weight * col
    names = sorted(set(categories))
    table = np.array([[category_distance(a, b, penalty) for b in names] for a in names])
    ci = np.array([names.index(c) for c in categories])
    fr = np.asarray(frames)
    d = d + table[ci[:, None], ci[None, :]] + penalty * (fr[:, None] == fr[None, :])
    np.fill_diagonal(d, 0.0)
    return np.clip((d + d.T) / 2, 0.0, None)


def cluster(dist: np.ndarray, threshold: float = 0.25, linkage: str = "average") -> np.ndarray:
    """Agglomerative clustering on a precomputed distance matrix -> labels [n]."""
    n = len(dist)
    if n == 0:
        return np.zeros(0, dtype=int)
    if n == 1:
        return np.zeros(1, dtype=int)
    from sklearn.cluster import AgglomerativeClustering

    model = AgglomerativeClustering(
        n_clusters=None, metric="precomputed", linkage=linkage, distance_threshold=threshold
    )
    return model.fit_predict(dist)


def quality_scores(
    sharpness: np.ndarray,
    area_frac: np.ndarray,
    det_score: np.ndarray,
    touches_edge: np.ndarray,
    mask_fill: np.ndarray | None = None,
    edge_penalty: float = 0.4,
) -> np.ndarray:
    """sharpness x area x detection score, normalised within the group.

    Boxes touching the frame edge are probably cropped garments, so they're
    multiplied by edge_penalty. Masks that fill little of their box (SAM grabbed
    a fragment) are down-weighted too.
    """
    s = np.asarray(sharpness, float)
    a = np.asarray(area_frac, float)
    q = (s / (s.max() + 1e-9)) * (a / (a.max() + 1e-9)) * np.asarray(det_score, float)
    q = q * np.where(np.asarray(touches_edge, bool), edge_penalty, 1.0)
    if mask_fill is not None:
        q = q * np.clip(np.asarray(mask_fill, float) / 0.3, 0.0, 1.0)
    return q


def best_index(group_idx: np.ndarray, sharpness, area_frac, det_score, touches_edge, mask_fill=None) -> int:
    """Index (into the full arrays) of the best detection in a group."""
    g = np.asarray(group_idx)
    pick = lambda x: None if x is None else np.asarray(x)[g]
    q = quality_scores(pick(sharpness), pick(area_frac), pick(det_score), pick(touches_edge), pick(mask_fill))
    return int(g[int(np.argmax(q))])


def category_shares(categories: list[str], weights) -> dict[str, float]:
    """Score-weighted share of each detector category within a cluster."""
    tally: dict[str, float] = defaultdict(float)
    for c, w in zip(categories, weights):
        tally[c] += float(w)
    total = sum(tally.values()) or 1.0
    return {c: v / total for c, v in tally.items()}


def vote_category(categories: list[str], weights) -> str:
    """Score-weighted majority category for a cluster."""
    shares = category_shares(categories, weights)
    return max(shares, key=shares.get)


def is_partial_cluster(slivers) -> bool:
    """Clusters made only of edge slivers (see detect.is_sliver) are a garment
    entering/leaving the shot; its full view lives in another cluster."""
    return bool(np.asarray(slivers, bool).all())


def merge_lookalikes(
    labels: np.ndarray,
    embeddings: np.ndarray,
    colors_lab: np.ndarray,
    frames: list[int],
    max_dist: float = 0.05,
    color_weight: float = 0.5,
) -> np.ndarray:
    """Merge clusters whose mean appearance is nearly identical and that never
    appear in the same frame, ignoring category.

    The detector's labels are noisy (pink shorts called "hoodie" in a couple
    of frames), and the hard top/bottom penalty would otherwise keep those
    views apart forever. Two clusters in the same frame are always distinct.
    """
    labels = labels.copy()
    fr = np.asarray(frames)
    lab = np.asarray(colors_lab, float)
    while True:
        ks = np.unique(labels)
        if len(ks) < 2:
            return labels
        members = [np.flatnonzero(labels == k) for k in ks]
        means = np.stack([embeddings[m].mean(0) for m in members])
        means /= np.linalg.norm(means, axis=1, keepdims=True) + 1e-9
        cols = np.stack([lab[m].mean(0) for m in members])
        d = 1 - means @ means.T + color_weight * np.linalg.norm(cols[:, None] - cols[None], axis=-1) / 100
        fsets = [set(fr[m].tolist()) for m in members]
        np.fill_diagonal(d, np.inf)
        for i, j in zip(*np.unravel_index(np.argsort(d, axis=None), d.shape)):
            if d[i, j] > max_dist:
                return labels
            if i < j and not (fsets[i] & fsets[j]):
                labels[labels == ks[j]] = ks[i]
                break
        else:
            return labels
