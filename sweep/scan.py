"""Scan a closet video into items: `python -m sweep.scan video.mp4 --out closet/`.

frames -> detect -> segment -> embed -> dedupe -> best crop -> tag -> save.
New items are appended to an existing closet; items that match something
already in the closet (same distance + threshold as dedupe) are not re-added.
"""

from __future__ import annotations

import argparse
import logging
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .closet import LOCK, Closet
from .colors import delta_e, mean_lab
from .dedupe import (category_shares, cluster, distance_matrix, is_partial_cluster,
                     merge_lookalikes, quality_scores)
from .detect import Detection, filter_detections, is_sliver, touches_edge
from .device import free_memory
from .frames import extract_frames, sharpness
from .segment import clean_mask, make_cutout, on_white
from .tagging import choose_category, tag_item
from .taxonomy import category_distance

log = logging.getLogger("sweep.scan")

ProgressFn = Callable[[float, str], None]

CATEGORY_BGR = {"top": (60, 180, 75), "bottom": (200, 130, 0), "outerwear": (48, 130, 245),
                "dress": (180, 30, 145), "shoes": (25, 225, 255)}


@dataclass
class ScanConfig:
    fps: float = 3.0
    keep_fraction: float = 0.7
    max_side: int = 1024
    box_threshold: float = 0.3
    min_area_frac: float = 0.01
    max_area_frac: float = 0.7
    threshold: float = 0.25          # dedupe distance threshold
    color_weight: float = 0.5
    linkage: str = "average"
    merge_threshold: float = 0.05   # cross-category look-alike merge
    min_detections: int = 2
    debug: bool = True
    device: str | None = None


@dataclass
class ScanResult:
    scan_id: str
    n_frames: int = 0
    n_detections: int = 0
    n_clusters: int = 0
    added: list[str] = field(default_factory=list)
    matched: list[str] = field(default_factory=list)
    dropped_small_clusters: int = 0
    dropped_partial_clusters: int = 0
    debug_dir: str = ""
    seconds: float = 0.0


def _bgr(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGBA2BGRA if rgb.shape[-1] == 4 else cv2.COLOR_RGB2BGR)


def _thumb(rgba: np.ndarray, size: int = 384) -> np.ndarray:
    rgb = on_white(rgba)
    h, w = rgb.shape[:2]
    s = size / max(h, w)
    rgb = cv2.resize(rgb, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), 255, np.uint8)
    y, x = (size - rgb.shape[0]) // 2, (size - rgb.shape[1]) // 2
    canvas[y:y + rgb.shape[0], x:x + rgb.shape[1]] = rgb
    return canvas


def _contact_sheet(tiles: list[np.ndarray], best: int, tile: int = 160, cols: int = 6) -> np.ndarray:
    rows = (len(tiles) + cols - 1) // cols
    sheet = np.full((rows * tile, min(cols, len(tiles)) * tile, 3), 255, np.uint8)
    for i, t in enumerate(tiles):
        im = _thumb(t, tile)
        if i == best:
            cv2.rectangle(im, (0, 0), (tile - 1, tile - 1), (0, 200, 0), 6)
        r, c = divmod(i, cols)
        sheet[r * tile:(r + 1) * tile, c * tile:(c + 1) * tile] = im
    return sheet


def _cli_progress(t_start: float) -> ProgressFn:
    """Log a line per stage change or every 10%, not on every tick."""
    last = {"pct": -100, "stage": None}

    def report(frac: float, msg: str) -> None:
        pct, stage = round(frac * 100), msg.split(":")[0].split("(")[0].split(" ")[0]
        if stage != last["stage"] or pct - last["pct"] >= 10 or frac >= 1:
            log.info("[%3d%%] %5.1fs  %s", pct, time.time() - t_start, msg)
            last.update(pct=pct, stage=stage)

    return report


def _load_default_models(cfg: ScanConfig):
    from .models import get_detector, get_embedder, get_segmenter

    return (get_detector(cfg.device, cfg.box_threshold), get_segmenter(cfg.device), get_embedder(cfg.device))


def run_scan(video: str | Path, out: str | Path, cfg: ScanConfig | None = None, *,
             detector=None, segmenter=None, embedder=None, progress: ProgressFn | None = None) -> ScanResult:
    cfg = cfg or ScanConfig()
    t_start = time.time()
    progress = progress or _cli_progress(t_start)
    closet = Closet(out)
    scan_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]
    dbg = closet.root / "debug" / scan_id
    res = ScanResult(scan_id=scan_id, debug_dir=str(dbg) if cfg.debug else "")
    if cfg.debug:
        for sub in ("frames", "detections", "masks", "clusters"):
            (dbg / sub).mkdir(parents=True, exist_ok=True)

    # 1. frames ---------------------------------------------------------------
    progress(0.0, "Reading video")
    frames = extract_frames(video, fps=cfg.fps, keep_fraction=cfg.keep_fraction, max_side=cfg.max_side)
    res.n_frames = len(frames)
    if cfg.debug:
        for f in frames:
            cv2.imwrite(str(dbg / "frames" / f"{f.index:04d}_t{f.time_s:06.2f}_s{f.sharpness:.0f}.jpg"),
                        _bgr(f.image), [cv2.IMWRITE_JPEG_QUALITY, 85])
    progress(0.12, f"Kept {len(frames)} sharp frames")

    if detector is None or segmenter is None or embedder is None:
        progress(0.13, "Loading models (first run downloads ~1.5 GB)")
        d, s, e = _load_default_models(cfg)
        detector, segmenter, embedder = detector or d, segmenter or s, embedder or e

    # 2. detect ---------------------------------------------------------------
    dets: list[Detection] = []
    chunk = 4
    for i in range(0, len(frames), chunk):
        batch = frames[i:i + chunk]
        for f, raw in zip(batch, detector.detect([f.image for f in batch])):
            for r, cat in filter_detections(raw, f.image.shape, cfg.min_area_frac, cfg.max_area_frac):
                h, w = f.image.shape[:2]
                x0, y0, x1, y1 = r.box
                dets.append(Detection(id=len(dets), frame=f.index, box=r.box, score=r.score, label=r.label,
                                      category=cat, area_frac=(x1 - x0) * (y1 - y0) / (w * h),
                                      touches_edge=touches_edge(r.box, f.image.shape),
                                      sliver=is_sliver(r.box, f.image.shape)))
        progress(0.15 + 0.3 * min(1.0, (i + chunk) / len(frames)), f"Detecting clothes: {len(dets)} found")
    res.n_detections = len(dets)
    free_memory()

    # 3. segment --------------------------------------------------------------
    by_frame: dict[int, list[Detection]] = {}
    for d in dets:
        by_frame.setdefault(d.frame, []).append(d)
    for n, (fi, fdets) in enumerate(by_frame.items()):
        img = frames[fi].image
        masks = segmenter.segment(img, [d.box for d in fdets])
        vis = _bgr(img).copy() if cfg.debug else None
        for d, m in zip(fdets, masks):
            m = clean_mask(m, d.box)
            x0, y0, x1, y1 = d.box
            d.mask_fill = float(m.sum()) / max(1.0, (x1 - x0) * (y1 - y0))
            d.cutout = make_cutout(img, m, d.box)
            d.sharpness = sharpness(d.cutout[..., :3])
            if cfg.debug:
                color = CATEGORY_BGR[d.category]
                overlay = vis.copy()
                overlay[m] = color
                vis = cv2.addWeighted(overlay, 0.35, vis, 0.65, 0)
                cv2.rectangle(vis, (int(x0), int(y0)), (int(x1), int(y1)), color, 2)
                cv2.putText(vis, f"#{d.id} {d.label} {d.score:.2f}", (int(x0) + 3, int(y0) + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.imwrite(str(dbg / "masks" / f"f{fi:04d}_d{d.id:04d}_{d.category}.png"), _bgr(d.cutout))
        if cfg.debug:
            cv2.imwrite(str(dbg / "detections" / f"{fi:04d}.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
        progress(0.45 + 0.3 * (n + 1) / max(1, len(by_frame)), f"Cutting out items ({n + 1}/{len(by_frame)} frames)")

    free_memory()
    dets = [d for d in dets if d.cutout is not None and d.cutout[..., 3].any()]
    if not dets:
        progress(1.0, "No clothing found")
        res.seconds = time.time() - t_start
        return res

    # 4. embed + per-detection color -------------------------------------------
    progress(0.76, f"Embedding {len(dets)} crops")
    emb = embedder.embed_images([on_white(d.cutout) for d in dets])
    labs = np.array([mean_lab(d.cutout) for d in dets])

    # 5. dedupe ----------------------------------------------------------------
    progress(0.86, "Grouping duplicates")
    dist = distance_matrix(emb, labs, [d.category for d in dets], [d.frame for d in dets], cfg.color_weight)
    labels = cluster(dist, cfg.threshold, cfg.linkage)
    labels = merge_lookalikes(labels, emb, labs, [d.frame for d in dets], cfg.merge_threshold, cfg.color_weight)
    groups = [np.flatnonzero(labels == k) for k in np.unique(labels)]
    groups.sort(key=len, reverse=True)
    res.n_clusters = len(groups)

    arr = lambda attr: np.array([getattr(d, attr) for d in dets])
    sharp, area, score, edge, fill = arr("sharpness"), arr("area_frac"), arr("score"), arr("touches_edge"), arr("mask_fill")
    sliver = arr("sliver")

    # 6-7. best crop + tags ----------------------------------------------------
    existing = closet.load_items()
    ex_ids, ex_vecs = closet.load_embeddings()
    ex_pos = {x: i for i, x in enumerate(ex_ids)}
    new_items, new_vecs = [], []
    now = time.time()
    for k, g in enumerate(groups):
        q = quality_scores(sharp[g], area[g], score[g], edge[g], fill[g])
        best = dets[g[int(np.argmax(q))]]
        vec = emb[g[np.argsort(-q)[:5]]].mean(0)  # mean of the 5 best views
        vec = vec / (np.linalg.norm(vec) + 1e-9)
        cat = choose_category(category_shares([dets[i].category for i in g], score[g] * area[g]), vec, embedder)
        too_few, partial = len(g) < cfg.min_detections, is_partial_cluster(sliver[g])
        if cfg.debug:
            note = "_dropped-few" if too_few else "_dropped-sliver" if partial else ""
            sheet = _contact_sheet([dets[i].cutout for i in g], int(np.argmax(q)))
            cv2.imwrite(str(dbg / "clusters" / f"c{k:03d}_{cat}_n{len(g)}{note}.jpg"), _bgr(sheet))
        if too_few:
            res.dropped_small_clusters += 1
            continue
        if partial:
            res.dropped_partial_clusters += 1
            continue
        tags = tag_item(cat, vec, best.cutout, [dets[i].label for i in g], embedder)

        # Already in the closet? Same distance definition as dedupe.
        match = None
        for it in existing:
            if it["id"] not in ex_pos or category_distance(it["category"], cat) >= 1:
                continue
            d = category_distance(it["category"], cat) + 1 - float(ex_vecs[ex_pos[it["id"]]] @ vec) + cfg.color_weight * float(
                delta_e(np.array(it["color"]["lab"]), np.array(tags["color"]["lab"]))) / 100
            if d < cfg.threshold and (match is None or d < match[1]):
                match = (it, d)
        if match:
            res.matched.append(match[0]["id"])
            continue

        item_id = uuid.uuid4().hex[:8]
        closet.items_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(closet.items_dir / f"{item_id}.png"), _bgr(best.cutout))
        cv2.imwrite(str(closet.items_dir / f"{item_id}_thumb.jpg"), _bgr(_thumb(best.cutout)),
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
        new_items.append({
            "id": item_id, "category": cat, **tags,
            "image": f"items/{item_id}.png", "thumb": f"items/{item_id}_thumb.jpg",
            "n_detections": int(len(g)),
            "labels": dict(Counter(dets[i].label for i in g)),
            "source": {"scan_id": scan_id, "frame": best.frame, "time_s": round(frames[best.frame].time_s, 2),
                       "box": [round(v, 1) for v in best.box], "score": round(best.score, 3),
                       "cluster": f"c{k:03d}"},
            "added_at": now, "user_edited": False,
        })
        new_vecs.append(vec)
        res.added.append(item_id)
        progress(0.88 + 0.1 * (k + 1) / len(groups), f"Tagged {len(new_items)} new items")

    # 8. save (re-read under the lock so concurrent app edits aren't lost) -------
    with LOCK:
        items = closet.load_items()
        ids_e, vecs_e = closet.load_embeddings()
        matched = Counter(res.matched)
        for it in items:
            if it["id"] in matched:
                it["last_seen_scan"] = scan_id
        closet.save_items(items + new_items)
        if new_vecs:
            all_vecs = np.vstack([vecs_e, new_vecs]) if len(vecs_e) else np.array(new_vecs)
            closet.save_embeddings(ids_e + [it["id"] for it in new_items], all_vecs)
    res.seconds = round(time.time() - t_start, 1)
    if cfg.debug:
        import json

        with open(dbg / "summary.json", "w") as f:
            json.dump({**res.__dict__, "video": str(video), "config": cfg.__dict__}, f, indent=2)
    progress(1.0, f"Done: {len(res.added)} new items, {len(res.matched)} already in closet")
    return res


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m sweep.scan", description=__doc__.split("\n")[0])
    p.add_argument("video")
    p.add_argument("--out", default="closet", help="closet folder (appended to if it exists)")
    p.add_argument("--fps", type=float, default=3.0)
    p.add_argument("--keep", type=float, default=0.7, help="fraction of sharpest frames to keep")
    p.add_argument("--max-side", type=int, default=1024)
    p.add_argument("--box-threshold", type=float, default=0.3, help="Grounding DINO box score threshold")
    p.add_argument("--threshold", type=float, default=0.25, help="dedupe distance threshold (higher = merge more)")
    p.add_argument("--color-weight", type=float, default=0.5)
    p.add_argument("--linkage", choices=["average", "complete", "single"], default="average")
    p.add_argument("--min-detections", type=int, default=2, help="drop items seen in fewer frames")
    p.add_argument("--device", default=None, help="force mps/cuda/cpu")
    p.add_argument("--no-debug", action="store_true")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(message)s")  # quiet HTTP/model-loading chatter
    logging.getLogger("sweep").setLevel(logging.INFO)
    cfg = ScanConfig(fps=a.fps, keep_fraction=a.keep, max_side=a.max_side, box_threshold=a.box_threshold,
                     threshold=a.threshold, color_weight=a.color_weight, linkage=a.linkage,
                     min_detections=a.min_detections, device=a.device, debug=not a.no_debug)
    res = run_scan(a.video, a.out, cfg)
    print(f"\nscan {res.scan_id}: {res.n_frames} frames, {res.n_detections} detections, "
          f"{res.n_clusters} clusters -> {len(res.added)} new items, {len(res.matched)} already in closet "
          f"({res.seconds}s)\n  dropped: {res.dropped_small_clusters} seen too few times, "
          f"{res.dropped_partial_clusters} only seen cut off at the frame edge")
    if res.debug_dir:
        print(f"debug images: {res.debug_dir}")
    items = {it["id"]: it for it in Closet(a.out).load_items()}
    for i in res.added:
        it = items[i]
        print(f"  {i}  {it['category']:<9} {it['subtype']:<20} {it['color']['name']:<11} {it['pattern']:<13} "
              f"formality={it['formality']} warmth={it['warmth']} seen={it['n_detections']}x")


if __name__ == "__main__":
    main()
