"""Precompute FashionCLIP embeddings + color tags for every Polyvore item (one pass over the images).

    python -m compat.embed_items                 # all splits, resumes where it left off
    python -m compat.embed_items --split test --batch 16

Output per split: data/polyvore/emb/<split>.npz with
    ids    (n,)      item ids
    emb    (n, 512)  L2-normalised FashionCLIP image embeddings (float32)
and data/polyvore/emb/<split>_colors.json  {item_id: tag_color record}.

Work is written in shards of --shard items; a shard only counts once it has been atomically
renamed into place, so an interrupted run (Ctrl-C, crash, lid closed) restarts at the first
missing shard. Memory stays bounded: images are streamed from parquet and the MPS allocator
is capped (sweep.device sets the watermark env vars before torch loads).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .data import DATA_DIR, SPLITS, iter_images, parquet_ids

OUT_DIR = DATA_DIR / "emb"


def product_rgba(img) -> np.ndarray:
    """RGBA cutout of a product shot: near-white regions touching the border become transparent."""
    import cv2

    rgb = np.asarray(img.convert("RGB"))
    if max(rgb.shape[:2]) > 256:  # colors don't need full resolution
        s = 256 / max(rgb.shape[:2])
        rgb = cv2.resize(rgb, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    white = (rgb.min(axis=2) > 235).astype(np.uint8)
    n, lab = cv2.connectedComponents(white, connectivity=4)
    border = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    bg = np.isin(lab, border[border > 0]) & (white > 0)
    alpha = np.where(bg, 0, 255).astype(np.uint8)
    if (alpha > 0).mean() < 0.02:  # e.g. a white item that bleeds into the border
        alpha[:] = 255
    return np.dstack([rgb, alpha])


def _color(img) -> dict:
    from sweep.tagging import tag_color

    c = tag_color(product_rgba(img))
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in c.items()}


def embed_split(split: str, embedder, shard: int, batch: int, workers: int) -> None:
    from sweep.device import free_memory

    shard_dir = OUT_DIR / split
    shard_dir.mkdir(parents=True, exist_ok=True)
    ids = parquet_ids(split)
    n_shards = (len(ids) + shard - 1) // shard
    done_ids = set()
    for k in range(n_shards):
        p = shard_dir / f"{k:05d}.npz"
        if p.exists():
            done_ids.update(np.load(p)["ids"].tolist())
    todo = len(ids) - len(done_ids)
    print(f"[{split}] {len(ids)} items, {len(done_ids)} cached, {todo} to embed", flush=True)

    pool = ThreadPoolExecutor(workers)
    stream = iter_images(split, skip=done_ids)
    k = len(done_ids) // shard  # shards are filled in parquet order, so this is the next index
    t0, seen = time.time(), 0
    while True:
        chunk = [x for _, x in zip(range(shard), stream)]
        if not chunk:
            break
        chunk_ids = [i for i, _ in chunk]
        imgs = [im for _, im in chunk]
        colors = pool.map(_color, imgs)  # CPU work overlaps with the GPU forward pass below
        emb = np.concatenate([embedder.embed_images([np.asarray(im) for im in imgs[j:j + batch]], batch)
                              for j in range(0, len(imgs), batch)]).astype(np.float32)
        colors = list(colors)
        tmp = shard_dir / f"{k:05d}.tmp.npz"
        np.savez(tmp, ids=np.array(chunk_ids), emb=emb,
                 colors=np.array([json.dumps(c) for c in colors]))
        os.replace(tmp, shard_dir / f"{k:05d}.npz")
        free_memory()
        k += 1
        seen += len(chunk)
        rate = seen / (time.time() - t0)
        print(f"[{split}] shard {k}/{n_shards}  {rate:.0f} img/s  eta {(todo - seen) / rate / 60:.1f} min",
              flush=True)
    pool.shutdown()
    merge(split, ids)


def merge(split: str, ids: list[str]) -> None:
    parts = [np.load(p) for p in sorted((OUT_DIR / split).glob("[0-9]*.npz"))]
    all_ids = np.concatenate([p["ids"] for p in parts])
    emb = np.concatenate([p["emb"] for p in parts])
    colors = dict(zip(all_ids.tolist(), (json.loads(c) for p in parts for c in p["colors"])))
    assert sorted(all_ids.tolist()) == sorted(ids), "shards don't cover the split exactly"
    np.savez(OUT_DIR / f"{split}.npz", ids=all_ids, emb=emb)
    (OUT_DIR / f"{split}_colors.json").write_text(json.dumps(colors))
    print(f"[{split}] merged -> {OUT_DIR / split}.npz  {emb.shape}", flush=True)


def load_embeddings(splits=SPLITS) -> dict[str, np.ndarray]:
    """item_id -> embedding for the given splits."""
    out = {}
    for s in splits:
        d = np.load(OUT_DIR / f"{s}.npz")
        out.update(zip(d["ids"].tolist(), d["emb"]))
    return out


def load_colors(splits=SPLITS) -> dict[str, dict]:
    out = {}
    for s in splits:
        out.update(json.loads((OUT_DIR / f"{s}_colors.json").read_text()))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Embed all Polyvore items with FashionCLIP")
    ap.add_argument("--split", choices=SPLITS, action="append")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--shard", type=int, default=2048)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args(argv)

    from sweep.embed import FashionClipEmbedder
    embedder = FashionClipEmbedder()
    print("device:", embedder.device, flush=True)
    for split in args.split or ["valid", "test", "train"]:
        embed_split(split, embedder, args.shard, args.batch, args.workers)


if __name__ == "__main__":
    main()
