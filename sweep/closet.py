"""Closet storage: items.json, embeddings.npz, item images, feedback.json.

Layout of a closet folder:
    items.json            list of item dicts
    embeddings.npz        ids + vectors (used to dedupe new scans against the closet)
    items/<id>.png        best RGBA cutout
    items/<id>_thumb.jpg  small thumbnail on white for the phone UI
    feedback.json         outfit thumbs up/down
    debug/<scan_id>/      frames/ detections/ masks/ clusters/ for each scan

No torch imports: the app imports this on every page load.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

# One process-wide lock: the Streamlit app edits the closet while a scan thread
# may be appending to it.
LOCK = threading.RLock()


def _atomic_write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


class Closet:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.items_dir = self.root / "items"

    @property
    def items_path(self) -> Path:
        return self.root / "items.json"

    @property
    def feedback_path(self) -> Path:
        return self.root / "feedback.json"

    # ---- items -------------------------------------------------------------
    def load_items(self) -> list[dict]:
        if not self.items_path.exists():
            return []
        with open(self.items_path) as f:
            return json.load(f)

    def save_items(self, items: list[dict]) -> None:
        with LOCK:
            _atomic_write_json(self.items_path, items)

    def image_path(self, item: dict, thumb: bool = True) -> Path:
        return self.root / (item.get("thumb") if thumb and item.get("thumb") else item["image"])

    def update_item(self, item_id: str, **fields) -> None:
        with LOCK:
            items = self.load_items()
            for it in items:
                if it["id"] == item_id:
                    it.update(fields)
            self.save_items(items)

    def delete_items(self, ids: list[str]) -> None:
        ids = set(ids)
        with LOCK:
            items = self.load_items()
            for it in items:
                if it["id"] in ids:
                    for key in ("image", "thumb"):
                        if it.get(key):
                            (self.root / it[key]).unlink(missing_ok=True)
            self.save_items([it for it in items if it["id"] not in ids])
            ids_e, vecs = self.load_embeddings()
            keep = [i for i, x in enumerate(ids_e) if x not in ids]
            self.save_embeddings([ids_e[i] for i in keep], vecs[keep] if len(vecs) else vecs)

    def merge_items(self, keep_id: str, merge_ids: list[str]) -> None:
        """Merge duplicates into keep_id: sum detection counts, average embeddings."""
        merge_ids = [m for m in merge_ids if m != keep_id]
        if not merge_ids:
            return
        with LOCK:
            items = {it["id"]: it for it in self.load_items()}
            keep = items[keep_id]
            keep["n_detections"] = sum(items[i].get("n_detections", 1) for i in [keep_id, *merge_ids])
            keep["merged_from"] = keep.get("merged_from", []) + merge_ids
            self.save_items(list(items.values()))
            ids_e, vecs = self.load_embeddings()
            pos = {x: i for i, x in enumerate(ids_e)}
            group = [pos[i] for i in [keep_id, *merge_ids] if i in pos]
            if group and keep_id in pos:
                v = vecs[group].mean(0)
                vecs[pos[keep_id]] = v / (np.linalg.norm(v) + 1e-9)
                self.save_embeddings(ids_e, vecs)
            self.delete_items(merge_ids)

    # ---- embeddings ---------------------------------------------------------
    def load_embeddings(self) -> tuple[list[str], np.ndarray]:
        p = self.root / "embeddings.npz"
        if not p.exists():
            return [], np.zeros((0, 0), np.float32)
        z = np.load(p, allow_pickle=False)
        return [str(x) for x in z["ids"]], z["vectors"]

    def save_embeddings(self, ids: list[str], vectors: np.ndarray) -> None:
        with LOCK:
            self.root.mkdir(parents=True, exist_ok=True)
            tmp = self.root / "embeddings.tmp.npz"
            np.savez(tmp, ids=np.array(ids, dtype=str), vectors=np.asarray(vectors, np.float32))
            os.replace(tmp, self.root / "embeddings.npz")

    # ---- feedback -----------------------------------------------------------
    def load_feedback(self) -> list[dict]:
        if not self.feedback_path.exists():
            return []
        with open(self.feedback_path) as f:
            return json.load(f)

    def add_feedback(self, item_ids: list[str], vote: int, context: dict | None = None) -> None:
        with LOCK:
            fb = self.load_feedback()
            fb.append({"items": sorted(item_ids), "vote": int(vote), "ts": time.time(), **(context or {})})
            _atomic_write_json(self.feedback_path, fb)

    def clear(self) -> None:
        with LOCK:
            shutil.rmtree(self.root, ignore_errors=True)
