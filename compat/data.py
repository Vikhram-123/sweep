"""Polyvore Outfits (disjoint split) loading.

Questions in the raw files name items as "<outfit_id>_<position>". We resolve those to real
item ids here, once, so nothing downstream can peek at the outfit id (in FITB, the correct
candidate is exactly the one whose outfit id matches the question's -- a free answer).
"""
from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterator

DATA_DIR = Path("data/polyvore")
SPLITS = ("train", "valid", "test")
_PARQUET = {"train": "train", "valid": "validation", "test": "test"}


@dataclass
class FITBQuestion:
    context: list[str]          # item ids of the partial outfit
    candidates: list[str]       # 4 item ids
    answer: int                 # index into candidates
    blank_position: int


@dataclass
class Split:
    name: str
    outfits: dict[str, list[str]]                       # set_id -> item ids
    compat: list[tuple[int, list[str]]] = field(default_factory=list)
    fitb: list[FITBQuestion] = field(default_factory=list)

    @property
    def item_ids(self) -> set[str]:
        return {i for items in self.outfits.values() for i in items}


@lru_cache(maxsize=1)
def item_metadata(data_dir: Path = DATA_DIR) -> dict[str, dict]:
    return json.loads((data_dir / "polyvore_item_metadata.json").read_text())


def item_category(item_id: str, data_dir: Path = DATA_DIR) -> str:
    """Coarse semantic category ("tops", "bottoms", "shoes", ...) -- the paper's types."""
    return item_metadata(data_dir)[item_id]["semantic_category"]


def load_split(name: str, data_dir: Path = DATA_DIR) -> Split:
    root = data_dir / "disjoint"
    raw = json.loads((root / f"{name}.json").read_text())
    outfits, index = {}, {}
    for o in raw:
        outfits[o["set_id"]] = [it["item_id"] for it in o["items"]]
        for it in o["items"]:
            index[f"{o['set_id']}_{it['index']}"] = it["item_id"]
    split = Split(name, outfits)

    for line in (root / f"compatibility_{name}.txt").read_text().splitlines():
        label, *keys = line.split()
        split.compat.append((int(label), [index[k] for k in keys]))

    for q in json.loads((root / f"fill_in_blank_{name}.json").read_text()):
        outfit = q["question"][0].split("_")[0]
        answers = [a.split("_")[0] == outfit for a in q["answers"]]
        assert sum(answers) == 1, q
        split.fitb.append(FITBQuestion(
            context=[index[k] for k in q["question"]],
            candidates=[index[k] for k in q["answers"]],
            answer=answers.index(True),
            blank_position=q["blank_position"],
        ))
    return split


def iter_images(name: str, data_dir: Path = DATA_DIR, batch_rows: int = 256,
                skip: set[str] | None = None) -> Iterator[tuple[str, "Image.Image"]]:
    """Stream (item_id, RGB PIL image) from a split's parquet, batch_rows at a time."""
    import pyarrow.parquet as pq
    from PIL import Image

    f = pq.ParquetFile(data_dir / "data" / "disjoint" / f"{_PARQUET[name]}.parquet")
    for batch in f.iter_batches(batch_size=batch_rows):
        ids = batch.column("item_id").to_pylist()
        imgs = batch.column("image").field("bytes").to_pylist()
        for item_id, data in zip(ids, imgs):
            if skip and item_id in skip:
                continue
            yield item_id, Image.open(io.BytesIO(data)).convert("RGB")


def parquet_ids(name: str, data_dir: Path = DATA_DIR) -> list[str]:
    import pyarrow.parquet as pq
    path = data_dir / "data" / "disjoint" / f"{_PARQUET[name]}.parquet"
    return pq.read_table(path, columns=["item_id"]).column("item_id").to_pylist()
