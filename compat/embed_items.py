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
ENCODERS = {
    "fashionclip": "patrickjohncyh/fashion-clip",   # CLIP ViT-B/32 fine-tuned on fashion products
    "clip": "openai/clip-vit-base-patch32",          # the original OpenAI CLIP it was fine-tuned from
    "resnet18": "microsoft/resnet-18",               # ImageNet-1k ResNet-18 (HF port; used in the first ablation)
    # torchvision IMAGENET1K_V1 -- the exact weights the published papers load (model_zoo resnet18):
    "tv_resnet18": "torchvision:resnet18:224:layer4",          # standard: 224 px, full network, 512-d
    "tv_resnet18_paper": "torchvision:resnet18:112:layer3",    # as in Vasileva et al.'s code: 112 px, layer4 dropped, 256-d
}


class TorchvisionResNet18:
    """torchvision ImageNet ResNet-18 (IMAGENET1K_V1), optionally truncated after layer3 like the
    reference implementation (mvasil/fashion-compatibility Resnet_18.py: 112 px input, no layer4,
    7x7 average pool -> 256-d). Global-average-pooled features, L2-normalised."""

    def __init__(self, size: int = 224, last: str = "layer4", device: str | None = None):
        import torch
        import torchvision
        from torchvision.transforms import v2 as T

        from sweep.device import align_weights, get_device

        self.torch = torch
        self.device = device or get_device()
        m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        layers = [m.conv1, m.bn1, m.relu, m.maxpool, m.layer1, m.layer2, m.layer3]
        if last == "layer4":
            layers.append(m.layer4)
        self.model = align_weights(torch.nn.Sequential(*layers, torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten())
                                   .to(self.device), self.device).eval()
        self.transform = T.Compose([T.ToImage(), T.Resize((size, size), antialias=True), T.ToDtype(torch.float32, scale=True),
                                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def embed_images(self, images: list[np.ndarray], batch_size: int = 64) -> np.ndarray:
        from sweep.embed import l2norm

        out = []
        for i in range(0, len(images), batch_size):
            x = self.torch.stack([self.transform(im) for im in images[i:i + batch_size]]).to(self.device)
            with self.torch.inference_mode():
                out.append(self.model(x).float().cpu().numpy())
        return l2norm(np.concatenate(out))


def out_dir(encoder: str = "fashionclip") -> Path:
    return OUT_DIR if encoder == "fashionclip" else OUT_DIR / encoder


class ResNetEmbedder:
    """Frozen ImageNet ResNet-18: global-average-pooled 512-d features, L2-normalised."""

    def __init__(self, model_id: str = ENCODERS["resnet18"], device: str | None = None):
        import torch
        from transformers import AutoImageProcessor, ResNetModel

        from sweep.device import align_weights, get_device

        self.torch = torch
        self.device = device or get_device()
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = align_weights(ResNetModel.from_pretrained(model_id).to(self.device), self.device).eval()

    def embed_images(self, images: list[np.ndarray], batch_size: int = 32) -> np.ndarray:
        from PIL import Image

        from sweep.embed import l2norm

        out = []
        for i in range(0, len(images), batch_size):
            batch = [Image.fromarray(im) for im in images[i:i + batch_size]]
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
            with self.torch.inference_mode():
                feats = self.model(**inputs).pooler_output.flatten(1)
            out.append(feats.float().cpu().numpy())
        return l2norm(np.concatenate(out))


def make_embedder(encoder: str):
    if encoder == "resnet18":
        return ResNetEmbedder()
    if ENCODERS[encoder].startswith("torchvision:"):
        _, _, size, last = ENCODERS[encoder].split(":")
        return TorchvisionResNet18(int(size), last)
    from sweep.embed import FashionClipEmbedder
    return FashionClipEmbedder(model_id=ENCODERS[encoder])


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


def embed_split(split: str, embedder, shard: int, batch: int, workers: int, encoder: str = "fashionclip") -> None:
    from sweep.device import free_memory

    root = out_dir(encoder)
    shard_dir = root / split
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
        with_colors = encoder == "fashionclip"
        colors = pool.map(_color, imgs) if with_colors else iter([{}] * len(imgs))  # overlaps the GPU pass
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
    merge(split, ids, encoder)


def merge(split: str, ids: list[str], encoder: str = "fashionclip") -> None:
    root = out_dir(encoder)
    parts = [np.load(p) for p in sorted((root / split).glob("[0-9]*.npz"))]
    all_ids = np.concatenate([p["ids"] for p in parts])
    emb = np.concatenate([p["emb"] for p in parts])
    colors = dict(zip(all_ids.tolist(), (json.loads(c) for p in parts for c in p["colors"])))
    assert sorted(all_ids.tolist()) == sorted(ids), "shards don't cover the split exactly"
    np.savez(root / f"{split}.npz", ids=all_ids, emb=emb)
    if encoder == "fashionclip":
        (root / f"{split}_colors.json").write_text(json.dumps(colors))
    print(f"[{split}] merged -> {root / split}.npz  {emb.shape}", flush=True)


def load_embeddings(splits=SPLITS, encoder: str = "fashionclip") -> dict[str, np.ndarray]:
    """item_id -> embedding for the given splits."""
    out = {}
    for s in splits:
        d = np.load(out_dir(encoder) / f"{s}.npz")
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
    ap.add_argument("--encoder", choices=list(ENCODERS), default="fashionclip")
    args = ap.parse_args(argv)

    embedder = make_embedder(args.encoder)
    print("encoder:", args.encoder, "device:", embedder.device, flush=True)
    for split in args.split or ["valid", "test", "train"]:
        embed_split(split, embedder, args.shard, args.batch, args.workers, args.encoder)


if __name__ == "__main__":
    main()
