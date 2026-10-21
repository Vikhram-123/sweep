"""End-to-end fine-tuning: image encoder + compatibility head, trained with InfoNCE on outfit pairs.

    python -m compat.finetune compat/configs/finetune_method.yaml               # full run
    python -m compat.finetune compat/configs/finetune_app.yaml --ckpt-dir /content/drive/MyDrive/sweep_ckpt/app
    python -m compat.finetune compat/configs/finetune_method.yaml --smoke        # a few steps on a tiny subset
    python -m compat.finetune compat/configs/finetune_method.yaml --find-batch   # largest batch that fits

What it does (each choice is explained in FINETUNE.md):
- encoder: ResNet-18 in the papers' layout (112 px, conv1..layer3, all layers trained), or a fashion
  CLIP/SigLIP vision tower with only its last `unfreeze_blocks` transformer blocks trainable
- head: the Model A MLP; loss: InfoNCE with in-batch same-category negatives and a learned temperature
- separate learning rates for encoder and head; the encoder stays frozen for the first
  `freeze_epochs`, then warms up linearly; cosine decay to zero for both
- mixed precision on CUDA (bf16 where supported, else fp16 + GradScaler); optional gradient checkpointing
- light augmentation: random resized crop + horizontal flip (no color changes: color is a style signal)
- checkpoint every `ckpt_every` steps and every epoch (atomic writes); automatic resume from --ckpt-dir
- validation every epoch, early stopping on validation AUC, test evaluated once with the best weights
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import math
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.nn import functional as F

from .data import DATA_DIR, SPLITS, Split, item_category, load_split
from .evaluate import RUNS_DIR
from .losses import info_nce_loss
from .models import PairwiseMLP
from .train import Bank, TripletSampler, evaluate_model, plot_curves


# ----------------------------------------------------------------------------- images
class ImageStore:
    """item_id -> JPEG bytes for the requested items, read once from the parquet shards."""

    def __init__(self, item_ids: set[str] | None = None, data_dir: Path = DATA_DIR):
        import pyarrow.parquet as pq

        self.data: dict[str, bytes] = {}
        for split in ("train", "validation", "test"):
            t = pq.read_table(data_dir / "data" / "disjoint" / f"{split}.parquet")
            ids = t.column("item_id").to_pylist()
            imgs = t.column("image").combine_chunks().field("bytes").to_pylist()
            for i, b in zip(ids, imgs):
                if item_ids is None or i in item_ids:
                    self.data.setdefault(i, b)
            del t, imgs

    def image(self, item_id: str):
        from PIL import Image
        return Image.open(io.BytesIO(self.data[item_id])).convert("RGB")


def make_transforms(size: int, mean, std):
    from torchvision.transforms import v2 as T
    norm = [T.ToImage(), T.ToDtype(torch.float32, scale=True), T.Normalize(mean, std)]
    train = T.Compose([T.RandomResizedCrop(size, scale=(0.6, 1.0), ratio=(0.8, 1.25), antialias=True),
                       T.RandomHorizontalFlip()] + norm)
    test = T.Compose([T.Resize((size, size), antialias=True)] + norm)
    return train, test


class PairDataset(torch.utils.data.Dataset):
    """Rows [anchor_row, positive_row, negative_row (unused), outfit] -> two augmented images."""

    def __init__(self, rows: np.ndarray, ids: list[str], store: ImageStore, transform):
        self.rows, self.ids, self.store, self.transform = rows, ids, store, transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, k):
        a, p, _, o = self.rows[k]
        return (self.transform(self.store.image(self.ids[a])), self.transform(self.store.image(self.ids[p])),
                torch.tensor([a, p, o]))


class ItemDataset(torch.utils.data.Dataset):
    def __init__(self, ids: list[str], store: ImageStore, transform):
        self.ids, self.store, self.transform = ids, store, transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, k):
        return self.transform(self.store.image(self.ids[k]))


# ----------------------------------------------------------------------------- encoders
class Encoder(nn.Module):
    """images (B,3,H,W) -> features (B, feat_dim). `blocks`: ordered trainable stages (earliest
    first); `tail`: modules after the last block (norm/pooling/projection), trained with the blocks."""

    def __init__(self, body: nn.Module, forward_fn, blocks: list[nn.Module], tail: list[nn.Module],
                 feat_dim: int, size: int, mean, std, enable_ckpt):
        super().__init__()
        self.body, self.forward_fn = body, forward_fn
        self.blocks, self.tail = blocks, tail
        self.feat_dim, self.size, self.mean, self.std = feat_dim, size, mean, std
        self._enable_ckpt = enable_ckpt

    def forward(self, x):
        return self.forward_fn(x)

    def set_trainable(self, n_blocks: int | str) -> list[nn.Parameter]:
        """Freeze everything, then unfreeze the last n blocks (+ tail). Returns the trainable params."""
        for p in self.body.parameters():
            p.requires_grad_(False)
        chosen = self.blocks if n_blocks == "all" else self.blocks[len(self.blocks) - int(n_blocks):]
        mods = list(chosen) + (self.tail if n_blocks == "all" or int(n_blocks) > 0 else [])
        params = []
        for m in mods:
            for p in m.parameters():
                p.requires_grad_(True)
                params.append(p)
        return list({id(p): p for p in params}.values())

    def enable_checkpointing(self):
        self._enable_ckpt()


def build_encoder(name: str) -> Encoder:
    if name == "resnet18_paper":
        import torchvision
        m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        body = nn.Sequential(stem, m.layer1, m.layer2, m.layer3)
        state = {"ckpt": False}

        def fwd(x):
            x = x.contiguous(memory_format=torch.channels_last)
            if state["ckpt"] and torch.is_grad_enabled():
                x = torch.utils.checkpoint.checkpoint_sequential(body, 4, x, use_reentrant=False)
            else:
                x = body(x)
            return x.mean(dim=(2, 3))  # global average pool: 7x7x256 at 112 px -> 256

        enc = Encoder(body.to(memory_format=torch.channels_last), fwd, [stem, m.layer1, m.layer2, m.layer3], [],
                      256, 112, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), lambda: state.update(ckpt=True))
        return enc

    if name == "fashionclip":
        from transformers import CLIPVisionModelWithProjection
        m = CLIPVisionModelWithProjection.from_pretrained("patrickjohncyh/fashion-clip")
        fwd = lambda x: m(pixel_values=x).image_embeds
        return Encoder(m, fwd, list(m.vision_model.encoder.layers), [m.vision_model.post_layernorm, m.visual_projection],
                       m.config.projection_dim, 224, (0.48145466, 0.4578275, 0.40821073),
                       (0.26862954, 0.26130258, 0.27577711), m.gradient_checkpointing_enable)

    if name == "zooclaw_fashionsiglip2":
        from transformers import AutoModel
        full = AutoModel.from_pretrained("srpone/zooclaw-fashionsiglip2")
        vm = full.vision_model
        fwd = lambda x: vm(pixel_values=x).pooler_output
        size = full.config.vision_config.image_size
        return Encoder(vm, fwd, list(vm.encoder.layers), [vm.post_layernorm, vm.head],
                       full.config.vision_config.hidden_size, size, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5),
                       lambda: full.gradient_checkpointing_enable())

    if name == "marqo_fashionsiglip":
        import open_clip
        m, _, pre = open_clip.create_model_and_transforms("hf-hub:Marqo/marqo-fashionSigLIP")
        v = m.visual
        trunk = v.trunk
        size = pre.transforms[0].size if isinstance(pre.transforms[0].size, int) else pre.transforms[0].size[0]
        norm = next(t for t in pre.transforms if t.__class__.__name__ == "Normalize")
        fwd = lambda x: v(x)
        tail = [mod for mod in (getattr(trunk, "norm", None), getattr(trunk, "fc_norm", None),
                                getattr(trunk, "attn_pool", None), getattr(v, "head", None)) if mod is not None]
        out_dim = m.visual.output_dim if hasattr(m.visual, "output_dim") else m.text_projection.shape[-1]
        return Encoder(v, fwd, list(trunk.blocks), tail, out_dim, size, tuple(norm.mean), tuple(norm.std),
                       lambda: trunk.set_grad_checkpointing(True))

    raise ValueError(f"unknown encoder {name!r}")


# ----------------------------------------------------------------------------- schedule
def lr_factor(step: int, total: int, warmup: int, start: int = 0) -> float:
    """0 before `start`; linear warmup over `warmup` steps; cosine decay to 0 at `total`."""
    if step < start:
        return 0.0
    s = step - start
    if s < warmup:
        return (s + 1) / warmup
    progress = (s - warmup) / max(1, total - start - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


# ----------------------------------------------------------------------------- evaluation
@torch.no_grad()
def encode_items(encoder: Encoder, ids: list[str], store: ImageStore, transform, device, amp, batch: int, workers: int):
    encoder.eval()
    dl = torch.utils.data.DataLoader(ItemDataset(ids, store, transform), batch_size=batch, num_workers=workers,
                                     pin_memory=device.type == "cuda")
    out = []
    for x in dl:
        with amp():
            out.append(encoder(x.to(device, non_blocking=True)).float().cpu())
    return torch.cat(out)


def evaluate_split(encoder, head, split: Split, store, transform, device, amp, cfg, workers) -> dict:
    ids = sorted({i for _, items in split.compat for i in items} | {i for q in split.fitb for i in q.context + q.candidates})
    feats = encode_items(encoder, ids, store, transform, device, amp, cfg["eval_batch_size"], workers)
    tix = {t: k for k, t in enumerate(cfg["type_names"])}
    bank = Bank(index={i: k for k, i in enumerate(ids)}, emb=feats,
                types=torch.tensor([tix[item_category(i)] for i in ids]), type_names=cfg["type_names"])
    head.eval()
    return evaluate_model(head, bank, split, str(device))


def subset_split(split: Split, n_outfits: int, n_questions: int) -> Split:
    """A small slice for smoke tests; questions are spread over the file so both labels appear."""
    from dataclasses import replace
    keep = dict(list(split.outfits.items())[:n_outfits])
    pick = lambda xs: xs[::max(1, len(xs) // n_questions)][:n_questions] if n_questions else []
    return replace(split, outfits=keep, compat=pick(split.compat), fitb=pick(split.fitb))


# ----------------------------------------------------------------------------- checkpoints
def save_atomic(obj, path: Path):
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def set_rng_state(s):
    random.setstate(s["python"]); np.random.set_state(s["numpy"]); torch.set_rng_state(s["torch"])
    if s["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])


# ----------------------------------------------------------------------------- main loop
def pick_device(pref: str) -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def amp_context(device: torch.device, cfg: dict):
    """(context factory, GradScaler or None, dtype name). AMP only on CUDA: bf16 if supported, else fp16."""
    if device.type != "cuda" or not cfg.get("amp", True):
        return contextlib.nullcontext, None, "fp32"
    if torch.cuda.is_bf16_supported():
        return (lambda: torch.autocast("cuda", dtype=torch.bfloat16)), None, "bf16"
    return (lambda: torch.autocast("cuda", dtype=torch.float16)), torch.amp.GradScaler("cuda"), "fp16"


def train(cfg: dict, ckpt_dir: Path, smoke: bool = False) -> dict:
    seed = cfg["seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = pick_device(cfg.get("device", "auto"))
    amp, scaler, amp_name = amp_context(device, cfg)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    workers = cfg.get("num_workers", min(8, os.cpu_count() or 2))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out = RUNS_DIR / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    train_split, valid, test = (load_split(s) for s in SPLITS)
    if smoke:
        train_split = subset_split(train_split, 200, 0)
        valid, test = subset_split(valid, 0, 200), subset_split(test, 0, 200)
    need = train_split.item_ids | {i for s in (valid, test) for _, it in s.compat for i in it} \
        | {i for s in (valid, test) for q in s.fitb for i in q.context + q.candidates}
    print(f"loading {len(need):,} item images into memory ...", flush=True)
    store = ImageStore(need)

    # Item rows for the training sampler (types are the 11 coarse categories).
    train_ids = sorted(train_split.item_ids)
    type_names = sorted({item_category(i) for s in (train_split, valid, test) for i in s.item_ids})
    cfg["type_names"] = type_names
    tix = {t: k for k, t in enumerate(type_names)}
    bank = Bank(index={i: k for k, i in enumerate(train_ids)}, emb=torch.zeros(len(train_ids), 1),
                types=torch.tensor([tix[item_category(i)] for i in train_ids]), type_names=type_names)
    sampler = TripletSampler(train_split, bank)
    types_t = bank.types

    encoder = build_encoder(cfg["encoder"])
    head = PairwiseMLP(in_dim=encoder.feat_dim, hidden=cfg["hidden"], out_dim=cfg["out_dim"],
                       dropout=cfg["dropout"], normalize=True)
    encoder.to(device); head.to(device)
    enc_params = encoder.set_trainable(cfg["unfreeze_blocks"])
    if cfg.get("grad_checkpointing", False):
        encoder.enable_checkpointing()
    log_scale = nn.Parameter(torch.tensor(float(np.log(1.0 / cfg.get("temperature_init", 0.07))), device=device))
    opt = torch.optim.AdamW([
        {"params": enc_params, "lr": cfg["lr_encoder"], "weight_decay": cfg["weight_decay_encoder"]},
        {"params": list(head.parameters()), "lr": cfg["lr_head"], "weight_decay": cfg["weight_decay_head"]},
        {"params": [log_scale], "lr": cfg["lr_head"], "weight_decay": 0.0},
    ])
    base_lrs = [g["lr"] for g in opt.param_groups]
    train_tf, test_tf = make_transforms(encoder.size, encoder.mean, encoder.std)

    B = cfg["batch_size"]
    pairs_per_epoch = min(cfg["pairs_per_epoch"], len(sampler)) if not smoke else B * cfg.get("smoke_steps", 3)
    steps_per_epoch = pairs_per_epoch // B
    total = steps_per_epoch * cfg["max_epochs"]
    freeze_steps = int(cfg["freeze_epochs"] * steps_per_epoch)
    warm_head = max(1, int(cfg.get("warmup_epochs_head", 0.1) * steps_per_epoch))
    warm_enc = max(1, int(cfg["warmup_epochs_encoder"] * steps_per_epoch))
    n_trainable = sum(p.numel() for p in enc_params)
    print(f"{cfg['name']}: encoder {cfg['encoder']} ({encoder.size}px, {n_trainable / 1e6:.1f}M trainable params, "
          f"unfreeze {cfg['unfreeze_blocks']}), head {sum(p.numel() for p in head.parameters()) / 1e3:.0f}k, "
          f"device {device}, amp {amp_name}, batch {B} pairs, {steps_per_epoch} steps/epoch, workers {workers}",
          flush=True)

    # ---- resume
    start_epoch, step, best, bad, history = 0, 0, -1.0, 0, []
    last = ckpt_dir / "last.pt"
    if last.exists() and not smoke:
        ck = torch.load(last, map_location=device, weights_only=False)
        encoder.load_state_dict(ck["encoder"]); head.load_state_dict(ck["head"])
        log_scale.data.copy_(ck["log_scale"]); opt.load_state_dict(ck["opt"])
        if scaler is not None and ck.get("scaler"):
            scaler.load_state_dict(ck["scaler"])
        start_epoch, step, best, bad, history = ck["epoch"], ck["step"], ck["best"], ck["bad"], ck["history"]
        set_rng_state(ck["rng"])
        print(f"resumed from {last}: epoch {start_epoch}, step {step}, best val AUC {best:.4f}", flush=True)

    def checkpoint(epoch_done: int, step_in_epoch: int):
        save_atomic({"encoder": encoder.state_dict(), "head": head.state_dict(), "log_scale": log_scale.detach(),
                     "opt": opt.state_dict(), "scaler": scaler.state_dict() if scaler else None,
                     "epoch": epoch_done, "step": step, "step_in_epoch": step_in_epoch, "best": best, "bad": bad,
                     "history": history, "rng": rng_state(), "config": cfg}, last)

    for epoch in range(start_epoch, cfg["max_epochs"]):
        # Each epoch's pairs come from an RNG seeded by (seed, epoch), so a resumed epoch replays exactly.
        rng = np.random.default_rng([seed, epoch])
        rows = sampler.epoch(rng)[:pairs_per_epoch]
        skip = step - epoch * steps_per_epoch  # batches of this epoch already done before a resume
        ds = PairDataset(rows[max(0, skip) * B:], train_ids, store, train_tf)
        dl = torch.utils.data.DataLoader(ds, batch_size=B, shuffle=False, drop_last=True, num_workers=workers,
                                         pin_memory=device.type == "cuda", persistent_workers=False,
                                         prefetch_factor=4 if workers else None)
        head.train()
        t0, losses, n, seen = time.time(), 0.0, 0, 0
        for xa, xp, meta in dl:
            frozen = step < freeze_steps
            # While frozen the encoder is a fixed feature extractor: eval mode, so BatchNorm running
            # statistics don't drift either. It switches to train mode the step it unfreezes.
            encoder.train(not frozen)
            for g, base, f in zip(opt.param_groups, base_lrs,
                                  (lr_factor(step, total, warm_enc, freeze_steps), lr_factor(step, total, warm_head),
                                   lr_factor(step, total, warm_head))):
                g["lr"] = base * f
            x = torch.cat([xa, xp]).to(device, non_blocking=True)
            a, p, o = meta.T.to(device)
            with amp():
                with torch.no_grad() if frozen else contextlib.nullcontext():
                    feats = encoder(x)
                z = head(feats.float())
            za, zp = z[:B], z[B:]
            tp = types_t.to(device)[p]
            valid_neg = (tp[None] == tp[:, None]) & (o[None] != o[:, None]) & (p[None] != p[:, None])
            loss = info_nce_loss(za.float(), zp.float(), valid_neg, log_scale.exp().clamp(max=100.0))
            opt.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_([p_ for g in opt.param_groups for p_ in g["params"]], 1.0)
                scaler.step(opt); scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_([p_ for g in opt.param_groups for p_ in g["params"]], 1.0)
                opt.step()
            step += 1; seen += 1
            losses += loss.item(); n += 1
            if step % cfg.get("log_every", 50) == 0 or smoke:
                rate = seen * 2 * B / (time.time() - t0)
                print(f"  epoch {epoch + 1} step {step}/{total}  loss {loss.item():.4f}  T {1 / log_scale.exp().item():.3f}"
                      f"  {'[encoder frozen] ' if frozen else ''}{rate:.0f} img/s", flush=True)
            if not smoke and step % cfg.get("ckpt_every", 500) == 0:
                checkpoint(epoch, step - epoch * steps_per_epoch)
        val = evaluate_split(encoder, head, valid, store, test_tf, device, amp, cfg, workers)
        h = {"epoch": epoch + 1, "train_loss": losses / max(n, 1), "active_frac": float("nan"),
             "val_auc": val["compat_auc"], "val_fitb": val["fitb_acc"], "seconds": round(time.time() - t0, 1),
             "temperature": float(1 / log_scale.exp().item())}
        history.append(h)
        improved = h["val_auc"] > best
        if improved:
            best, bad = h["val_auc"], 0
            if not smoke:
                save_atomic({"encoder": encoder.state_dict(), "head": head.state_dict(),
                             "log_scale": log_scale.detach(), "config": cfg, "epoch": epoch + 1,
                             "val": val}, ckpt_dir / "best.pt")
        else:
            bad += 1
        print(f"epoch {epoch + 1}: loss {h['train_loss']:.4f}  val AUC {h['val_auc']:.4f}  FITB {100 * h['val_fitb']:.1f}%"
              f"  T {h['temperature']:.3f}  {h['seconds']}s{'  *' if improved else ''}", flush=True)
        if not smoke:
            checkpoint(epoch + 1, 0)
        if bad >= cfg["patience"] or smoke and epoch + 1 >= cfg.get("smoke_epochs", 1):
            break

    # ---- final: best weights, validation + test once
    if (ckpt_dir / "best.pt").exists() and not smoke:
        ck = torch.load(ckpt_dir / "best.pt", map_location=device, weights_only=False)
        encoder.load_state_dict(ck["encoder"]); head.load_state_dict(ck["head"])
    best_epoch = max(history, key=lambda r: r["val_auc"])["epoch"]
    with open(out / "history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0])); w.writeheader(); w.writerows(history)
    plot_curves(history, out / "curves.png", cfg["name"], mid_title="(n/a)")
    results = {}
    for split in (valid, test):
        m = {"name": cfg["name"], "split": split.name,
             **evaluate_split(encoder, head, split, store, test_tf, device, amp, cfg, workers),
             "best_epoch": best_epoch, "notes": cfg.get("notes", "") + (" [SMOKE TEST]" if smoke else "")}
        (out / f"metrics_{split.name}.json").write_text(json.dumps(m, indent=2))
        shutil.copy(out / f"metrics_{split.name}.json", ckpt_dir / f"metrics_{split.name}.json")
        results[split.name] = m
        print(f"{cfg['name']:<28} {split.name:<5}  AUC {m['compat_auc']:.3f}  FITB {100 * m['fitb_acc']:.1f}%", flush=True)
    return results


def find_batch(cfg: dict) -> int:
    """Largest batch (pairs) whose forward+backward fits, halving from cfg['batch_size'] on OOM."""
    device = pick_device(cfg.get("device", "auto"))
    amp, _, _ = amp_context(device, cfg)
    encoder = build_encoder(cfg["encoder"]).to(device)
    params = encoder.set_trainable(cfg["unfreeze_blocks"])
    if cfg.get("grad_checkpointing", False):
        encoder.enable_checkpointing()
    head = PairwiseMLP(in_dim=encoder.feat_dim, hidden=cfg["hidden"], out_dim=cfg["out_dim"]).to(device)
    B = cfg["batch_size"]
    while B >= 8:
        try:
            x = torch.randn(2 * B, 3, encoder.size, encoder.size, device=device)
            with amp():
                z = head(encoder(x).float())
            z.square().mean().backward()
            encoder.zero_grad(set_to_none=True); head.zero_grad(set_to_none=True)
            peak = torch.cuda.max_memory_allocated() / 2 ** 30 if device.type == "cuda" else float("nan")
            print(f"batch {B} pairs ({2 * B} images) fits; peak memory {peak:.1f} GB", flush=True)
            return B
        except torch.OutOfMemoryError:
            print(f"batch {B} pairs: out of memory", flush=True)
            del x
            torch.cuda.empty_cache()
            B //= 2
    raise SystemExit("even batch 8 does not fit")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("config", type=Path)
    ap.add_argument("--ckpt-dir", type=Path, default=None, help="where last.pt/best.pt live (e.g. Google Drive)")
    ap.add_argument("--smoke", action="store_true", help="a few steps on a tiny subset, to check it runs")
    ap.add_argument("--find-batch", action="store_true")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text())
    for kv in args.set:
        k, v = kv.split("=", 1)
        cfg[k] = yaml.safe_load(v)
    if args.find_batch:
        find_batch(cfg)
        return
    if args.smoke:
        cfg = {**cfg, "name": "_smoke_" + cfg["name"], "batch_size": cfg.get("smoke_batch_size", 8),
               "eval_batch_size": 16, "num_workers": 0}
    ckpt = args.ckpt_dir or Path("checkpoints") / cfg["name"]
    train(cfg, ckpt, smoke=args.smoke)


if __name__ == "__main__":
    main()
