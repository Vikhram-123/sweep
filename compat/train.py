"""Train a compatibility model on frozen item embeddings.

    python -m compat.train compat/configs/model_a.yaml [--set key=value ...]

Writes results/runs/<name>/: config.yaml, history.csv, curves.png, model.pt, metrics_{valid,test}.json

Config options beyond the model's own (defaults reproduce Model A):
    encoder    fashionclip | clip | resnet18        which frozen image embeddings to train on
    features   image | concat | mean                 image only, or image + FashionCLIP title-text
                                                     embedding (concatenated, or averaged)
    negatives  random | knn | batch_semihard | batch_hard
               random: a random training item of the positive's type
               knn: with probability knn_frac, one of the positive's knn_k nearest same-type
                    neighbours (input space, cos <= knn_max_cos so near-duplicates aren't used)
               batch_*: mined inside each batch among same-type items from other outfits
    distance   l2 | sq                               used for both the loss and evaluation
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score

from .data import SPLITS, Split, item_category, load_split
from .embed_items import load_embeddings
from .evaluate import RUNS_DIR
from .losses import info_nce_loss, squared_triplet_loss, triplet_loss
from .models import build_model

TEXT_ENCODER = "text"  # FashionCLIP text embeddings of item titles, written by compat.embed_text


# ----------------------------------------------------------------------------- data on tensors
@dataclass
class Bank:
    """All items of all splits as one matrix, plus each item's type index."""
    index: dict[str, int]
    emb: torch.Tensor          # (n, d)
    types: torch.Tensor        # (n,) long
    type_names: list[str]

    @classmethod
    def build(cls, encoder: str = "fashionclip", features: str = "image") -> "Bank":
        embs = load_embeddings(SPLITS, encoder)
        ids = sorted(embs)
        x = np.stack([embs[i] for i in ids])
        if features != "image":
            txt = load_embeddings(SPLITS, TEXT_ENCODER)
            t = np.stack([txt[i] for i in ids])
            if features == "concat":
                x = np.concatenate([x, t], axis=1)
            elif features == "mean":
                x = x + t
                x /= np.linalg.norm(x, axis=1, keepdims=True)
            else:
                raise ValueError(f"unknown features {features!r}")
        type_names = sorted({item_category(i) for i in ids})
        tix = {t: k for k, t in enumerate(type_names)}
        return cls(index={i: k for k, i in enumerate(ids)},
                   emb=torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)),
                   types=torch.tensor([tix[item_category(i)] for i in ids]),
                   type_names=type_names)


def same_type_knn(bank: Bank, rows: np.ndarray, k: int, max_cos: float, chunk: int = 2048) -> np.ndarray:
    """(len(bank), k) nearest same-type neighbours among `rows` (other rows are left as -1)."""
    emb = torch.nn.functional.normalize(bank.emb, dim=-1)
    types = bank.types.numpy()
    out = np.full((len(bank.emb), k), -1, np.int64)
    for t in np.unique(types[rows]):
        pool = rows[types[rows] == t]
        e = emb[pool]
        for s in range(0, len(pool), chunk):
            sim = e[s:s + chunk] @ e.T
            sim[sim > max_cos] = -2.0  # itself and near-duplicates: not a real negative
            idx = sim.topk(min(k, len(pool) - 1), dim=1).indices.numpy()
            out[pool[s:s + chunk], :idx.shape[1]] = pool[idx]
    return out


class TripletSampler:
    """(anchor, positive, negative, outfit) rows: anchor/positive from the same training outfit,
    negative an item of the positive's type (category-aware negatives)."""

    def __init__(self, split: Split, bank: Bank, negatives: str = "random", knn_k: int = 50,
                 knn_max_cos: float = 0.95, knn_frac: float = 1.0):
        self.outfits = [np.array([bank.index[i] for i in items]) for items in split.outfits.values()]
        train_rows = np.unique(np.concatenate(self.outfits))
        types = bank.types.numpy()
        self.by_type = {t: train_rows[types[train_rows] == t] for t in np.unique(types[train_rows])}
        self.types = types
        a, p, o = [], [], []
        for k, rows in enumerate(self.outfits):
            i, j = np.meshgrid(rows, rows, indexing="ij")
            keep = i != j
            a.append(i[keep])
            p.append(j[keep])
            o.append(np.full(keep.sum(), k))
        self.anchor, self.positive, self.outfit = np.concatenate(a), np.concatenate(p), np.concatenate(o)
        self.negatives, self.knn_frac = negatives, knn_frac
        self.knn = same_type_knn(bank, train_rows, knn_k, knn_max_cos) if negatives == "knn" else None

    def __len__(self) -> int:
        return len(self.anchor)

    def epoch(self, rng: np.random.Generator) -> np.ndarray:
        """(len, 4) shuffled [anchor, positive, negative, outfit] with freshly drawn negatives."""
        neg = np.empty_like(self.positive)
        ptypes = self.types[self.positive]
        for t, pool in self.by_type.items():
            m = ptypes == t
            neg[m] = pool[rng.integers(len(pool), size=m.sum())]
        if self.knn is not None:
            use = rng.random(len(neg)) < self.knn_frac
            cand = self.knn[self.positive[use], rng.integers(self.knn.shape[1], size=use.sum())]
            ok = cand >= 0
            idx = np.flatnonzero(use)[ok]
            neg[idx] = cand[ok]
        clash = neg == self.positive  # vanishingly rare; redraw once
        neg[clash] = [self.by_type[t][rng.integers(len(self.by_type[t]))] for t in ptypes[clash]]
        trip = np.stack([self.anchor, self.positive, neg, self.outfit], axis=1)
        return trip[rng.permutation(len(trip))]


def pair_dist(za: torch.Tensor, zb: torch.Tensor, distance: str) -> torch.Tensor:
    d2 = ((za - zb) ** 2).sum(-1)
    return d2 if distance == "sq" else d2.clamp_min(1e-12).sqrt()


def mine_batch_negatives(za, zp, zn, ptype, outfit, rows, distance: str, mode: str) -> torch.Tensor:
    """Replace each row's negative with one mined from the batch: candidates are every positive and
    negative in the batch with the row's positive type, from a different outfit, not the positive
    item itself. semihard = closest candidate farther than the positive (else the drawn negative);
    hard = closest candidate."""
    cand_z = torch.cat([zp, zn])
    cand_t = torch.cat([ptype, ptype])
    cand_o = torch.cat([outfit, torch.full_like(outfit, -1)])
    cand_r = torch.cat([rows[:, 1], rows[:, 2]])
    with torch.no_grad():
        d = torch.cdist(za, cand_z) ** (2 if distance == "sq" else 1)
        valid = (cand_t[None, :] == ptype[:, None]) & (cand_o[None, :] != outfit[:, None]) \
            & (cand_r[None, :] != rows[:, 1:2])
        if mode == "batch_semihard":
            d_ap = pair_dist(za, zp, distance)[:, None]
            valid &= d > d_ap
        d = d.masked_fill(~valid, float("inf"))
        best = d.argmin(1)
        has = torch.isfinite(d.gather(1, best[:, None])).squeeze(1)
    mined = cand_z[best]
    return torch.where(has[:, None], mined, zn)


# ----------------------------------------------------------------------------- evaluation
@torch.no_grad()
def outfit_scores(model, bank: Bank, outfits: list[list[str]], device: str, chunk: int = 65536,
                  distance: str = "l2") -> np.ndarray:
    """Mean pairwise score (-distance) per outfit, all pairs batched through the model."""
    oi, a, b = [], [], []
    for k, items in enumerate(outfits):
        rows = [bank.index[i] for i in items]
        for x in range(len(rows)):
            for y in range(x + 1, len(rows)):
                oi.append(k), a.append(rows[x]), b.append(rows[y])
    oi, a, b = np.array(oi), torch.tensor(a), torch.tensor(b)
    d = []
    for s in range(0, len(a), chunk):
        ra, rb = a[s:s + chunk], b[s:s + chunk]
        ta, tb = bank.types[ra].to(device), bank.types[rb].to(device)
        za = model(bank.emb[ra].to(device), ta, tb)
        zb = model(bank.emb[rb].to(device), tb, ta)
        dd = (za - zb).norm(dim=-1)
        d.append((dd ** 2 if distance == "sq" else dd).cpu().numpy())
    d = np.concatenate(d)
    sums = np.bincount(oi, weights=-d, minlength=len(outfits))
    counts = np.bincount(oi, minlength=len(outfits))
    return sums / np.maximum(counts, 1)


def evaluate_model(model, bank: Bank, split: Split, device: str, distance: str = "l2") -> dict:
    model.eval()
    labels = [l for l, _ in split.compat]
    auc = roc_auc_score(labels, outfit_scores(model, bank, [items for _, items in split.compat], device,
                                              distance=distance))
    fitb_outfits = [q.context + [c] for q in split.fitb for c in q.candidates]
    s = outfit_scores(model, bank, fitb_outfits, device, distance=distance).reshape(len(split.fitb), 4)
    acc = float(np.mean(s.argmax(1) == np.array([q.answer for q in split.fitb])))
    return {"compat_auc": float(auc), "fitb_acc": acc}


# ----------------------------------------------------------------------------- training
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def plot_curves(history: list[dict], path: Path, title: str,
                mid_title: str = "triplets violating the margin") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    ax[0].plot(ep, [h["train_loss"] for h in history], label="train loss")
    ax[0].set_title("training loss")
    ax[1].plot(ep, [h["active_frac"] for h in history], color="tab:orange")
    ax[1].set_title(mid_title)
    ax[2].plot(ep, [h["val_auc"] for h in history], label="val compat AUC")
    ax[2].plot(ep, [h["val_fitb"] for h in history], label="val FITB acc")
    ax[2].set_title("validation")
    ax[2].legend()
    for a in ax:
        a.set_xlabel("epoch")
        a.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def finish_history(history: list[dict], out: Path, title: str,
                   mid_title: str = "triplets violating the margin") -> None:
    with open(out / "history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0]))
        w.writeheader()
        w.writerows(history)
    plot_curves(history, out / "curves.png", title, mid_title)


def finish(cfg, model, bank, best_state, history, out, valid, test, device, distance) -> dict:
    """Save history/curves/checkpoint, then evaluate the early-stopped weights on valid and test."""
    finish_history(history, out, cfg["name"])

    model.load_state_dict(best_state)
    torch.save({"state_dict": best_state, "config": cfg, "type_names": bank.type_names}, out / "model.pt")
    best_epoch = max(history, key=lambda h: h["val_auc"])["epoch"]
    results = {}
    for split in (valid, test):  # test is touched exactly once, with the early-stopped weights
        m = {"name": cfg["name"], "split": split.name, **evaluate_model(model, bank, split, device, distance),
             "best_epoch": best_epoch, "notes": cfg.get("notes", "")}
        (out / f"metrics_{split.name}.json").write_text(json.dumps(m, indent=2))
        results[split.name] = m
        print(f"{cfg['name']:<28} {split.name:<5}  AUC {m['compat_auc']:.3f}  FITB {100 * m['fitb_acc']:.1f}%",
              flush=True)
    return results


def train(cfg: dict) -> dict:
    seed_everything(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    device = cfg.get("device", "cpu")
    distance = cfg.get("distance", "l2")
    negatives = cfg.get("negatives", "random")
    out = RUNS_DIR / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    bank = Bank.build(cfg.get("encoder", "fashionclip"), cfg.get("features", "image"))
    train_split, valid, test = (load_split(s) for s in SPLITS)
    sampler = TripletSampler(train_split, bank, negatives, cfg.get("knn_k", 50), cfg.get("knn_max_cos", 0.95),
                             cfg.get("knn_frac", 1.0))
    model = build_model({**cfg, "in_dim": bank.emb.shape[1]}, n_types=len(bank.type_names)).to(device)
    if negatives.startswith("batch_") and cfg["model"] != "pairwise":
        raise ValueError("batch mining is implemented for the pairwise model only")
    loss_fn = squared_triplet_loss if distance == "sq" else triplet_loss
    loss_kind = cfg.get("loss", "triplet")
    if loss_kind == "infonce" and not cfg.get("normalize", True):
        raise ValueError("InfoNCE uses cosine similarities: set normalize: true")
    # InfoNCE temperature, learned in log space (CLIP-style); scale = 1/temperature, capped at 100.
    log_scale = torch.nn.Parameter(torch.tensor(float(np.log(1.0 / cfg.get("temperature_init", 0.07)))),
                                   requires_grad=cfg.get("learn_temperature", True))
    # Type-pair projections (Model B) can get their own learning rate and no weight decay
    # (config proj_lr_mult); with one shared rate they barely move away from their init.
    proj = [p for n, p in model.named_parameters() if n in ("masks", "mats")]
    rest = [p for n, p in model.named_parameters() if n not in ("masks", "mats")]
    groups = [{"params": rest}]
    if proj:
        groups.append({"params": proj, "lr": cfg["lr"] * cfg.get("proj_lr_mult", 1.0),
                       "weight_decay": cfg["weight_decay"] if cfg.get("proj_lr_mult", 1.0) == 1.0 else 0.0})
    if loss_kind == "infonce" and log_scale.requires_grad:
        groups.append({"params": [log_scale], "weight_decay": 0.0})
    opt = torch.optim.Adam(groups, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    print(f"{cfg['name']}: {sum(p.numel() for p in model.parameters()):,} params, input {bank.emb.shape[1]}-d, "
          f"{len(sampler):,} triplets/epoch, negatives {negatives}, distance {distance}, device {device}", flush=True)

    history, best, best_state, bad = [], -1.0, None, 0
    for epoch in range(1, cfg["max_epochs"] + 1):
        model.train()
        t0, losses, active, n = time.time(), 0.0, 0.0, 0
        trips = torch.from_numpy(sampler.epoch(rng))
        for s in range(0, len(trips), cfg["batch_size"]):
            rows = trips[s:s + cfg["batch_size"]]
            a, p, ng, outfit = rows.T
            ta, tp = bank.types[a].to(device), bank.types[p].to(device)
            za = model(bank.emb[a].to(device), ta, tp)
            zp = model(bank.emb[p].to(device), tp, ta)
            zn = model(bank.emb[ng].to(device), tp, ta)  # negative has the positive's type
            if loss_kind == "infonce":
                outfit_d, p_d = outfit.to(device), p.to(device)
                valid_neg = (tp[None, :] == tp[:, None]) & (outfit_d[None, :] != outfit_d[:, None]) \
                    & (p_d[None, :] != p_d[:, None])
                scale = log_scale.exp().clamp(max=100.0)
                loss = info_nce_loss(za, zp, valid_neg, scale)
            else:
                if negatives.startswith("batch_"):
                    zn = mine_batch_negatives(za, zp, zn, tp, outfit.to(device), rows.to(device), distance, negatives)
                loss = loss_fn(za, zp, zn, cfg["margin"])
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():  # diagnostic, independent of how the loss is written
                if loss_kind == "infonce":  # anchor ranks some valid in-batch negative above its positive
                    sim = za @ zp.T
                    beaten = (sim.masked_fill(~valid_neg, -2.0).max(1).values >= sim.diagonal())
                    active += float(beaten.sum())
                else:
                    gap = pair_dist(za, zn, distance) - pair_dist(za, zp, distance)
                    active += float((gap < cfg["margin"]).sum())
            losses += loss.item() * len(a)
            n += len(a)
        val = evaluate_model(model, bank, valid, device, distance)
        h = {"epoch": epoch, "train_loss": losses / n, "active_frac": active / n,
             "val_auc": val["compat_auc"], "val_fitb": val["fitb_acc"], "seconds": round(time.time() - t0, 1)}
        if loss_kind == "infonce":
            h["temperature"] = float(1.0 / log_scale.exp().clamp(max=100.0))
        history.append(h)
        improved = h["val_auc"] > best
        if improved:
            best, bad = h["val_auc"], 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        print(f"epoch {epoch:3d}  loss {h['train_loss']:.4f}  active {h['active_frac']:.3f}  "
              f"val AUC {h['val_auc']:.4f}  FITB {100 * h['val_fitb']:.1f}%  {h['seconds']}s"
              f"{'  *' if improved else ''}", flush=True)
        if bad >= cfg["patience"]:
            print(f"early stop: no val AUC improvement for {cfg['patience']} epochs", flush=True)
            break
    return finish(cfg, model, bank, best_state, history, out, valid, test, device, distance)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Train a compatibility model")
    ap.add_argument("config", type=Path)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="override config values, e.g. --set margin=0.3 name=model_a_m03")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text())
    for kv in args.set:
        k, v = kv.split("=", 1)
        cfg[k] = yaml.safe_load(v)
    if cfg["model"] == "outfit":
        from .outfit_train import train_outfit_model
        train_outfit_model(cfg)
    elif cfg["model"] == "transformer":
        from .transformer import train_transformer
        train_transformer(cfg)
    else:
        train(cfg)


if __name__ == "__main__":
    main()
