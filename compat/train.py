"""Train a compatibility model on frozen FashionCLIP embeddings.

    python -m compat.train compat/configs/model_a.yaml

Writes results/runs/<name>/: config.yaml, history.csv, curves.png, model.pt, metrics_{valid,test}.json
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
from .losses import triplet_loss
from .models import build_model


# ----------------------------------------------------------------------------- data on tensors
@dataclass
class Bank:
    """All items of all splits as one matrix, plus each item's type index."""
    index: dict[str, int]
    emb: torch.Tensor          # (n, 512)
    types: torch.Tensor        # (n,) long
    type_names: list[str]

    @classmethod
    def build(cls) -> "Bank":
        embs = load_embeddings(SPLITS)
        ids = sorted(embs)
        type_names = sorted({item_category(i) for i in ids})
        tix = {t: k for k, t in enumerate(type_names)}
        return cls(index={i: k for k, i in enumerate(ids)},
                   emb=torch.from_numpy(np.stack([embs[i] for i in ids])),
                   types=torch.tensor([tix[item_category(i)] for i in ids]),
                   type_names=type_names)


class TripletSampler:
    """(anchor, positive, negative) item rows: anchor/positive from the same training outfit,
    negative a random training item of the positive's type (category-aware negatives)."""

    def __init__(self, split: Split, bank: Bank):
        self.outfits = [np.array([bank.index[i] for i in items]) for items in split.outfits.values()]
        train_rows = np.unique(np.concatenate(self.outfits))
        types = bank.types.numpy()
        self.by_type = {t: train_rows[types[train_rows] == t] for t in np.unique(types[train_rows])}
        self.types = types
        a, p = [], []
        for rows in self.outfits:
            i, j = np.meshgrid(rows, rows, indexing="ij")
            keep = i != j
            a.append(i[keep])
            p.append(j[keep])
        self.anchor, self.positive = np.concatenate(a), np.concatenate(p)

    def __len__(self) -> int:
        return len(self.anchor)

    def epoch(self, rng: np.random.Generator) -> np.ndarray:
        """(len, 3) shuffled triplets with freshly drawn negatives."""
        neg = np.empty_like(self.positive)
        ptypes = self.types[self.positive]
        for t, pool in self.by_type.items():
            m = ptypes == t
            neg[m] = pool[rng.integers(len(pool), size=m.sum())]
        clash = neg == self.positive  # vanishingly rare; redraw once
        neg[clash] = [self.by_type[t][rng.integers(len(self.by_type[t]))] for t in ptypes[clash]]
        trip = np.stack([self.anchor, self.positive, neg], axis=1)
        return trip[rng.permutation(len(trip))]


# ----------------------------------------------------------------------------- evaluation
@torch.no_grad()
def outfit_scores(model, bank: Bank, outfits: list[list[str]], device: str, chunk: int = 65536) -> np.ndarray:
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
        d.append((za - zb).norm(dim=-1).cpu().numpy())
    d = np.concatenate(d)
    sums = np.bincount(oi, weights=-d, minlength=len(outfits))
    counts = np.bincount(oi, minlength=len(outfits))
    return sums / np.maximum(counts, 1)


def evaluate_model(model, bank: Bank, split: Split, device: str) -> dict:
    model.eval()
    labels = [l for l, _ in split.compat]
    auc = roc_auc_score(labels, outfit_scores(model, bank, [items for _, items in split.compat], device))
    fitb_outfits = [q.context + [c] for q in split.fitb for c in q.candidates]
    s = outfit_scores(model, bank, fitb_outfits, device).reshape(len(split.fitb), 4)
    acc = float(np.mean(s.argmax(1) == np.array([q.answer for q in split.fitb])))
    return {"compat_auc": float(auc), "fitb_acc": acc}


# ----------------------------------------------------------------------------- training
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def plot_curves(history: list[dict], path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    ax[0].plot(ep, [h["train_loss"] for h in history], label="train loss")
    ax[0].set_title("triplet loss")
    ax[1].plot(ep, [h["active_frac"] for h in history], color="tab:orange")
    ax[1].set_title("triplets violating the margin")
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


def train(cfg: dict) -> dict:
    seed_everything(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    device = cfg.get("device", "cpu")
    out = RUNS_DIR / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    bank = Bank.build()
    train_split, valid, test = (load_split(s) for s in SPLITS)
    sampler = TripletSampler(train_split, bank)
    model = build_model(cfg, n_types=len(bank.type_names)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    print(f"{cfg['name']}: {sum(p.numel() for p in model.parameters()):,} params, "
          f"{len(sampler):,} triplets/epoch, device {device}", flush=True)

    history, best, best_state, bad = [], -1.0, None, 0
    for epoch in range(1, cfg["max_epochs"] + 1):
        model.train()
        t0, losses, active, n = time.time(), 0.0, 0.0, 0
        trips = torch.from_numpy(sampler.epoch(rng))
        for s in range(0, len(trips), cfg["batch_size"]):
            a, p, ng = trips[s:s + cfg["batch_size"]].T
            ta, tp = bank.types[a].to(device), bank.types[p].to(device)
            za = model(bank.emb[a].to(device), ta, tp)
            zp = model(bank.emb[p].to(device), tp, ta)
            zn = model(bank.emb[ng].to(device), tp, ta)  # negative has the positive's type
            loss = triplet_loss(za, zp, zn, cfg["margin"])
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():  # diagnostic, independent of how the loss is written
                gap = (za - zn).norm(dim=-1) - (za - zp).norm(dim=-1)
                active += float((gap < cfg["margin"]).sum())
            losses += loss.item() * len(a)
            n += len(a)
        val = evaluate_model(model, bank, valid, device)
        h = {"epoch": epoch, "train_loss": losses / n, "active_frac": active / n,
             "val_auc": val["compat_auc"], "val_fitb": val["fitb_acc"], "seconds": round(time.time() - t0, 1)}
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

    with open(out / "history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0]))
        w.writeheader()
        w.writerows(history)
    plot_curves(history, out / "curves.png", cfg["name"])

    model.load_state_dict(best_state)
    torch.save({"state_dict": best_state, "config": cfg, "type_names": bank.type_names}, out / "model.pt")
    best_epoch = max(history, key=lambda h: h["val_auc"])["epoch"]
    results = {}
    for split in (valid, test):  # test is touched exactly once, with the early-stopped weights
        m = {"name": cfg["name"], "split": split.name, **evaluate_model(model, bank, split, device),
             "best_epoch": best_epoch, "notes": cfg.get("notes", "")}
        (out / f"metrics_{split.name}.json").write_text(json.dumps(m, indent=2))
        results[split.name] = m
        print(f"{cfg['name']:<28} {split.name:<5}  AUC {m['compat_auc']:.3f}  FITB {100 * m['fitb_acc']:.1f}%")
    return results


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
    train(cfg)


if __name__ == "__main__":
    main()
