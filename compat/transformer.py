"""Outfit transformer (first version): scores a whole outfit at once instead of averaging pairs.

    python -m compat.train compat/configs/transformer.yaml

Each item is a token (projected frozen embedding + learned type embedding); a [CLS] token reads
the set through self-attention and a small head turns it into one logit: real outfit vs fake.
No positional encoding, so the score doesn't depend on item order. Trained with binary cross-
entropy on real training outfits vs. fakes made by swapping items for random same-type items
(all items, like the test negatives, or just one, like FITB). See TRANSFORMER.md.
"""
from __future__ import annotations

import time

import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score
from torch import nn

from .data import SPLITS, Split, load_split
from .evaluate import RUNS_DIR
from .train import Bank, finish_history, seed_everything


class OutfitTransformer(nn.Module):
    def __init__(self, in_dim: int, n_types: int, d_model: int = 128, heads: int = 4, layers: int = 2,
                 ff: int = 256, dropout: float = 0.1, **_):
        super().__init__()
        # LayerNorm puts item tokens on the same scale as the type embeddings (norm ~11 at init).
        # Without it, unit-norm inputs project to norm ~0.45: the type embedding (identical for
        # real and fake outfits) drowns out the item signal and training sat at chance.
        self.item_proj = nn.Sequential(nn.Linear(in_dim, d_model), nn.LayerNorm(d_model))
        self.type_emb = nn.Embedding(n_types, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        layer = nn.TransformerEncoderLayer(d_model, heads, ff, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1))

    def forward(self, x: torch.Tensor, types: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        """x (B, L, in_dim), types (B, L), pad (B, L) True where padding -> (B,) logits."""
        tok = self.item_proj(x) + self.type_emb(types)
        tok = torch.cat([self.cls.expand(len(x), -1, -1), tok], dim=1)
        pad = torch.cat([torch.zeros(len(x), 1, dtype=torch.bool, device=pad.device), pad], dim=1)
        h = self.encoder(tok, src_key_padding_mask=pad)
        return self.head(self.norm(h[:, 0])).squeeze(-1)


def pack(rows: list[np.ndarray], bank: Bank, device: str):
    """List of item-row arrays -> padded (x, types, pad) tensors."""
    L = max(len(r) for r in rows)
    idx = np.zeros((len(rows), L), np.int64)
    pad = np.ones((len(rows), L), bool)
    for k, r in enumerate(rows):
        idx[k, :len(r)] = r
        pad[k, :len(r)] = False
    idx_t = torch.from_numpy(idx)
    return bank.emb[idx_t].to(device), bank.types[idx_t].to(device), torch.from_numpy(pad).to(device)


@torch.no_grad()
def score_outfits(model, bank: Bank, outfits: list[list[str]], device: str, batch: int = 1024) -> np.ndarray:
    model.eval()
    rows = [np.array([bank.index[i] for i in items]) for items in outfits]
    out = []
    for s in range(0, len(rows), batch):
        out.append(model(*pack(rows[s:s + batch], bank, device)).float().cpu().numpy())
    return np.concatenate(out)


def evaluate_transformer(model, bank: Bank, split: Split, device: str) -> dict:
    labels = [l for l, _ in split.compat]
    auc = roc_auc_score(labels, score_outfits(model, bank, [items for _, items in split.compat], device))
    s = score_outfits(model, bank, [q.context + [c] for q in split.fitb for c in q.candidates], device)
    acc = float(np.mean(s.reshape(len(split.fitb), 4).argmax(1) == np.array([q.answer for q in split.fitb])))
    return {"compat_auc": float(auc), "fitb_acc": acc}


class FakeOutfits:
    """Fake outfits from real ones: swap every item (like the test negatives) or one item (like FITB)
    for a random training item of the same type."""

    def __init__(self, outfits: list[np.ndarray], bank: Bank):
        types = bank.types.numpy()
        rows = np.unique(np.concatenate(outfits))
        self.by_type = {t: rows[types[rows] == t] for t in np.unique(types[rows])}
        self.types = types

    def swap(self, outfit: np.ndarray, rng, single: bool) -> np.ndarray:
        fake = outfit.copy()
        which = [rng.integers(len(outfit))] if single else range(len(outfit))
        for k in which:
            pool = self.by_type[self.types[outfit[k]]]
            fake[k] = pool[rng.integers(len(pool))]
        return fake


def train_transformer(cfg: dict) -> dict:
    seed_everything(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    device = cfg.get("device", "cpu")
    out = RUNS_DIR / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    bank = Bank.build(cfg.get("encoder", "fashionclip"), cfg.get("features", "image"))
    train_split, valid, test = (load_split(s) for s in SPLITS)
    real = [np.array([bank.index[i] for i in items]) for items in train_split.outfits.values()]
    fakes = FakeOutfits(real, bank)
    model = OutfitTransformer(bank.emb.shape[1], len(bank.type_names), **cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    bce = nn.BCEWithLogitsLoss()
    print(f"{cfg['name']}: {sum(p.numel() for p in model.parameters()):,} params, input {bank.emb.shape[1]}-d, "
          f"{len(real)} real outfits/epoch (+ as many fakes), device {device}", flush=True)

    history, best, best_state, bad = [], -1.0, None, 0
    for epoch in range(1, cfg["max_epochs"] + 1):
        model.train()
        t0, losses, wrong, n = time.time(), 0.0, 0, 0
        single = rng.random(len(real)) < cfg.get("single_swap_frac", 0.5)
        data = real + [fakes.swap(o, rng, s) for o, s in zip(real, single)]
        labels = np.r_[np.ones(len(real)), np.zeros(len(real))].astype(np.float32)
        order = rng.permutation(len(data))
        for s in range(0, len(order), cfg["batch_size"]):
            b = order[s:s + cfg["batch_size"]]
            logits = model(*pack([data[i] for i in b], bank, device))
            y = torch.from_numpy(labels[b]).to(device)
            loss = bce(logits, y)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses += loss.item() * len(b)
            wrong += int(((logits > 0).float() != y).sum())
            n += len(b)
        val = evaluate_transformer(model, bank, valid, device)
        h = {"epoch": epoch, "train_loss": losses / n, "active_frac": wrong / n,
             "val_auc": val["compat_auc"], "val_fitb": val["fitb_acc"], "seconds": round(time.time() - t0, 1)}
        history.append(h)
        improved = h["val_auc"] > best
        if improved:
            best, bad = h["val_auc"], 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        print(f"epoch {epoch:3d}  loss {h['train_loss']:.4f}  train err {h['active_frac']:.3f}  "
              f"val AUC {h['val_auc']:.4f}  FITB {100 * h['val_fitb']:.1f}%  {h['seconds']}s"
              f"{'  *' if improved else ''}", flush=True)
        if bad >= cfg["patience"]:
            print(f"early stop: no val AUC improvement for {cfg['patience']} epochs", flush=True)
            break

    finish_history(history, out, cfg["name"], mid_title="training error (real vs fake)")
    model.load_state_dict(best_state)
    torch.save({"state_dict": best_state, "config": cfg, "type_names": bank.type_names}, out / "model.pt")
    best_epoch = max(history, key=lambda h: h["val_auc"])["epoch"]
    import json
    results = {}
    for split in (valid, test):
        m = {"name": cfg["name"], "split": split.name, **evaluate_transformer(model, bank, split, device),
             "best_epoch": best_epoch, "notes": cfg.get("notes", "")}
        (out / f"metrics_{split.name}.json").write_text(json.dumps(m, indent=2))
        results[split.name] = m
        print(f"{cfg['name']:<28} {split.name:<5}  AUC {m['compat_auc']:.3f}  FITB {100 * m['fitb_acc']:.1f}%",
              flush=True)
    return results
