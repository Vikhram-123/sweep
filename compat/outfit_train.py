"""Outfit-level training: score a candidate against a whole outfit context (FITB-style).

    python -m compat.train compat/configs/listwise.yaml        (model: listwise -> dispatched here)

Each training outfit contributes every item once per epoch as the held-out target. The rest of the
outfit is the context; candidates are the target plus `n_neg` random items of the target's type.

    D(context, c) = pooled pairwise distance between candidate c and the context items
        pool: mean       -- average over context items (CSA-Net eq. 5; equals our usual outfit score)
              attention  -- softmax(q(c) . k(item)) weights, q/k from the item embedding + a type embedding

Losses (cfg loss):
    listwise     softmax cross-entropy over the candidates on -D * scale (learned temperature)
    outfit_rank  CSA-Net's hinge: max(0, D_pos - D_neg + margin), D_neg = closest negative farther
                 than the positive (semi-hard), falling back to the closest negative
Optional: infonce_weight > 0 adds InfoNCE on item pairs from the triplet sampler, each step.
"""
from __future__ import annotations

import time

import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.nn import functional as F

from .data import SPLITS, Split, load_split
from .evaluate import RUNS_DIR
from .losses import info_nce_loss
from .models import build_model
from .train import Bank, TripletSampler, finish_history, seed_everything


class AttentionPool(nn.Module):
    """Weights for context items given a query item: softmax over items of q(query) . k(item) / sqrt(d)."""

    def __init__(self, emb_dim: int, n_types: int, d: int = 32):
        super().__init__()
        self.type_emb = nn.Embedding(n_types, emb_dim)
        nn.init.normal_(self.type_emb.weight, std=0.02)
        self.q = nn.Linear(emb_dim, d)
        self.k = nn.Linear(emb_dim, d)
        self.scale = d ** -0.5

    def forward(self, gq, tq, gk, tk, mask):
        """gq (B,Q,e) tq (B,Q) | gk (B,L,e) tk (B,L) | mask (B,Q,L) True = usable -> weights (B,Q,L)."""
        q = self.q(gq + self.type_emb(tq))
        k = self.k(gk + self.type_emb(tk))
        logits = torch.einsum("bqd,bld->bql", q, k) * self.scale
        return torch.softmax(logits.masked_fill(~mask, float("-inf")), dim=-1)


def general_embedding(model, x):
    """The type-independent embedding used as attention input (the model's MLP output)."""
    return model.net(x)


def pooled_distance(model, pool, bank: Bank, q_rows, c_rows, mask, device):
    """Pooled distance between each query item and its context items.

    q_rows (B,Q), c_rows (B,L) long item rows; mask (B,Q,L) bool -- which context items each query uses.
    Returns (B,Q). Pairwise distances use the model's type-pair spaces: query projected for the
    context item's type, context item projected for the query's type.
    """
    B, Q = q_rows.shape
    L = c_rows.shape[1]
    xq, xc = bank.emb[q_rows].to(device), bank.emb[c_rows].to(device)
    tq, tc = bank.types[q_rows].to(device), bank.types[c_rows].to(device)
    # Expand to every (query, context) pair and project each side into that pair's space.
    xq_e = xq[:, :, None, :].expand(B, Q, L, xq.shape[-1]).reshape(-1, xq.shape[-1])
    xc_e = xc[:, None, :, :].expand(B, Q, L, xc.shape[-1]).reshape(-1, xc.shape[-1])
    tq_e = tq[:, :, None].expand(B, Q, L).reshape(-1)
    tc_e = tc[:, None, :].expand(B, Q, L).reshape(-1)
    zq = model(xq_e, tq_e, tc_e)
    zc = model(xc_e, tc_e, tq_e)
    d = (zq - zc).norm(dim=-1).reshape(B, Q, L)
    mask = mask.to(device)
    if pool is None:
        w = mask.float() / mask.float().sum(-1, keepdim=True).clamp_min(1)
    else:
        w = pool(general_embedding(model, xq), tq, general_embedding(model, xc), tc, mask)
        w = torch.nan_to_num(w)  # a query with no usable context gets all-zero weights
    return (w * d).sum(-1)


# ----------------------------------------------------------------------------- data
class HoldoutSampler:
    """Every item of every training outfit as the held-out target once per epoch."""

    def __init__(self, split: Split, bank: Bank, n_neg: int, max_items: int = 16):
        self.outfits = [np.array([bank.index[i] for i in items])[:max_items] for items in split.outfits.values()]
        self.outfits = [o for o in self.outfits if len(o) >= 2]
        types = bank.types.numpy()
        rows = np.unique(np.concatenate(self.outfits))
        self.by_type = {t: rows[types[rows] == t] for t in np.unique(types[rows])}
        self.types, self.n_neg = types, n_neg
        self.pairs = [(k, j) for k, o in enumerate(self.outfits) for j in range(len(o))]

    def __len__(self):
        return len(self.pairs)

    def batches(self, rng, batch_size: int):
        order = rng.permutation(len(self.pairs))
        for s in range(0, len(order), batch_size):
            sel = [self.pairs[i] for i in order[s:s + batch_size]]
            L = max(len(self.outfits[k]) - 1 for k, _ in sel)
            ctx = np.zeros((len(sel), L), np.int64)
            cmask = np.zeros((len(sel), L), bool)
            cand = np.zeros((len(sel), self.n_neg + 1), np.int64)
            for b, (k, j) in enumerate(sel):
                o = self.outfits[k]
                rest = np.delete(o, j)
                ctx[b, :len(rest)], cmask[b, :len(rest)] = rest, True
                pool = self.by_type[self.types[o[j]]]
                neg = pool[rng.integers(len(pool), size=self.n_neg)]
                while (clash := neg == o[j]).any():  # never use the target itself as a negative
                    neg[clash] = pool[rng.integers(len(pool), size=clash.sum())]
                cand[b] = np.r_[o[j], neg]
            yield torch.from_numpy(ctx), torch.from_numpy(cmask), torch.from_numpy(cand)


# ----------------------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate_outfit_model(model, pool, bank: Bank, split: Split, device: str, batch: int = 512) -> dict:
    model.eval()
    if pool is not None:
        pool.eval()

    def compat_scores(outfits):
        out = []
        for s in range(0, len(outfits), batch):
            chunk = [np.array([bank.index[i] for i in o]) for o in outfits[s:s + batch]]
            L = max(len(o) for o in chunk)
            rows = np.zeros((len(chunk), L), np.int64)
            valid = np.zeros((len(chunk), L), bool)
            for b, o in enumerate(chunk):
                rows[b, :len(o)], valid[b, :len(o)] = o, True
            r, v = torch.from_numpy(rows), torch.from_numpy(valid)
            mask = v[:, None, :] & v[:, :, None] & ~torch.eye(L, dtype=torch.bool)[None]
            d = pooled_distance(model, pool, bank, r, r, mask, device).cpu()  # (B, L): item vs rest
            out.append(-(d * v).sum(1) / v.sum(1))
        return torch.cat(out).numpy()

    def fitb_scores(questions):
        out = []
        for s in range(0, len(questions), batch):
            qs = questions[s:s + batch]
            L = max(len(q.context) for q in qs)
            ctx = np.zeros((len(qs), L), np.int64)
            valid = np.zeros((len(qs), L), bool)
            for b, q in enumerate(qs):
                ctx[b, :len(q.context)] = [bank.index[i] for i in q.context]
                valid[b, :len(q.context)] = True
            cand = torch.tensor([[bank.index[c] for c in q.candidates] for q in qs])
            mask = torch.from_numpy(valid)[:, None, :].expand(len(qs), cand.shape[1], L)
            out.append(-pooled_distance(model, pool, bank, cand, torch.from_numpy(ctx), mask, device).cpu())
        return torch.cat(out).numpy()

    labels = [l for l, _ in split.compat]
    auc = roc_auc_score(labels, compat_scores([items for _, items in split.compat]))
    s = fitb_scores(split.fitb)
    acc = float(np.mean(s.argmax(1) == np.array([q.answer for q in split.fitb])))
    return {"compat_auc": float(auc), "fitb_acc": acc}


# ----------------------------------------------------------------------------- training
def outfit_loss(D: torch.Tensor, kind: str, scale: torch.Tensor, margin: float) -> torch.Tensor:
    """D (B, 1 + n_neg): pooled distances, column 0 = the real target."""
    if kind == "listwise":
        return F.cross_entropy(-D * scale, torch.zeros(len(D), dtype=torch.long, device=D.device))
    if kind == "outfit_rank":
        d_pos, d_neg = D[:, :1], D[:, 1:]
        semi = d_neg.masked_fill(d_neg <= d_pos, float("inf")).min(1).values
        hardest = d_neg.min(1).values
        agg = torch.where(torch.isfinite(semi), semi, hardest)
        return torch.relu(d_pos.squeeze(1) - agg + margin).mean()
    raise ValueError(kind)


def train_outfit_model(cfg: dict) -> dict:
    seed_everything(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    device = cfg.get("device", "cpu")
    out = RUNS_DIR / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    bank = Bank.build(cfg.get("encoder", "fashionclip"), cfg.get("features", "image"))
    train_split, valid, test = (load_split(s) for s in SPLITS)
    sampler = HoldoutSampler(train_split, bank, cfg.get("n_neg", 15))
    model = build_model({**cfg, "model": cfg.get("projection_model", "pairwise"), "in_dim": bank.emb.shape[1]},
                        n_types=len(bank.type_names)).to(device)
    emb_dim = model.net[-1].out_features
    pool = AttentionPool(emb_dim, len(bank.type_names), cfg.get("attn_dim", 32)).to(device) \
        if cfg.get("pool", "mean") == "attention" else None
    log_scale = nn.Parameter(torch.tensor(float(np.log(1.0 / cfg.get("temperature_init", 0.07)))))
    params = list(model.parameters()) + (list(pool.parameters()) if pool is not None else [])
    opt = torch.optim.Adam([{"params": params}, {"params": [log_scale], "weight_decay": 0.0}],
                           lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    lam = cfg.get("infonce_weight", 0.0)
    pairs = TripletSampler(train_split, bank) if lam > 0 else None
    kind = cfg.get("loss", "listwise")
    n_params = sum(p.numel() for p in params)
    print(f"{cfg['name']}: {n_params:,} params, input {bank.emb.shape[1]}-d, {len(sampler):,} held-out targets/epoch, "
          f"{cfg.get('n_neg', 15)} negatives, loss {kind}, pool {cfg.get('pool', 'mean')}, "
          f"projection {cfg.get('projection_model', 'pairwise')}, infonce_weight {lam}, device {device}", flush=True)

    history, best, best_state, bad = [], -1.0, None, 0
    for epoch in range(1, cfg["max_epochs"] + 1):
        model.train()
        if pool is not None:
            pool.train()
        t0, losses, wrong, n = time.time(), 0.0, 0, 0
        pair_iter = None
        if pairs is not None:
            trips = torch.from_numpy(pairs.epoch(rng))
            pair_iter = iter(range(0, len(trips), cfg["pair_batch_size"]))
        for ctx, cmask, cand in sampler.batches(rng, cfg["batch_size"]):
            mask = cmask[:, None, :].expand(len(ctx), cand.shape[1], ctx.shape[1])
            D = pooled_distance(model, pool, bank, cand, ctx, mask, device)
            scale = log_scale.exp().clamp(max=100.0)
            loss = outfit_loss(D, kind, scale, cfg.get("margin", 0.2))
            if pair_iter is not None:
                s = next(pair_iter, None)
                if s is None:
                    pair_iter = iter(range(0, len(trips), cfg["pair_batch_size"]))
                    s = next(pair_iter)
                a, p, _, outfit = trips[s:s + cfg["pair_batch_size"]].T
                ta, tp = bank.types[a].to(device), bank.types[p].to(device)
                za, zp = model(bank.emb[a].to(device), ta, tp), model(bank.emb[p].to(device), tp, ta)
                o, pd = outfit.to(device), p.to(device)
                valid_neg = (tp[None] == tp[:, None]) & (o[None] != o[:, None]) & (pd[None] != pd[:, None])
                loss = loss + lam * info_nce_loss(za, zp, valid_neg, scale)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses += loss.item() * len(ctx)
            wrong += int((D.detach().argmin(1) != 0).sum())
            n += len(ctx)
        val = evaluate_outfit_model(model, pool, bank, valid, device)
        h = {"epoch": epoch, "train_loss": losses / n, "active_frac": wrong / n, "val_auc": val["compat_auc"],
             "val_fitb": val["fitb_acc"], "seconds": round(time.time() - t0, 1),
             "temperature": float(1.0 / log_scale.exp().clamp(max=100.0))}
        history.append(h)
        improved = h["val_auc"] > best
        if improved:
            best, bad = h["val_auc"], 0
            best_state = {"model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                          "pool": {k: v.detach().cpu().clone() for k, v in pool.state_dict().items()} if pool else None,
                          "log_scale": log_scale.detach().cpu().clone()}
        else:
            bad += 1
        print(f"epoch {epoch:3d}  loss {h['train_loss']:.4f}  train FITB-{cfg.get('n_neg', 15) + 1} err {h['active_frac']:.3f}  "
              f"T {h['temperature']:.3f}  val AUC {h['val_auc']:.4f}  FITB {100 * h['val_fitb']:.1f}%  {h['seconds']}s"
              f"{'  *' if improved else ''}", flush=True)
        if bad >= cfg["patience"]:
            print(f"early stop: no val AUC improvement for {cfg['patience']} epochs", flush=True)
            break

    finish_history(history, out, cfg["name"], mid_title=f"train error picking the real item (1 of {cfg.get('n_neg', 15) + 1})")
    model.load_state_dict(best_state["model"])
    if pool is not None:
        pool.load_state_dict(best_state["pool"])
    torch.save({**best_state, "config": cfg, "type_names": bank.type_names}, out / "model.pt")
    best_epoch = max(history, key=lambda h: h["val_auc"])["epoch"]
    import json
    results = {}
    for split in (valid, test):
        m = {"name": cfg["name"], "split": split.name, **evaluate_outfit_model(model, pool, bank, split, device),
             "best_epoch": best_epoch, "notes": cfg.get("notes", "")}
        (out / f"metrics_{split.name}.json").write_text(json.dumps(m, indent=2))
        results[split.name] = m
        print(f"{cfg['name']:<28} {split.name:<5}  AUC {m['compat_auc']:.3f}  FITB {100 * m['fitb_acc']:.1f}%", flush=True)
    return results
