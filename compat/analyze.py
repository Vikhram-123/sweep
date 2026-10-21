"""Where does type-awareness help? Per-type-pair pair-level AUC for two trained runs.

    python -m compat.analyze model_a model_b

For every item pair inside a test compatibility outfit, the label is the outfit's label (real vs
random swap). Pair AUC per type pair = how well -distance separates real from fake pairs of those
types. Writes results/type_pairs.md and results/type_pairs.png.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from .data import load_split
from .evaluate import RUNS_DIR
from .models import build_model
from .train import Bank


def load_run(name: str, n_types: int):
    ck = torch.load(RUNS_DIR / name / "model.pt", weights_only=False)
    model = build_model(ck["config"], n_types)
    model.load_state_dict(ck["state_dict"])
    return model.eval()


@torch.no_grad()
def pair_table(models: dict, bank: Bank, split) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    a, b, y = [], [], []
    for label, items in split.compat:
        rows = [bank.index[i] for i in items]
        for x in range(len(rows)):
            for z in range(x + 1, len(rows)):
                a.append(rows[x]), b.append(rows[z]), y.append(label)
    a, b, y = torch.tensor(a), torch.tensor(b), np.array(y)
    ta, tb = bank.types[a], bank.types[b]
    dist = {}
    for name, m in models.items():
        dist[name] = (m(bank.emb[a], ta, tb) - m(bank.emb[b], tb, ta)).norm(dim=-1).numpy()
    return ta.numpy(), tb.numpy(), y, dist


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("other")
    ap.add_argument("--min-pairs", type=int, default=500)
    args = ap.parse_args(argv)

    bank = Bank.build()
    models = {n: load_run(n, len(bank.type_names)) for n in (args.base, args.other)}
    ta, tb, y, dist = pair_table(models, bank, load_split("test"))
    lo, hi = np.minimum(ta, tb), np.maximum(ta, tb)

    rows = []
    for u in range(len(bank.type_names)):
        for v in range(u, len(bank.type_names)):
            m = (lo == u) & (hi == v)
            if m.sum() < args.min_pairs or len(set(y[m])) < 2:
                continue
            aucs = {n: roc_auc_score(y[m], -d[m]) for n, d in dist.items()}
            rows.append((bank.type_names[u], bank.type_names[v], int(m.sum()), aucs[args.base], aucs[args.other]))
    rows.sort(key=lambda r: r[4] - r[3], reverse=True)

    other = models[args.other]
    mask_dev = {}
    if hasattr(other, "masks"):
        mk = other.masks.detach()
        mk = mk / mk.norm(dim=-1, keepdim=True)
        ones = torch.ones(mk.shape[1]) / mk.shape[1] ** 0.5
        for (u, v), k in ((uv, other.pair_index[uv].item()) for uv in
                          [(i, j) for i in range(len(bank.type_names)) for j in range(i, len(bank.type_names))]):
            mask_dev[(bank.type_names[u], bank.type_names[v])] = 1 - float(mk[k] @ ones)

    L = [f"# Pair-level AUC by type pair: {args.base} vs {args.other} (test set)", "",
         "Each item pair in a compatibility outfit is labelled real/fake by its outfit; AUC of -distance per "
         f"type pair (pairs with ≥{args.min_pairs} examples). Sorted by gain.", "",
         f"| type pair | pairs | {args.base} | {args.other} | gain |"
         + (" mask moved from 1s (1 − cos) |" if mask_dev else ""),
         "|---|---|---|---|---|" + ("---|" if mask_dev else "")]
    for u, v, n, a, b in rows:
        L.append(f"| {u}–{v} | {n} | {a:.3f} | {b:.3f} | {b - a:+.3f} |"
                 + (f" {mask_dev[(u, v)]:.3f} |" if mask_dev else ""))
    gains = np.array([r[4] - r[3] for r in rows])
    weights = np.array([r[2] for r in rows])
    L += ["", f"Pair-weighted mean gain: {np.average(gains, weights=weights):+.4f}; "
          f"{(gains > 0).sum()}/{len(gains)} type pairs improve."]
    Path("results/type_pairs.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 0.28 * len(rows) + 1.2))
    labels = [f"{u}–{v}" for u, v, *_ in rows][::-1]
    ax.barh(labels, gains[::-1], color=["tab:green" if g > 0 else "tab:red" for g in gains[::-1]])
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel(f"pair AUC gain, {args.other} − {args.base}")
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    fig.savefig("results/type_pairs.png", dpi=120)


if __name__ == "__main__":
    main()
