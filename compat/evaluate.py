"""Evaluation harness: compatibility AUC and fill-in-the-blank accuracy for any outfit scorer.

A scorer is any callable  score(item_ids: list[str]) -> float  (higher = more compatible).
Results are written to results/runs/<name>/metrics.json; compat.report collects them into a table.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.metrics import roc_auc_score

from .data import Split

Scorer = Callable[[list[str]], float]
RUNS_DIR = Path("results/runs")


def compat_auc(split: Split, score: Scorer) -> float:
    labels = [l for l, _ in split.compat]
    scores = [score(items) for _, items in split.compat]
    return float(roc_auc_score(labels, scores))


def fitb_accuracy(split: Split, score: Scorer, rng: np.random.Generator | None = None) -> float:
    """Ties are broken at random, so a constant scorer gets chance (25%), not 'always pick slot 0'."""
    rng = rng or np.random.default_rng(0)
    correct = 0
    for q in split.fitb:
        s = np.array([score(q.context + [c]) for c in q.candidates])
        best = np.flatnonzero(s == s.max())
        correct += int(rng.choice(best) == q.answer)
    return correct / len(split.fitb)


def evaluate(name: str, score: Scorer, split: Split, extra: dict | None = None, save: bool = True) -> dict:
    t0 = time.time()
    m = {"name": name, "split": split.name, "compat_auc": compat_auc(split, score),
         "fitb_acc": fitb_accuracy(split, score), "eval_seconds": None, **(extra or {})}
    m["eval_seconds"] = round(time.time() - t0, 1)
    print(f"{name:<28} {split.name:<5}  AUC {m['compat_auc']:.3f}  FITB {100 * m['fitb_acc']:.1f}%  "
          f"({m['eval_seconds']}s)", flush=True)
    if save:
        out = RUNS_DIR / name
        out.mkdir(parents=True, exist_ok=True)
        (out / f"metrics_{split.name}.json").write_text(json.dumps(m, indent=2))
    return m
