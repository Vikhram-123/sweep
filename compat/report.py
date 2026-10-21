"""Collect results/runs/*/metrics_test.json into results/results.md and results/results.csv.

    python -m compat.report
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np

from .evaluate import RUNS_DIR

# Published Polyvore Outfits-D (disjoint) numbers: Vasileva et al., ECCV 2018, Table 5.
PUBLISHED = [
    ("paper: Bi-LSTM + VSE (Han et al.)", "published", 0.62, 0.394),
    ("paper: SiameseNet", "published", 0.81, 0.518),
    ("paper: type-aware CSN T1:1", "published", 0.82, 0.525),
    ("paper: type-aware CSN + VSE + Sim + Metric (512-D)", "published", 0.84, 0.552),
]
ORDER = ["random", "rules", "rules_clothing", "clip_cosine", "model_a", "model_b"]


def collect() -> list[dict]:
    """One row per model; runs named <model>_seed<N> are pooled as mean +- std over seeds."""
    groups: dict[str, list[dict]] = {}
    for p in sorted(RUNS_DIR.glob("*/metrics_test.json")):
        m = json.loads(p.read_text())
        groups.setdefault(re.sub(r"_seed\d+$", "", m["name"]), []).append(m)
    rows = []
    for name, ms in groups.items():
        auc, fitb = np.array([m["compat_auc"] for m in ms]), np.array([m["fitb_acc"] for m in ms])
        rows.append({"model": name, "source": "ours", "seeds": len(ms),
                     "compat_auc": auc.mean(), "compat_auc_std": auc.std(ddof=1) if len(ms) > 1 else None,
                     "fitb_acc": fitb.mean(), "fitb_acc_std": fitb.std(ddof=1) if len(ms) > 1 else None,
                     "notes": ms[0].get("notes", "")})
    rank = {n: i for i, n in enumerate(ORDER)}
    rows.sort(key=lambda r: (rank.get(r["model"], len(ORDER)), r["model"]))
    rows += [{"model": n, "source": s, "seeds": None, "compat_auc": a, "compat_auc_std": None,
              "fitb_acc": f, "fitb_acc_std": None, "notes": "Vasileva et al. 2018, Table 5 (Polyvore Outfits-D)"}
             for n, s, a, f in PUBLISHED]
    return rows


def _fmt(mean: float, std: float | None, pct: bool = False) -> str:
    k = 100 if pct else 1
    body = f"{k * mean:.1f}%" if pct else f"{mean:.3f}"
    if std is None:
        return body
    return body + (f" ± {k * std:.1f}" if pct else f" ± {std:.3f}")


def main(out_dir: Path = Path("results")) -> None:
    rows = collect()
    with open(out_dir / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    lines = ["# Outfit compatibility — Polyvore Outfits, disjoint test set", "",
             "Trained models: mean ± std over seeds (column *seeds*).", "",
             "| model | source | seeds | compat AUC | FITB acc | notes |", "|---|---|---|---|---|---|"]
    lines += [f"| {r['model']} | {r['source']} | {r['seeds'] or ''} | {_fmt(r['compat_auc'], r['compat_auc_std'])} "
              f"| {_fmt(r['fitb_acc'], r['fitb_acc_std'], pct=True)} | {r['notes']} |" for r in rows]
    (out_dir / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
