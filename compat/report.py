"""Collect results/runs/*/metrics_test.json into results/results.md and results/results.csv.

    python -m compat.report
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from .evaluate import RUNS_DIR

# Published Polyvore Outfits-D (disjoint) numbers: Vasileva et al., ECCV 2018, Table 5.
PUBLISHED = [
    ("paper: Bi-LSTM + VSE (Han et al.)", "published", 0.62, 0.394),
    ("paper: SiameseNet", "published", 0.81, 0.518),
    ("paper: type-aware CSN T1:1", "published", 0.82, 0.525),
    ("paper: type-aware CSN + VSE + Sim + Metric (512-D)", "published", 0.84, 0.552),
]
ORDER = ["random", "rules", "rules_clothing", "clip_cosine"]


def collect() -> list[dict]:
    rows = []
    for p in sorted(RUNS_DIR.glob("*/metrics_test.json")):
        m = json.loads(p.read_text())
        rows.append({"model": m["name"], "source": "ours", "compat_auc": m["compat_auc"],
                     "fitb_acc": m["fitb_acc"], "notes": m.get("notes", "")})
    rank = {n: i for i, n in enumerate(ORDER)}
    rows.sort(key=lambda r: (rank.get(r["model"], len(ORDER)), r["model"]))
    rows += [{"model": n, "source": s, "compat_auc": a, "fitb_acc": f, "notes": "Vasileva et al. 2018, Table 5 (Polyvore Outfits-D)"}
             for n, s, a, f in PUBLISHED]
    return rows


def main(out_dir: Path = Path("results")) -> None:
    rows = collect()
    with open(out_dir / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    lines = ["# Outfit compatibility — Polyvore Outfits, disjoint test set", "",
             "| model | source | compat AUC | FITB acc | notes |", "|---|---|---|---|---|"]
    lines += [f"| {r['model']} | {r['source']} | {r['compat_auc']:.3f} | {100 * r['fitb_acc']:.1f}% | {r['notes']} |"
              for r in rows]
    (out_dir / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
