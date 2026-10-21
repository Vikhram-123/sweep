"""Aggregate runs for the method/app result files: python -m compat.track run_a run_b ..."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

R = Path("results/runs")


def agg(base: str) -> dict | None:
    rs = [d for d in [R / base] + [R / f"{base}_seed{s}" for s in (1, 2)] if (d / "metrics_test.json").exists()]
    if not rs:
        return None
    v = [json.loads((d / "metrics_valid.json").read_text()) for d in rs]
    t = [json.loads((d / "metrics_test.json").read_text()) for d in rs]
    temps = []
    for d in rs:
        h = list(csv.DictReader(open(d / "history.csv")))
        best = max(h, key=lambda r: float(r["val_auc"]))
        if "temperature" in best:
            temps.append(float(best["temperature"]))
    sd = lambda x: float(np.std(x, ddof=1)) if len(x) > 1 else None
    return {"name": base, "seeds": len(rs), "val_auc": np.mean([x["compat_auc"] for x in v]),
            "val_fitb": np.mean([x["fitb_acc"] for x in v]),
            "test_auc": np.mean([x["compat_auc"] for x in t]), "test_auc_sd": sd([x["compat_auc"] for x in t]),
            "test_fitb": np.mean([x["fitb_acc"] for x in t]), "test_fitb_sd": sd([x["fitb_acc"] for x in t]),
            "best_epochs": [x["best_epoch"] for x in t], "temperature": np.mean(temps) if temps else None,
            "notes": t[0].get("notes", "")}


def row(a: dict, label: str | None = None) -> str:
    pm = lambda m, s, pct=False: (f"{100 * m:.1f}%" if pct else f"{m:.3f}") + (
        "" if s is None else (f" ± {100 * s:.1f}" if pct else f" ± {s:.3f}"))
    return (f"| {label or a['name']} | {a['seeds']} | {a['val_auc']:.3f} | {100 * a['val_fitb']:.1f}% | "
            f"{pm(a['test_auc'], a['test_auc_sd'])} | {pm(a['test_fitb'], a['test_fitb_sd'], True)} |")


if __name__ == "__main__":
    for n in sys.argv[1:]:
        a = agg(n)
        print(row(a) if a else f"| {n} | missing |", "| T=%.3f" % a["temperature"] if a and a["temperature"] else "",
              "| best epochs", a["best_epochs"] if a else "")
