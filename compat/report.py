"""Collect results/runs/*/metrics_*.json into results/results.md and results/results.csv.

    python -m compat.report

Runs named <model>_seed<N> are pooled with <model> as mean ± std over seeds. Validation columns are
what every choice was made on; test columns are reported, never used for selection.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np
import yaml

from .evaluate import RUNS_DIR

# Published Polyvore Outfits-D (disjoint) results. All fine-tune their ResNet-18 end to end.
PUBLISHED = [
    ("Bi-LSTM + VSE (Han et al. 2017)", "ResNet-18 (fine-tuned) + text", 0.62, 0.394, "Vasileva et al. 2018, Table 5"),
    ("SiameseNet", "ResNet-18 (fine-tuned)", 0.81, 0.518, "Vasileva et al. 2018, Table 5"),
    ("Type-aware CSN T1:1", "ResNet-18 (fine-tuned)", 0.82, 0.525, "Vasileva et al. 2018, Table 5"),
    ("Type-aware CSN + VSE + Sim + Metric (512-D)", "ResNet-18 (fine-tuned) + text", 0.84, 0.552,
     "Vasileva et al. 2018, Table 5"),
    ("CSA-Net (Lin et al. 2020)", "ResNet-18 (fine-tuned)", 0.87, 0.5926, "CVPR 2020, Table 1"),
    ("OutfitTransformer (Sarkar et al. 2023)", "ResNet-18 (fine-tuned) + text", 0.88, 0.5948, "WACV 2023, Tables 1–2"),
]
ENCODER_LABEL = {"fashionclip": "FashionCLIP (frozen)", "clip": "CLIP ViT-B/32 (frozen)",
                 "resnet18": "ResNet-18 ImageNet (frozen)", "text": "FashionCLIP text (frozen)"}

# (section title, regex on the pooled run name), in display order; first match wins.
SECTIONS = [
    ("Baselines", r"^(random|rules|rules_clothing|\w+_cosine)$"),
    ("Model A / Model B (FashionCLIP)", r"^model_[ab](_masklr10|_full)?$"),
    ("Encoder ablation (Model A, identical config)", r"^model_a_(resnet18|clip)$"),
    ("Hard negatives (Model A, FashionCLIP)", r"^model_a_neg_"),
    ("Squared vs plain distance (Model A, FashionCLIP)", r"^model_a_sq"),
    ("Image + title text (Model A, FashionCLIP)", r"^model_a_text_"),
    ("Hyperparameter sweep winner", r"^sweep_best"),
    ("Combined model", r"^combined"),
    ("Outfit transformer", r"^transformer"),
]


def _encoder(run_dir: Path, name: str) -> str:
    cfg_path = run_dir / "config.yaml"
    if cfg_path.exists():
        cfg = yaml.safe_load(cfg_path.read_text())
        enc = ENCODER_LABEL[cfg.get("encoder", "fashionclip")]
        feats = cfg.get("features", "image")
        return enc + ("" if feats == "image" else f" + title text ({feats})")
    enc = {"openai_clip_cosine": "clip"}.get(name, name.removesuffix("_cosine"))
    if name.endswith("_cosine") and enc in ENCODER_LABEL:
        return ENCODER_LABEL[enc]
    return {"rules": "color tags + FashionCLIP zero-shot",
            "rules_clothing": "color tags + FashionCLIP zero-shot"}.get(name, "—")


def collect() -> list[dict]:
    groups: dict[str, list[tuple[dict, dict | None, Path]]] = {}
    for p in sorted(RUNS_DIR.glob("*/metrics_test.json")):
        if p.parent.name.startswith(("_", "sweep_trial")):
            continue  # scratch runs and sweep trials (reported separately)
        m = json.loads(p.read_text())
        v = p.parent / "metrics_valid.json"
        groups.setdefault(re.sub(r"_seed\d+$", "", m["name"]), []).append(
            (m, json.loads(v.read_text()) if v.exists() else None, p.parent))
    rows = []
    for name, runs in groups.items():
        t_auc = np.array([m["compat_auc"] for m, _, _ in runs])
        t_fitb = np.array([m["fitb_acc"] for m, _, _ in runs])
        vals = [v for _, v, _ in runs if v]
        sd = lambda x: float(x.std(ddof=1)) if len(x) > 1 else None
        section = next((i for i, (_, rx) in enumerate(SECTIONS) if re.search(rx, name)), len(SECTIONS))
        rows.append({"section": section, "model": name, "source": "ours", "encoder": _encoder(runs[0][2], name),
                     "seeds": len(runs),
                     "val_auc": float(np.mean([v["compat_auc"] for v in vals])) if vals else None,
                     "val_fitb": float(np.mean([v["fitb_acc"] for v in vals])) if vals else None,
                     "compat_auc": float(t_auc.mean()), "compat_auc_std": sd(t_auc),
                     "fitb_acc": float(t_fitb.mean()), "fitb_acc_std": sd(t_fitb),
                     "notes": runs[0][0].get("notes", "")})
    rows.sort(key=lambda r: (r["section"], r["model"]))
    rows += [{"section": len(SECTIONS) + 1, "model": n, "source": "published", "encoder": e, "seeds": None,
              "val_auc": None, "val_fitb": None, "compat_auc": a, "compat_auc_std": None, "fitb_acc": f,
              "fitb_acc_std": None, "notes": note} for n, e, a, f, note in PUBLISHED]
    return rows


def _fmt(mean, std=None, pct=False) -> str:
    if mean is None:
        return ""
    k = 100 if pct else 1
    body = f"{k * mean:.1f}%" if pct else f"{mean:.3f}"
    if std is None:
        return body
    return body + (f" ± {k * std:.1f}" if pct else f" ± {std:.3f}")


def table(rows: list[dict]) -> list[str]:
    head = ["| model | encoder | seeds | val AUC | val FITB | **test AUC** | **test FITB** | notes |",
            "|---|---|---|---|---|---|---|---|"]
    out, current = [], None
    for r in rows:
        title = SECTIONS[r["section"]][0] if r["section"] < len(SECTIONS) else \
            ("Other" if r["section"] == len(SECTIONS) else "Published (Polyvore Outfits disjoint)")
        if title != current:
            out += ["", f"**{title}**", ""] + head
            current = title
        out.append(f"| {r['model']} | {r['encoder']} | {r['seeds'] or ''} | {_fmt(r['val_auc'])} | "
                   f"{_fmt(r['val_fitb'], pct=True)} | {_fmt(r['compat_auc'], r['compat_auc_std'])} | "
                   f"{_fmt(r['fitb_acc'], r['fitb_acc_std'], pct=True)} | {r['notes']} |")
    return out


def main(out_dir: Path = Path("results")) -> None:
    rows = collect()
    with open(out_dir / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[k for k in rows[0] if k != "section"], extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    lines = ["# Outfit compatibility — Polyvore Outfits, disjoint split", "",
             "Trained models: mean ± std over seeds. Every choice was made on **validation**; test is "
             "reported only. Validation shares 26% of its items with train, so it runs higher than test."]
    lines += table(rows)
    (out_dir / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
