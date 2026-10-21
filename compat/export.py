"""Export a trained pairwise model to numpy for the app (sweep/compat.py runs it without torch).

    python -m compat.export model_a --out sweep/compat_model.npz

Besides the weights, stores a calibration computed on *clothing-only* validation outfits (what
the app scores): mean/std of the model's outfit score, and the std of the old rule-based visual
terms (color + pattern + contrast), so sweep.outfits can put the model on the rules' scale.
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import torch

from sweep.outfits import score_outfit

from .analyze import load_run
from .baselines import TO_SWEEP, build_rule_items
from .data import item_category, load_split
from .embed_items import load_colors, load_embeddings
from .evaluate import RUNS_DIR


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--out", type=Path, default=Path("sweep/compat_model.npz"))
    args = ap.parse_args(argv)

    ck = torch.load(RUNS_DIR / args.run / "model.pt", weights_only=False)
    cfg = ck["config"]
    if cfg["model"] != "pairwise":
        raise SystemExit("only the pairwise model (Model A) is exported; the app has no Polyvore types")
    sd = ck["state_dict"]
    weights = {"w1": sd["net.0.weight"].numpy(), "b1": sd["net.0.bias"].numpy(),
               "w2": sd["net.3.weight"].numpy(), "b2": sd["net.3.bias"].numpy()}

    # Calibration on clothing-only validation outfits.
    from sweep.compat import CompatModel
    model = CompatModel({**weights, "normalize": np.array(cfg["normalize"])})
    valid = load_split("valid")
    embs, colors = load_embeddings(["valid"]), load_colors(["valid"])
    rule_items = build_rule_items(valid.item_ids, embs, colors)
    m_scores, r_scores = [], []
    for _, items in valid.compat:
        clothes = [i for i in items if item_category(i) in TO_SWEEP]
        if len(clothes) < 2:
            continue
        m_scores.append(model.outfit_score(np.stack([embs[i] for i in clothes])))
        _, _, br = score_outfit([rule_items[i] for i in clothes], occasion=None, weather=None)
        r_scores.append(br["color"] + br["pattern"] + br["contrast"])

    torch_check = load_run(args.run, n_types=11)
    x = torch.from_numpy(np.stack([embs[i] for i in list(valid.item_ids)[:256]]))
    with torch.no_grad():
        ref = torch_check(x).numpy()
    err = float(np.abs(model.embed(x.numpy()) - ref).max())
    assert err < 1e-5, f"numpy forward pass disagrees with torch: {err}"

    metrics = {s: json.loads((RUNS_DIR / args.run / f"metrics_{s}.json").read_text()) for s in ("valid", "test")}
    meta = {"run": args.run, "exported": date.today().isoformat(), "encoder": "patrickjohncyh/fashion-clip",
            "config": cfg, "valid": metrics["valid"], "test": metrics["test"],
            "calibration_outfits": len(m_scores), "max_abs_err_vs_torch": err}
    np.savez(args.out, **weights, normalize=np.array(cfg["normalize"]),
             score_mean=np.array(np.mean(m_scores)), score_std=np.array(np.std(m_scores)),
             rules_visual_std=np.array(np.std(r_scores)), meta=np.array(json.dumps(meta)))
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e3:.0f} kB): {len(m_scores)} calibration outfits, "
          f"model score {np.mean(m_scores):.3f} ± {np.std(m_scores):.3f}, rules visual std {np.std(r_scores):.3f}, "
          f"max |numpy - torch| = {err:.1e}")


if __name__ == "__main__":
    main()
