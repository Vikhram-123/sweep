"""Train/test leakage audit for the disjoint split, and re-evaluation on the clean test subset.

    python -m compat.leakage                 # threshold 0.98
    python -m compat.leakage --threshold 0.95

A test item is "leaked" if its id is also in train, or its FashionCLIP embedding has cosine >
threshold with some train item (near-duplicate image). A test pair is "memorised" if both items
are leaked and their train matches were worn together in one train outfit -- the leak that could
actually hand a compatibility model the answer. Writes results/leakage.md (+ a contact sheet of
near-duplicate pairs to sanity-check the threshold).
"""
from __future__ import annotations

import argparse
import io
import itertools
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np

from .data import DATA_DIR, load_split
from .embed_items import load_embeddings

OUT = Path("results")


def nearest_train(test_ids, test_emb, train_emb, chunk: int = 1024):
    """For each test item: (max cosine to any train item, index of that train item)."""
    best = np.empty(len(test_ids), np.float32)
    arg = np.empty(len(test_ids), np.int64)
    for s in range(0, len(test_ids), chunk):
        sim = test_emb[s:s + chunk] @ train_emb.T
        arg[s:s + chunk] = sim.argmax(1)
        best[s:s + chunk] = sim[np.arange(len(sim)), arg[s:s + chunk]]
    return best, arg


def contact_sheet(pairs: list[tuple[str, str, float]], path: Path) -> None:
    import pyarrow.parquet as pq
    from PIL import Image, ImageDraw

    want = {i for a, b, _ in pairs for i in (a, b)}
    imgs = {}
    for split in ("train", "test"):
        t = pq.read_table(DATA_DIR / "data" / "disjoint" / f"{split}.parquet", filters=[("item_id", "in", list(want))])
        for iid, im in zip(t.column("item_id").to_pylist(), t.column("image").to_pylist()):
            imgs.setdefault(iid, Image.open(io.BytesIO(im["bytes"])).convert("RGB").resize((120, 120)))
    sheet = Image.new("RGB", (260 * 4, 150 * ((len(pairs) + 3) // 4)), "white")
    draw = ImageDraw.Draw(sheet)
    for k, (a, b, c) in enumerate(pairs):
        x, y = (k % 4) * 260, (k // 4) * 150
        sheet.paste(imgs[a], (x + 5, y + 5))
        sheet.paste(imgs[b], (x + 130, y + 5))
        draw.text((x + 5, y + 130), f"test vs train  cos {c:.3f}", fill="black")
    sheet.save(path)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.98)
    ap.add_argument("--runs", nargs="*", default=["model_a", "model_a_seed1", "model_a_seed2",
                                                  "model_b", "model_b_seed1", "model_b_seed2"])
    args = ap.parse_args(argv)
    thr = args.threshold

    train, test = load_split("train"), load_split("test")
    emb = load_embeddings(["train", "test"])
    train_ids = sorted(train.item_ids)
    test_ids = sorted(test.item_ids)
    E_train = np.stack([emb[i] for i in train_ids])
    E_test = np.stack([emb[i] for i in test_ids])

    # --- item level -------------------------------------------------------------------------
    same_id = set(test_ids) & set(train_ids)
    best, arg = nearest_train(test_ids, E_test, E_train)
    nn_train = {t: train_ids[a] for t, a in zip(test_ids, arg)}
    cos = dict(zip(test_ids, best))
    near = {t for t in test_ids if cos[t] > thr}
    leaked = same_id | near
    hist = {f"> {c}": int((best > c).sum()) for c in (0.90, 0.95, 0.98, 0.99, 0.995)}

    # --- pair level: was the test pair worn together in train? ------------------------------
    train_outfits_of = defaultdict(set)
    for oid, items in train.outfits.items():
        for i in items:
            train_outfits_of[i].add(oid)

    def match(t):  # train item standing in for test item t
        return t if t in same_id else nn_train[t]

    def memorised(a, b):
        return a in leaked and b in leaked and bool(train_outfits_of[match(a)] & train_outfits_of[match(b)])

    def outfit_flags(items):
        return any(i in leaked for i in items), any(memorised(a, b) for a, b in itertools.combinations(items, 2))

    comp = {1: [0, 0, 0], 0: [0, 0, 0]}  # label -> [n, with leaked item, with memorised pair]
    for label, items in test.compat:
        has_item, has_pair = outfit_flags(items)
        comp[label][0] += 1
        comp[label][1] += has_item
        comp[label][2] += has_pair
    fitb_ctx = sum(any(i in leaked for i in q.context) for q in test.fitb)
    fitb_ans = sum(q.candidates[q.answer] in leaked for q in test.fitb)
    fitb_any = sum(any(i in leaked for i in q.context + q.candidates) for q in test.fitb)
    fitb_pair = sum(any(memorised(c, q.candidates[q.answer]) for c in q.context) for q in test.fitb)

    # --- clean subset re-evaluation ---------------------------------------------------------
    clean = replace(test,
                    compat=[(l, it) for l, it in test.compat if not any(i in leaked for i in it)],
                    fitb=[q for q in test.fitb if not any(i in leaked for i in q.context + q.candidates)])

    from .baselines import cosine_scorer
    from .evaluate import compat_auc, fitb_accuracy
    from .analyze import load_run
    from .evaluate import RUNS_DIR
    from .train import Bank, evaluate_model

    rows = [("clip_cosine", compat_auc(test, cosine_scorer(emb)), fitb_accuracy(test, cosine_scorer(emb)),
             compat_auc(clean, cosine_scorer(emb)), fitb_accuracy(clean, cosine_scorer(emb)))]
    bank = Bank.build()
    for run in args.runs:
        if not (RUNS_DIR / run / "model.pt").exists():
            continue
        m = load_run(run, len(bank.type_names))
        full, cl = evaluate_model(m, bank, test, "cpu"), evaluate_model(m, bank, clean, "cpu")
        rows.append((run, full["compat_auc"], full["fitb_acc"], cl["compat_auc"], cl["fitb_acc"]))

    # --- contact sheet: what does each similarity level look like? --------------------------
    rng = np.random.default_rng(0)
    pairs = []
    for lo, hi in [(0.90, 0.95), (0.95, 0.98), (0.98, 0.99), (0.99, 1.01)]:
        cand = [t for t in test_ids if lo < cos[t] <= hi and t not in same_id]
        for t in rng.choice(cand, size=min(4, len(cand)), replace=False):
            pairs.append((t, nn_train[t], float(cos[t])))
    contact_sheet(pairs, OUT / f"leakage_pairs.png")

    pct = lambda a, b: f"{a} / {b} ({100 * a / b:.1f}%)"
    L = [f"# Train/test leakage audit — disjoint split (near-duplicate threshold: cosine > {thr})", "",
         "## Items", "",
         f"- Test items: {len(test_ids)}; train items: {len(train_ids)}",
         f"- Identical item ids in train and test: **{len(same_id)}**",
         f"- Test items whose nearest train item (FashionCLIP cosine) exceeds: "
         + ", ".join(f"{k}: {v}" for k, v in hist.items()),
         f"- Leaked test items (same id or cosine > {thr}): **{pct(len(leaked), len(test_ids))}**",
         "- `leakage_pairs.png`: random test/train pairs per similarity band (0.90–0.95, 0.95–0.98, "
         "0.98–0.99, > 0.99), to check what the threshold means visually.", "",
         "## Test questions touched", "",
         "| task | questions | contain a leaked item | contain a memorised pair* |", "|---|---|---|---|",
         f"| compat, real outfits | {comp[1][0]} | {pct(comp[1][1], comp[1][0])} | {pct(comp[1][2], comp[1][0])} |",
         f"| compat, random outfits | {comp[0][0]} | {pct(comp[0][1], comp[0][0])} | {pct(comp[0][2], comp[0][0])} |",
         f"| FITB (context, any) | {len(test.fitb)} | {pct(fitb_ctx, len(test.fitb))} | — |",
         f"| FITB (correct answer) | {len(test.fitb)} | {pct(fitb_ans, len(test.fitb))} | "
         f"{pct(fitb_pair, len(test.fitb))} |",
         f"| FITB (any item) | {len(test.fitb)} | {pct(fitb_any, len(test.fitb))} | — |", "",
         "\\* both items leaked and their train matches appear together in one train outfit.", "",
         "## Re-evaluation with every question that touches a leaked item removed", "",
         f"Clean subset: {len(clean.compat)} compat outfits "
         f"({sum(l for l, _ in clean.compat)} real), {len(clean.fitb)} FITB questions.", "",
         "| run | AUC (full test) | AUC (clean) | Δ | FITB (full) | FITB (clean) | Δ |", "|---|---|---|---|---|---|---|"]
    for name, fa, ff, ca, cf in rows:
        L.append(f"| {name} | {fa:.3f} | {ca:.3f} | {ca - fa:+.3f} | {100 * ff:.1f}% | {100 * cf:.1f}% | "
                 f"{100 * (cf - ff):+.1f} |")
    suffix = "" if thr == 0.98 else f"_{thr}"
    (OUT / f"leakage{suffix}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
