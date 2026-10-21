"""Write results/data_summary.md: sizes of the disjoint split and its test tasks.

    python -m compat.summary
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from .data import DATA_DIR, SPLITS, item_category, item_metadata, load_split, parquet_ids


def main(out: Path = Path("results/data_summary.md")) -> None:
    splits = {s: load_split(s) for s in SPLITS}
    meta = item_metadata()
    rows, cats_all = [], Counter()
    for s, sp in splits.items():
        items = sp.item_ids
        img_ids = set(parquet_ids(s))
        sizes = [len(v) for v in sp.outfits.values()]
        pos = sum(l for l, _ in sp.compat)
        rows.append((s, len(sp.outfits), len(items), len(items - img_ids), sum(sizes) / len(sizes),
                     min(sizes), max(sizes), len(sp.compat), pos, len(sp.fitb)))
        if s == "train":
            cats_all.update(item_category(i) for i in items)

    overlap = {f"{a}/{b}": len(splits[a].item_ids & splits[b].item_ids)
               for a, b in [("train", "valid"), ("train", "test"), ("valid", "test")]}
    fine = {meta[i]["category_id"] for sp in splits.values() for i in sp.item_ids}
    answer_pos = Counter(q.answer for q in splits["test"].fitb)
    test = splits["test"]
    fitb_same = sum(len({item_category(i) for i in q.candidates}) == 1 for q in test.fitb)
    type_sig = lambda items: tuple(sorted(item_category(i) for i in items))
    pos_sigs = Counter(type_sig(items) for l, items in test.compat if l == 1)
    neg_sigs = Counter(type_sig(items) for l, items in test.compat if l == 0)
    pair_types = Counter()
    for items in splits["train"].outfits.values():
        cs = sorted(item_category(i) for i in items)
        pair_types.update((a, b) for k, a in enumerate(cs) for b in cs[k + 1:] if a != b)

    L = ["# Polyvore Outfits — disjoint split", "",
         f"Source: huggingface.co/datasets/mvasil/polyvore-outfits (authors' repo), hashes verified. "
         f"Data lives in `{DATA_DIR}/` (gitignored).", "",
         "| split | outfits | unique items | items w/o image | mean size | min | max | compat outfits | compat positives | FITB questions |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    L += ["| " + " | ".join(f"{v:.2f}" if isinstance(v, float) else str(v) for v in r) + " |" for r in rows]
    L += ["", f"**Item overlap between splits** (should be 0 for the disjoint split): {overlap}", "",
          f"**Categories:** {len(cats_all)} coarse semantic types (used as the paper's \"types\"), "
          f"{len(fine)} fine-grained category ids.", "",
          "| type | train items |", "|---|---|"]
    L += [f"| {c} | {n} |" for c, n in cats_all.most_common()]
    L += ["", f"**Distinct type pairs co-occurring in train outfits:** {len(pair_types)}. Most common: "
          + ", ".join(f"{a}–{b} ({n})" for (a, b), n in pair_types.most_common(8)), "",
          "**Test tasks**", "",
          "- *Compatibility*: each line is a label (1 = real outfit, 0 = items drawn at random from the "
          "split) and the outfit's items. Metric: ROC AUC of outfit scores.",
          "- *Fill-in-the-blank*: a partial outfit plus 4 candidates; exactly one came from the outfit. "
          "Metric: accuracy (chance = 25%). "
          f"Correct-answer position in test: {dict(sorted(answer_pos.items()))}.", "",
          "**What a scorer can and can't exploit**", "",
          f"- FITB candidates share the answer's type in {fitb_same}/{len(test.fitb)} questions, "
          "so category alone can't pick the answer.",
          f"- Negative outfits have the same multiset of types as positives "
          f"({'identical' if pos_sigs == neg_sigs else 'different'} distributions): each item is "
          "replaced by a random item of the same type. Category-pairing rules carry no signal; only "
          "visual compatibility does.",
          f"- Validation shares {overlap['train/valid']} items with train, so validation scores are "
          "optimistic; they are used for early stopping only, never reported as results.", ""]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
