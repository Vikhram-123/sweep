# Squared vs plain Euclidean distance (step 4) — Model A, FashionCLIP

Squared distance is used in **both** the triplet loss and the evaluation score (mean pairwise −d²),
so training and testing measure the same quantity. Each distance got its own margin, tuned on
validation (seed 0), then the winner ran 3 seeds.

**Margin search (seed 0, validation AUC):**

| margin | 0.05 | 0.1 | 0.2 | 0.3 | 0.4 | 0.8 |
|---|---|---|---|---|---|---|
| squared distance | 0.9251 | 0.9299 | 0.9328 | — | **0.9368** | 0.9275 |
| plain distance | — | 0.9303 | **0.9367** | 0.9298 | 0.9213 | — |

**3 seeds at each distance's best margin:**

| distance | margin | val AUC | val FITB | test AUC | test FITB |
|---|---|---|---|---|---|
| **plain** ‖a − b‖ | 0.2 | **0.936** | **68.7%** | **0.915 ± 0.001** | **67.0% ± 0.4** |
| squared ‖a − b‖² | 0.4 | 0.935 | 68.1% | 0.914 ± 0.002 | 66.4% ± 0.2 |

**Result: no difference that matters; plain distance kept** (marginally better on validation, and
simpler).

## Why they end up the same

- On L2-normalised embeddings, ‖a − b‖² = 2 − 2·cos(a, b), and plain distance is its square root.
  Both are monotone in cosine, so they rank any *single* pair identically. They can only differ in
  (1) how the margin's effective size varies across the space and (2) how the outfit average weights
  close vs. far pairs.
- **The tuned margins line up.** Near d ≈ 1 (typical pairs), a plain-distance gap of 0.2
  corresponds to a squared-distance gap of about 2 · 1 · 0.2 = 0.4. That's exactly the margin each
  search picked. Once each distance gets its best margin, they demand the same separation where
  most triplets live.
- **The remaining difference is the outfit average.** Averaging d² gives more weight to the most
  distant (worst-matching) pair. That could in principle help catch "the one item that doesn't
  belong", but with random-swap negatives every item is wrong, so there's nothing for it to pick out.
