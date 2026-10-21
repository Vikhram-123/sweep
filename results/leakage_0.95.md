# Train/test leakage audit — disjoint split (near-duplicate threshold: cosine > 0.95)

## Items

- Test items: 70035; train items: 71967
- Identical item ids in train and test: **84**
- Test items whose nearest train item (FashionCLIP cosine) exceeds: > 0.9: 10976, > 0.95: 1841, > 0.98: 728, > 0.99: 495, > 0.995: 297
- Leaked test items (same id or cosine > 0.95): **1841 / 70035 (2.6%)**
- `leakage_pairs.png`: random test/train pairs per similarity band (0.90–0.95, 0.95–0.98, 0.98–0.99, > 0.99), to check what the threshold means visually.

## Test questions touched

| task | questions | contain a leaked item | contain a memorised pair* |
|---|---|---|---|
| compat, real outfits | 15145 | 1927 / 15145 (12.7%) | 2 / 15145 (0.0%) |
| compat, random outfits | 15145 | 2007 / 15145 (13.3%) | 0 / 15145 (0.0%) |
| FITB (context, any) | 15145 | 1548 / 15145 (10.2%) | — |
| FITB (correct answer) | 15145 | 442 / 15145 (2.9%) | 1 / 15145 (0.0%) |
| FITB (any item) | 15145 | 2969 / 15145 (19.6%) | — |

\* both items leaked and their train matches appear together in one train outfit.

## Re-evaluation with every question that touches a leaked item removed

Clean subset: 26356 compat outfits (13218 real), 12176 FITB questions.

| run | AUC (full test) | AUC (clean) | Δ | FITB (full) | FITB (clean) | Δ |
|---|---|---|---|---|---|---|
| clip_cosine | 0.753 | 0.755 | +0.002 | 52.4% | 52.8% | +0.4 |
| model_a | 0.916 | 0.914 | -0.002 | 67.3% | 67.3% | +0.1 |
| model_a_seed1 | 0.915 | 0.913 | -0.001 | 67.2% | 67.4% | +0.1 |
| model_a_seed2 | 0.914 | 0.912 | -0.002 | 66.6% | 66.9% | +0.2 |
| model_b | 0.914 | 0.913 | -0.002 | 66.5% | 66.9% | +0.4 |
| model_b_seed1 | 0.911 | 0.909 | -0.002 | 66.1% | 66.5% | +0.4 |
| model_b_seed2 | 0.909 | 0.906 | -0.002 | 66.7% | 66.9% | +0.2 |
