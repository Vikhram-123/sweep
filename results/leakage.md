# Train/test leakage audit — disjoint split (near-duplicate threshold: cosine > 0.98)

## Items

- Test items: 70035; train items: 71967
- Identical item ids in train and test: **84**
- Test items whose nearest train item (FashionCLIP cosine) exceeds: > 0.9: 10976, > 0.95: 1841, > 0.98: 728, > 0.99: 495, > 0.995: 297
- Leaked test items (same id or cosine > 0.98): **728 / 70035 (1.0%)**
- `leakage_pairs.png`: random test/train pairs per similarity band (0.90–0.95, 0.95–0.98, 0.98–0.99, > 0.99), to check what the threshold means visually.

## Test questions touched

| task | questions | contain a leaked item | contain a memorised pair* |
|---|---|---|---|
| compat, real outfits | 15145 | 737 / 15145 (4.9%) | 2 / 15145 (0.0%) |
| compat, random outfits | 15145 | 803 / 15145 (5.3%) | 0 / 15145 (0.0%) |
| FITB (context, any) | 15145 | 582 / 15145 (3.8%) | — |
| FITB (correct answer) | 15145 | 178 / 15145 (1.2%) | 1 / 15145 (0.0%) |
| FITB (any item) | 15145 | 1189 / 15145 (7.9%) | — |

\* both items leaked and their train matches appear together in one train outfit.

## Re-evaluation with every question that touches a leaked item removed

Clean subset: 28750 compat outfits (14408 real), 13956 FITB questions.

| run | AUC (full test) | AUC (clean) | Δ | FITB (full) | FITB (clean) | Δ |
|---|---|---|---|---|---|---|
| clip_cosine | 0.753 | 0.754 | +0.001 | 52.4% | 52.3% | -0.1 |
| model_a | 0.916 | 0.915 | -0.001 | 67.3% | 67.1% | -0.2 |
| model_a_seed1 | 0.915 | 0.914 | -0.000 | 67.2% | 67.0% | -0.2 |
| model_a_seed2 | 0.914 | 0.914 | -0.001 | 66.6% | 66.5% | -0.1 |
| model_b | 0.914 | 0.914 | -0.001 | 66.5% | 66.4% | -0.1 |
| model_b_seed1 | 0.911 | 0.910 | -0.001 | 66.1% | 66.0% | -0.1 |
| model_b_seed2 | 0.909 | 0.908 | -0.001 | 66.7% | 66.5% | -0.2 |
