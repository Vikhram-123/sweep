# Train/test leakage audit — disjoint split (near-duplicate threshold: cosine > 0.9)

## Items

- Test items: 70035; train items: 71967
- Identical item ids in train and test: **84**
- Test items whose nearest train item (FashionCLIP cosine) exceeds: > 0.9: 10976, > 0.95: 1841, > 0.98: 728, > 0.99: 495, > 0.995: 297
- Leaked test items (same id or cosine > 0.9): **10976 / 70035 (15.7%)**
- `leakage_pairs.png`: random test/train pairs per similarity band (0.90–0.95, 0.95–0.98, 0.98–0.99, > 0.99), to check what the threshold means visually.

## Test questions touched

| task | questions | contain a leaked item | contain a memorised pair* |
|---|---|---|---|
| compat, real outfits | 15145 | 8177 / 15145 (54.0%) | 6 / 15145 (0.0%) |
| compat, random outfits | 15145 | 8657 / 15145 (57.2%) | 0 / 15145 (0.0%) |
| FITB (context, any) | 15145 | 7093 / 15145 (46.8%) | — |
| FITB (correct answer) | 15145 | 2382 / 15145 (15.7%) | 3 / 15145 (0.0%) |
| FITB (any item) | 15145 | 10538 / 15145 (69.6%) | — |

\* both items leaked and their train matches appear together in one train outfit.

## Re-evaluation with every question that touches a leaked item removed

Clean subset: 13456 compat outfits (6968 real), 4607 FITB questions.

| run | AUC (full test) | AUC (clean) | Δ | FITB (full) | FITB (clean) | Δ |
|---|---|---|---|---|---|---|
| clip_cosine | 0.753 | 0.757 | +0.004 | 52.4% | 53.8% | +1.4 |
| model_a | 0.916 | 0.907 | -0.008 | 67.3% | 68.6% | +1.3 |
| model_a_seed1 | 0.915 | 0.905 | -0.010 | 67.2% | 68.6% | +1.3 |
| model_a_seed2 | 0.914 | 0.908 | -0.007 | 66.6% | 68.5% | +1.9 |
| model_b | 0.914 | 0.906 | -0.009 | 66.5% | 68.6% | +2.0 |
| model_b_seed1 | 0.911 | 0.902 | -0.009 | 66.1% | 67.1% | +1.0 |
| model_b_seed2 | 0.909 | 0.899 | -0.009 | 66.7% | 68.0% | +1.3 |
