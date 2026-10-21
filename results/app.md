# App track — best image-only model for Sweep, any encoder

Same protocol: official disjoint split and FITB files, selection on validation, 3 seeds, test reported only.

## Starting point

| model | encoder | val AUC | val FITB | test AUC | test FITB |
|---|---|---|---|---|---|
| Model A (triplet, 128-d) | FashionCLIP (frozen) | 0.936 | 68.7% | 0.915 ± 0.001 | 67.0% ± 0.4 |


## Step 1 — InfoNCE vs triplet (FashionCLIP, 128-d)

| model | seeds | val AUC | val FITB | **test AUC** | **test FITB** |
|---|---|---|---|---|---|
| triplet (Model A) | 3 | 0.936 | 68.7% | 0.915 ± 0.001 | 67.0% ± 0.4 |
| **InfoNCE**, in-batch same-category negatives, learned τ | 3 | 0.943 | 70.1% | 0.921 ± 0.001 | 68.2% ± 0.3 |

InfoNCE helps: +0.007 validation AUC, test 0.921 / 68.2% (vs 0.915 / 67.0%). Learned τ ≈ 0.09 (close
to CLIP's 0.07). It peaks after 1–2 epochs, so it learns fast and then slowly overfits.
