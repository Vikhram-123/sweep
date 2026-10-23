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


## Steps 2–4 — outfit-level losses and pooling (screening, seed 0)

| model | val AUC | val FITB | test AUC | test FITB |
|---|---|---|---|---|
| InfoNCE (step-1 winner, seed 0) | 0.943 | 70.3% | 0.921 | 68.1% |
| listwise FITB loss, 16 candidates, mean pooling | 0.944 | 70.6% | 0.926 | 68.7% |
| listwise + InfoNCE (weight 1) | 0.942 | 69.9% | 0.921 | 67.6% |
| **listwise + type-aware attention pooling** | 0.945 | 71.1% | 0.929 | 69.3% |
| CSA-Net outfit ranking hinge (margin 0.3), mean pooling | 0.930 | 68.6% | 0.908 | 66.1% |
| CSA-Net subspaces + outfit ranking hinge | 0.929 | 67.8% | 0.905 | 66.7% |
| CSA-Net subspaces + listwise loss | 0.943 | 70.9% | 0.925 | 68.8% |

**Screening winner on validation: listwise + attention pooling** (+0.002 val AUC, +0.7 FITB over InfoNCE seed 0).
The ranking matches the ResNet-18 track: listwise ≈ CSA + listwise > InfoNCE ≈ listwise + InfoNCE ≫ outfit ranking hinge.
The margins here are within seed noise (±0.002). Seeds 1–2 of the winner follow.
