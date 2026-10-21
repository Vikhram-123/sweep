# Hard negatives (step 3) — Model A on FashionCLIP, 3 seeds each

Selection metric: **validation compatibility AUC**. Test is reported only.

| negatives | val AUC | val FITB | test AUC | test FITB |
|---|---|---|---|---|
| **random same-type (plain Model A)** | **0.936** | 68.7% | **0.915 ± 0.001** | 67.0% ± 0.4 |
| batch semi-hard (closest in-batch candidate farther than the positive) | 0.934 | 69.3% | 0.912 ± 0.002 | 66.9% ± 0.8 |
| 50% kNN (positive's 50 nearest same-type items) + 50% random | 0.911 | 66.6% | 0.888 ± 0.002 | 64.1% ± 0.1 |
| 100% kNN | 0.752 | 53.8% | 0.735 ± 0.010 | 52.8% ± 0.7 |
| batch hardest | 0.735 | 39.9% | 0.719 ± 0.027 | 37.9% ± 3.5 |

**Winner on validation: plain random negatives.** Semi-hard is within noise (−0.002 val AUC, +0.6 val
FITB, test FITB 66.9 ± 0.8 vs 67.0 ± 0.4); every harder scheme is worse.

## Why harder negatives hurt here

1. **False negatives.** The items most similar to the positive are usually *also* compatible with the
   anchor: if a white tee goes with these jeans, it goes with the five most similar pairs of jeans too.
   kNN negatives (even with near-duplicates excluded at cos > 0.95) teach the model to push away
   items that should be close. That directly contradicts the positive pairs, so the objective can't
   be satisfied: with 100% kNN negatives, 64–70% of triplets still violate the margin after
   training, against about 45% for random negatives.
2. **Train/test mismatch.** The benchmark's fake outfits are built from *random* same-type items.
   A model trained only on subtle near-misses never learns the coarse "this clearly doesn't belong"
   signal that the test actually measures. Mixing 50% random negatives back in recovers most of
   the loss (0.888) but not all of it.
3. **Hardest-negative collapse.** Always picking the closest candidate drove all embeddings to a
   single point: training loss sat at exactly 0.200 = the margin, with 100% of triplets active, which
   only happens when d(a,p) = d(a,n) = 0. This is the known failure that motivated *semi-hard*
   mining (FaceNet, Schroff et al. 2015). Semi-hard avoids collapse, which is why it's the only
   mined variant that roughly matches random sampling.

## Takeaway

Hard negatives help when the task's negatives are hard and labels are clean. Here, compatibility
labels are noisy and positive-unlabelled (an item absent from an outfit isn't *incompatible*),
and the test negatives are random. Easy random negatives match the evaluation, and they rarely
contain false negatives.
