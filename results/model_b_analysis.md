# Why type-awareness (Model B) didn't help over Model A

All numbers: Polyvore Outfits disjoint, mean over 3 seeds. Selection is by validation; test is
reported only.

| model | what changes vs Model A | val AUC | test AUC | test FITB |
|---|---|---|---|---|
| Model A | — (one shared 128-d space) | **0.936** | **0.915 ± 0.001** | **67.0% ± 0.4** |
| Model B | + a learned diagonal mask per type pair (66 masks, +8.4k params) | 0.934 | 0.911 ± 0.003 | 66.5% ± 0.3 |
| Model B, masks at 10× LR | masks free to move (no weight decay) | 0.934 | 0.913 ± 0.002 | 66.8% ± 0.3 |
| Model B, full projection | + a full 128×128 matrix per type pair (+1.08M params) | 0.919 | 0.894 ± 0.002 | 64.0% ± 0.6 |

On validation, the more type-specific capacity we add, the worse the model gets.

## The evidence, step by step

1. **With the default setup, the masks barely move.** They're initialised to 1, so Model B starts
   as Model A. After training, each mask's cosine to the all-ones vector is 0.95–0.98, and
   mask-to-mask cosine has a median of 0.978. Every type pair is effectively still using the shared
   space. Per-type-pair AUC changes are noise around zero (−0.019 to +0.014; 27 of 58 pairs
   improve; pair-weighted mean −0.0004). See `type_pairs.md`.

2. **Letting the masks move doesn't help either.** With a 10× learning rate and no weight decay, the
   masks genuinely diverge (median mask-to-mask cosine 0.68, some entries negative). So the type
   pairs really do get different spaces, yet validation AUC doesn't improve (0.934). The type-specific
   spaces are learnable; they just aren't useful here. See `type_pairs_masklr10.md`.

3. **Where they move, they overfit.** Across type pairs (masklr10 run, test pairs):
   - Spearman(training pairs of that type, how far the mask moved) = **−0.92**: rare type pairs move
     their masks the most, because each mask only gets gradient from its own few hundred examples;
   - Spearman(training pairs, AUC gain) = **+0.45**: common pairs gain a little and rare pairs lose;
   - Spearman(mask movement, AUC gain) = **−0.50**: the more a mask moved, the worse its pair got.

   The worst case is accessories–scarves (656 test pairs): its mask moved the most and it lost
   0.031 AUC.

4. **More capacity per pair overfits more.** Full 128×128 matrices (66 × 16k = 1.08M extra
   parameters, 6.6× Model A's total) fit the training triplets about as well as Model A (final
   "active" fraction 0.43–0.46 vs 0.43) but lose 0.017 validation AUC. That's the classic signature
   of overfitting. Vasileva et al. saw a tiny *gain* from full projections (Table 7b, +0.01) because
   their ResNet-18 embedding was being learned at the same time.

## Why the paper needed type-awareness and we don't

The paper's argument: in a single space, if shoes must be close to both a top and some jeans, the
top and jeans are pulled together too, even if they don't match. Type-specific spaces break that
transitivity. Two things make this matter less here:

- **A much richer input.** FashionCLIP's 512-d features already separate style, formality, color
  and material. A shared MLP into 128 dimensions has enough room to satisfy "near the shoes" for
  both the top and the jeans without collapsing them onto each other. The paper's embedding
  started from ImageNet ResNet-18 features, which know far less about fashion.
- **The data is spread thin.** 66 type pairs, with a median of about 1.5k training co-occurrences
  each and a minimum of 234. A shared space pools all of that evidence; per-pair parameters split it
  up, and the rare pairs overfit.

**Bottom line:** type-awareness was a fix for a weak, jointly-trained embedding. With a strong
frozen fashion encoder, a single shared space is both simpler and slightly better on validation, so
Model A stays the baseline for every later experiment.
