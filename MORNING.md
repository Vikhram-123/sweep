# Good morning — overnight results

Branch `compat-model`, all pushed to GitHub. Every step ran to completion. Nothing was skipped, and
every trained result is 3 seeds. Every choice was made on **validation**; test numbers are reported
but were never used to pick anything.

## TL;DR

1. **The encoder is the whole story.** Same Model A config, only the frozen encoder changes:
   ImageNet ResNet-18 **0.801** → OpenAI CLIP **0.886** → FashionCLIP **0.915** test AUC. With the
   papers' own backbone, our model lands where their baselines do (SiameseNet 0.81). CLIP alone
   beats every published disjoint result.
2. **One idea helped: adding the product title as text.** Image + title embedding, concatenated:
   **0.926 AUC, 69.1% FITB** (vs 0.915 / 67.0%). Averaging the two helped less.
3. **Everything else didn't:** type-aware projections (Model B), hard negatives (kNN ones were badly
   worse, and hardest-in-batch collapsed), squared distance, the hyperparameter sweep (its winner
   was noise), and the outfit transformer (0.904 AUC but only 59% FITB).
4. **Best model (validation):** Model A on FashionCLIP image + title text (concat). Best model *the
   app can use*: plain image-only Model A, because scanned closet items have no titles. That one
   is already exported and integrated on the `compat-integration` branch, unmerged as you asked.

## What ran, in plain English

| step | question | answer (by validation) |
|---|---|---|
| 1 | Why didn't type-aware projections (Model B) help? | FashionCLIP + a shared space already has room; per-type-pair parameters split thin data and overfit (rare pairs move most and lose most, ρ = −0.92). Full 128×128 projections are clearly worse (0.894). |
| 2 | How much is the encoder? | Nearly everything: ResNet-18 0.801, CLIP 0.886, FashionCLIP 0.915. About 3/4 of the gain over ResNet-18 is web-scale CLIP pretraining, about 1/4 is fashion fine-tuning. |
| 3 | Do hard negatives help? | No. kNN negatives are mostly *false* negatives (similar jeans also go with the top) → 0.735. Hardest-in-batch collapsed all embeddings to one point → 0.719. Semi-hard ≈ random (0.912). Random wins. |
| 4 | Squared vs plain distance? | A tie once each gets its own margin (plain at 0.2 ≈ squared at 0.4, the same effective gap). Plain kept. |
| 5 | Does product text help? | Yes: concat 0.926 / 69.1% beats mean 0.923 / 68.0% beats image-only 0.915 / 67.0%. Text alone gets 0.819. |
| 6 | Better hyperparameters? | The 16-trial random search "winner" (out_dim 256, dropout 0.2) was +0.0025 val AUC on one seed. Over 3 seeds it was exactly equal to the default. The surface is flat; only margin mattered (0.2 ≫ 0.1 or 0.3). |
| 7 | Combined model (text + sweep winner) | 0.945 val, 0.928 ± 0.001 test AUC, 69.0% FITB: the same as text-concat alone on validation. |
| 8 | Outfit transformer | 0.904 AUC (with text) and 0.892 (image only), but FITB only about 59%. Pairwise wins. The first version sat at chance (see "Problems" below). |

## All results (Polyvore Outfits disjoint)

Trained rows: mean ± std over 3 seeds. Validation shares 26% of its items with train, so it runs
about 0.02 higher than test; it's used for selection only. Also in `results/results.md` and
`results/results.csv`.

**Baselines**

| model | encoder | seeds | val AUC | val FITB | **test AUC** | **test FITB** | notes |
|---|---|---|---|---|---|---|---|
| fashionclip_cosine | FashionCLIP (frozen) | 1 |  |  | 0.753 | 52.4% | mean pairwise cosine of frozen fashionclip embeddings |
| openai_clip_cosine | CLIP ViT-B/32 (frozen) | 1 |  |  | 0.638 | 42.0% | mean pairwise cosine of frozen clip embeddings |
| random | — | 1 |  |  | 0.502 | 24.6% |  |
| resnet18_cosine | ResNet-18 ImageNet (frozen) | 1 |  |  | 0.581 | 33.5% | mean pairwise cosine of frozen resnet18 embeddings |
| rules | color tags + FashionCLIP zero-shot | 1 |  |  | 0.505 | 23.7% |  |
| rules_clothing | color tags + FashionCLIP zero-shot | 1 |  |  | 0.512 | 24.3% |  |
| text_cosine | FashionCLIP text (frozen) | 1 |  |  | 0.657 | 41.7% | mean pairwise cosine of frozen text embeddings |

**Model A / Model B (FashionCLIP)**

| model | encoder | seeds | val AUC | val FITB | **test AUC** | **test FITB** | notes |
|---|---|---|---|---|---|---|---|
| model_a | FashionCLIP (frozen) | 3 | 0.936 | 68.7% | 0.915 ± 0.001 | 67.0% ± 0.4 | MLP 512-256-128 on frozen FashionCLIP, triplet loss, same-type negatives |
| model_b | FashionCLIP (frozen) | 3 | 0.934 | 68.1% | 0.911 ± 0.003 | 66.5% ± 0.3 | Model A + learned diagonal mask per type pair (66 pairs), masks init to 1 |
| model_b_full | FashionCLIP (frozen) | 3 | 0.919 | 66.4% | 0.894 ± 0.002 | 64.0% ± 0.6 | Model B with a full 128x128 projection per type pair (paper Table 7b ablation) |
| model_b_masklr10 | FashionCLIP (frozen) | 3 | 0.934 | 68.1% | 0.913 ± 0.002 | 66.8% ± 0.3 | Model B, masks at 10x learning rate and no weight decay |

**Encoder ablation (Model A, identical config)**

| model | encoder | seeds | val AUC | val FITB | **test AUC** | **test FITB** | notes |
|---|---|---|---|---|---|---|---|
| model_a_clip | CLIP ViT-B/32 (frozen) | 3 | 0.911 | 64.3% | 0.886 ± 0.001 | 61.9% ± 0.2 | Model A on frozen OpenAI CLIP ViT-B/32 features |
| model_a_resnet18 | ResNet-18 ImageNet (frozen) | 3 | 0.829 | 51.3% | 0.801 ± 0.001 | 49.5% ± 0.4 | Model A on frozen ImageNet ResNet-18 features |

**Hard negatives (Model A, FashionCLIP)**

| model | encoder | seeds | val AUC | val FITB | **test AUC** | **test FITB** | notes |
|---|---|---|---|---|---|---|---|
| model_a_neg_hard | FashionCLIP (frozen) | 3 | 0.735 | 39.9% | 0.719 ± 0.027 | 37.9% ± 3.5 | Model A, batch hardest mined negatives |
| model_a_neg_knn | FashionCLIP (frozen) | 3 | 0.752 | 53.8% | 0.735 ± 0.010 | 52.8% ± 0.7 | Model A, negatives from the positive's 50 nearest same-type neighbours (cos <= 0.95) |
| model_a_neg_knn50 | FashionCLIP (frozen) | 3 | 0.911 | 66.6% | 0.888 ± 0.002 | 64.1% ± 0.1 | Model A, half kNN negatives, half random same-type |
| model_a_neg_semihard | FashionCLIP (frozen) | 3 | 0.934 | 69.3% | 0.912 ± 0.002 | 66.9% ± 0.8 | Model A, batch semi-hard mined negatives (closest farther than the positive) |

**Squared vs plain distance (Model A, FashionCLIP)**

| model | encoder | seeds | val AUC | val FITB | **test AUC** | **test FITB** | notes |
|---|---|---|---|---|---|---|---|
| model_a_sq | FashionCLIP (frozen) | 3 | 0.935 | 68.1% | 0.914 ± 0.002 | 66.4% ± 0.2 | Model A, squared distance in loss and eval, margin 0.4 (tuned on val) |

**Image + title text (Model A, FashionCLIP)**

| model | encoder | seeds | val AUC | val FITB | **test AUC** | **test FITB** | notes |
|---|---|---|---|---|---|---|---|
| model_a_text_concat | FashionCLIP (frozen) + title text (concat) | 3 | 0.945 | 70.4% | 0.926 ± 0.003 | 69.1% ± 0.3 | Model A, features=concat |
| model_a_text_mean | FashionCLIP (frozen) + title text (mean) | 3 | 0.938 | 68.6% | 0.923 ± 0.003 | 68.0% ± 0.4 | Model A, features=mean |
| model_a_text_only | FashionCLIP text (frozen) | 3 | 0.834 | 50.6% | 0.819 ± 0.001 | 49.6% ± 0.5 | Model A, encoder=text |

**Combined model**

| model | encoder | seeds | val AUC | val FITB | **test AUC** | **test FITB** | notes |
|---|---|---|---|---|---|---|---|
| combined | FashionCLIP (frozen) + title text (concat) | 3 | 0.945 | 70.4% | 0.928 ± 0.001 | 69.0% ± 0.4 | Model A + title text (concat) + sweep winner (out_dim 256, dropout 0.2) |

**Outfit transformer**

| model | encoder | seeds | val AUC | val FITB | **test AUC** | **test FITB** | notes |
|---|---|---|---|---|---|---|---|
| transformer | FashionCLIP (frozen) | 3 | 0.913 | 60.5% | 0.892 ± 0.003 | 58.7% ± 0.4 | outfit transformer, image-only inputs |
| transformer_text | FashionCLIP (frozen) + title text (concat) | 3 | 0.922 | 60.4% | 0.904 ± 0.002 | 58.9% ± 0.4 | outfit transformer, image + title text (concat) inputs |

**Published (Polyvore Outfits disjoint)**

| model | encoder | seeds | val AUC | val FITB | **test AUC** | **test FITB** | notes |
|---|---|---|---|---|---|---|---|
| Bi-LSTM + VSE (Han et al. 2017) | ResNet-18 (fine-tuned) + text |  |  |  | 0.620 | 39.4% | Vasileva et al. 2018, Table 5 |
| SiameseNet | ResNet-18 (fine-tuned) |  |  |  | 0.810 | 51.8% | Vasileva et al. 2018, Table 5 |
| Type-aware CSN T1:1 | ResNet-18 (fine-tuned) |  |  |  | 0.820 | 52.5% | Vasileva et al. 2018, Table 5 |
| Type-aware CSN + VSE + Sim + Metric (512-D) | ResNet-18 (fine-tuned) + text |  |  |  | 0.840 | 55.2% | Vasileva et al. 2018, Table 5 |
| CSA-Net (Lin et al. 2020) | ResNet-18 (fine-tuned) |  |  |  | 0.870 | 59.3% | CVPR 2020, Table 1 |
| OutfitTransformer (Sarkar et al. 2023) | ResNet-18 (fine-tuned) + text |  |  |  | 0.880 | 59.5% | WACV 2023, Tables 1–2 |

Details per experiment: `results/model_b_analysis.md`, `encoder_ablation.md`, `hard_negatives.md`,
`distance.md`, `text_features.md`, `sweep.md`, `TRANSFORMER.md`; leakage audit in `leakage.md`.

## What worked, what didn't, and my best guess why

**Worked**
- *A strong frozen encoder.* FashionCLIP's features already separate style, material, formality and
  color, so a 164k-parameter head on top beats end-to-end ResNet-18 systems. A frozen ResNet-18 with
  the same head lands at the published SiameseNet level, which tells me the training recipe is
  sound and the encoder is the lever.
- *Title text, concatenated.* Names say what a 224-px product shot can't: material, fit, occasion.
  Concatenating keeps image and text separate so the MLP can weight each. Averaging blurs them,
  because image and text vectors don't coincide even in CLIP's shared space.
- *Random same-type negatives.* They match how the test's fake outfits are built, and they rarely
  contain false negatives.

**Didn't work**
- *Type-aware spaces:* a fix for weak, jointly trained embeddings. With rich frozen features they
  only add parameters, and rare type pairs overfit.
- *Hard negatives:* compatibility labels are "positive-unlabelled". An item not in an outfit isn't
  incompatible, so "hard" negatives are often good matches. Pushing them away fights the positives.
- *Squared distance:* on unit vectors it's a monotone transform of plain distance, and margin tuning
  makes them equivalent.
- *Hyperparameter tuning:* flat surface. Picking the best of 16 noisy single-seed trials selects for
  luck (a winner's curse), which the 3-seed rerun exposed.
- *Transformer:* 17k outfits is little data for attention, the inputs are frozen, and the test fakes
  are random swaps, which pairwise averaging already handles. Its FITB is far behind because pairwise
  triplets *are* FITB-style comparisons (386k per epoch), whereas the transformer gets one real/fake
  label per outfit.

## Problems hit overnight, and what I did

- **Model B full-projection ablation, first attempt: killed for memory.** Evaluation gathered one
  128×128 matrix per pair: 65k pairs × 64 KB = 4.3 GB in one tensor, and the process reached 5.2 GB
  RSS. I killed it, rewrote the projection as one matmul per type pair (verified identical to the
  old output), and reran all 3 seeds without issue.
- **Queue overlap, about 15 minutes.** My first queue script's memory gate had a 20-minute timeout,
  so it started a small CPU training run while the ablation was still going (two CPU MLP jobs, 71%
  memory free, no harm done). I fixed the gate to never overlap jobs. Everything after that ran
  strictly one at a time.
- **Transformer v1 trained at chance** (loss = ln 2, AUC 0.50). It could still overfit 256 outfits,
  so the code path was fine. The cause was scale: item tokens had norm 0.45 vs 11 for the type
  embeddings, which carry no real-vs-fake signal. A LayerNorm on the item projection fixed it, and
  all 6 runs were redone. Full story in `TRANSFORMER.md`.
- **Transformer + text runs hit the 60-epoch cap** (best epochs 46–59, still creeping up by about
  0.003 per 10 epochs). They're slightly undertrained, but not enough to change the ranking.
- **Small fixes:** a baseline name collision (`clip_cosine` would have been overwritten by OpenAI
  CLIP; it's now `fashionclip_cosine` / `openai_clip_cosine`), and run notes containing ": " were
  parsed by YAML as dictionaries (fixed in the saved JSON).
- **Regression check:** the rewritten training script reproduces the original Model A seed 0
  *bit-for-bit* (val 0.936661, test 0.915832, best epoch 10).
- **Your "don't change the app or exported model" rule.** Before your overnight message arrived I had
  already finished milestone 6 (numpy export, `sweep/compat.py`, model/rules toggle, 7 tests). I
  moved it, untouched, to its own branch, **`compat-integration`** (one commit on top of the
  experiments' start), and kept it out of `compat-model`. Nothing in the app or the export changed
  overnight.

## Which model I'd pick, and why

- **On the benchmark: Model A on FashionCLIP image + title text (concat), default hyperparameters.**
  It's tied for best validation (0.945 AUC / 70.4% FITB) with the "combined" model, and simpler
  (128-d output instead of 256). Test: 0.926 ± 0.003 AUC, 69.1% ± 0.3 FITB.
- **For Sweep today: image-only Model A** (val 0.936; test 0.915 / 67.0%). The app's items come from
  video and have no product titles. The text model would be getting inputs it never saw in training.
  This model is already exported to numpy and wired into the app on `compat-integration`.

## Interview questions, 3 per experiment

**1. Model B (type-aware)**
1. What problem do type-specific spaces solve in Vasileva et al., and why didn't that problem bite here?
2. How did you tell "the masks didn't train" apart from "type spaces don't help"? (A 10× mask
   learning rate: the masks diverged, but AUC didn't move.)
3. Why do rare type pairs overfit, and what evidence do you have? (ρ = −0.92 between training count
   and mask movement; movement vs gain ρ = −0.50.)

**2. Encoder ablation**
1. How much of your gain over the papers is the encoder vs your method? (With ResNet-18 you match
   SiameseNet; CLIP adds +0.085 and fashion fine-tuning +0.029.)
2. Why compare CLIP vs FashionCLIP specifically? (Same architecture and input; the only difference is
   fashion fine-tuning, so it's a clean ablation.)
3. What can't this ablation rule out? (CLIP's pretraining may have seen Polyvore pages: pretraining
   exposure, as opposed to train/test leakage.)

**3. Hard negatives**
1. Why did nearest-neighbour negatives make things much worse? (False negatives; and a mismatch with
   random test negatives.)
2. What is the collapse failure of hardest-negative mining, and how did you detect it? (Loss pinned at
   exactly the margin with 100% active triplets, so all distances are 0.)
3. When *would* hard negatives help? (Clean labels and hard test negatives, e.g. face ID. Not
   positive-unlabelled compatibility data with random test fakes.)

**4. Squared vs plain distance**
1. On L2-normalised vectors, how are the two related? (d² = 2 − 2cos; both monotone in cosine.)
2. Why must the margin be retuned when switching? (Same margin, different effective gap; a 0.2 plain
   margin ≈ 0.4 squared near d ≈ 1, which is what tuning found.)
3. Where could they still differ? (How the outfit mean weights the worst pair, and how the margin
   varies across the space.)

**5. Text features**
1. Why did concatenation beat averaging? (Averaging forces both into one point and blurs them;
   the modality gap. Concatenation lets the MLP weight each.)
2. What shortcut could inflate the text gain, and how would you test it? (Same-shop naming style;
   strip brands and boilerplate and re-run.)
3. Why not ship the text model? (No titles in the app: a train/serve skew. You'd need generated text
   and a re-evaluation.)

**6. Hyperparameter sweep**
1. Why random search over grid search? (16 draws explore every hyperparameter at many values;
   a grid wastes budget on the unimportant ones.)
2. The sweep winner gave +0.0025 on one seed and +0.000 over three. What happened? (Winner's curse:
   max over noisy trials. Always re-run the winner with fresh seeds.)
3. Which hyperparameter mattered? (The margin: the top 4 trials all used 0.2; 0.1 and 0.3 sit at the
   bottom.)

**7. Combined model**
1. How did you decide what to stack? (Only changes that improved *validation* in their own
   experiment: text concat, and the sweep winner, which turned out to be neutral.)
2. Why report test only once per configuration? (So test can never influence a choice; otherwise it
   becomes a second validation set and the number is optimistic.)
3. Why does validation run about 0.02 above test? (26% of validation items appear in train;
   test-train overlap is about 0.1%.)

**8. Outfit transformer**
1. Walk through the architecture (item tokens + type embeddings, [CLS], no positions, padding mask,
   BCE).
2. Why did it lose on FITB by about 8 points while being close on AUC? (Training signal: about 386k
   FITB-like triplet comparisons vs about 34k outfit labels per epoch.)
3. It first trained at chance: how did you debug it? (Overfit a tiny set → scales → LayerNorm.)

## What I'd try next

1. **Fine-tune the encoder's last blocks** (FashionCLIP vision tower, small LR) with Model A's loss.
   That's where every published gain came from, and we've only used frozen features.
2. **Deployable text:** generate a "title" for scanned items from Sweep's own tags ("navy slim
   jeans") or FashionCLIP zero-shot captions. Retrain on the same kind of generated text for
   Polyvore items, so train and serve match. This also tests the naming-style shortcut.
3. **Harder evaluation:** build a hard-negative compatibility test for Polyvore Outfits, e.g. fakes
   that swap a single item for a *similar-looking* same-type item. (The authors' Hugging Face repo
   also has a `maryland_polyvore_hardneg` set, but it's for the older Maryland Polyvore dataset and
   wasn't downloaded.) That's where a transformer's context-awareness should
   actually pay off, and where it's worth revisiting with more negatives per outfit and a FITB-style
   (contrastive, 1-of-4) training objective.
4. **Merge `compat-integration` after a look**, and use the in-app toggle plus 👍/👎 (each vote
   records which scorer produced the outfit) as a small real-world A/B test of model vs rules on
   your own closet.
