# Method track — ImageNet ResNet-18, image only

Equal footing with the papers: torchvision `IMAGENET1K_V1` ResNet-18 (the exact weights the reference code
loads), images only, official disjoint split and FITB files. Selection on validation; test reported only.
Trained rows: mean ± std over 3 seeds. **Phase 1 uses frozen features**, so the backbone is not
fine-tuned, a handicap relative to every published row (they train ResNet-18 end to end).

## Published targets (Polyvore Outfits disjoint)

| method | ResNet-18 | test AUC | test FITB |
|---|---|---|---|
| SiameseNet (Vasileva et al. 2018) | fine-tuned, 112 px, layers 1–3, 64-d | 0.81 | 51.8% |
| Type-aware CSN T1:1, image only (Vasileva et al. 2018) | fine-tuned | 0.82 | 52.5% |
| CSA-Net (Lin et al. 2020) | fine-tuned, 64-d | 0.87 | 59.3% |

## Step 0 — which frozen ResNet-18 layout (Model A, 64-d output)

| features | cosine AUC | cosine FITB | val AUC | val FITB | **test AUC** | **test FITB** |
|---|---|---|---|---|---|---|
| torchvision ResNet-18, paper layout (112 px, layers 1–3, 256-d) | 0.588 | 36.7% | 0.852 | 54.8% | 0.822 ± 0.002 | 52.2% ± 0.3 |
| torchvision ResNet-18, standard (224 px, full, 512-d) | 0.577 | 33.2% | 0.831 | 51.4% | 0.804 ± 0.002 | 49.7% ± 0.4 |

**Chosen on validation: the paper layout** (112 px, layer4 dropped). It is also the layout the papers fine-tune, so Phase 3 compares like with like. Layer-3 features (mid-level: texture, color, pattern) carry more compatibility signal than layer-4's object-category features, which are tuned to tell ImageNet classes apart, not styles.


## Step 1 — InfoNCE vs triplet (frozen paper-layout ResNet-18, 64-d)

| model | seeds | val AUC | val FITB | **test AUC** | **test FITB** |
|---|---|---|---|---|---|
| triplet (Model A) | 3 | 0.852 | 54.8% | 0.822 ± 0.002 | 52.2% ± 0.3 |
| **InfoNCE**, in-batch same-category negatives, learned τ | 3 | 0.885 | 58.8% | 0.847 ± 0.004 | 56.1% ± 0.1 |

InfoNCE helps a lot on ResNet-18: +0.033 validation AUC, and test 0.847 / 56.1% (vs 0.822 / 52.2%).
It now beats SiameseNet (0.81/51.8) and CSN (0.82/52.5) with a frozen backbone. Learned temperature
τ ≈ 0.25: weaker features → a softer softmax over candidates. Best epochs 12–18.


## Steps 2–4 — outfit-level losses and pooling (screening, seed 0)

| model | val AUC | val FITB | test AUC | test FITB |
|---|---|---|---|---|
| InfoNCE (step-1 winner, seed 0) | 0.883 | 58.3% | 0.851 | 56.0% |
| listwise FITB loss, 16 candidates, mean pooling | 0.884 | 59.0% | 0.850 | 56.4% |
| listwise + InfoNCE (weight 1) | 0.876 | 58.4% | 0.843 | 55.6% |
| **listwise + type-aware attention pooling** | 0.886 | 59.2% | 0.857 | 57.1% |
| CSA-Net outfit ranking hinge (margin 0.3), mean pooling | 0.856 | 55.1% | 0.822 | 52.8% |
| CSA-Net subspaces + outfit ranking hinge | 0.844 | 53.6% | 0.809 | 51.7% |
| CSA-Net subspaces + listwise loss | 0.882 | 58.0% | 0.844 | 55.7% |

**Screening winner on validation: listwise + attention pooling** (+0.004 val AUC, +0.9 FITB over InfoNCE seed 0).
The listwise loss alone ties InfoNCE, and adding InfoNCE on top hurts. The outfit ranking hinge is clearly
worse (−0.03 to −0.04 val AUC), with or without CSA subspaces. CSA subspaces with the listwise loss come close to plain listwise.
The gaps between the top three rows are about the size of seed noise. Seeds 1–2 of the winner follow.
