# Outfit compatibility — Polyvore Outfits, disjoint split

Trained models: mean ± std over seeds. Every choice was made on **validation**; test is reported only. Validation shares 26% of its items with train, so it runs higher than test.

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

**Published (Polyvore Outfits disjoint)**

| model | encoder | seeds | val AUC | val FITB | **test AUC** | **test FITB** | notes |
|---|---|---|---|---|---|---|---|
| Bi-LSTM + VSE (Han et al. 2017) | ResNet-18 (fine-tuned) + text |  |  |  | 0.620 | 39.4% | Vasileva et al. 2018, Table 5 |
| SiameseNet | ResNet-18 (fine-tuned) |  |  |  | 0.810 | 51.8% | Vasileva et al. 2018, Table 5 |
| Type-aware CSN T1:1 | ResNet-18 (fine-tuned) |  |  |  | 0.820 | 52.5% | Vasileva et al. 2018, Table 5 |
| Type-aware CSN + VSE + Sim + Metric (512-D) | ResNet-18 (fine-tuned) + text |  |  |  | 0.840 | 55.2% | Vasileva et al. 2018, Table 5 |
| CSA-Net (Lin et al. 2020) | ResNet-18 (fine-tuned) |  |  |  | 0.870 | 59.3% | CVPR 2020, Table 1 |
| OutfitTransformer (Sarkar et al. 2023) | ResNet-18 (fine-tuned) + text |  |  |  | 0.880 | 59.5% | WACV 2023, Tables 1–2 |
