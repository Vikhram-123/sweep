# Outfit compatibility — Polyvore Outfits, disjoint split

Trained models: mean ± std over seeds. Every choice was made on **validation**; test is reported only. Validation shares 26% of its items with train, so it runs higher than test.

**Baselines**

| model | encoder | seeds | val AUC | val FITB | **test AUC** | **test FITB** | notes |
|---|---|---|---|---|---|---|---|
| fashionclip_cosine | FashionCLIP (frozen) | 1 |  |  | 0.753 | 52.4% | mean pairwise cosine of frozen fashionclip embeddings |
| random | — | 1 |  |  | 0.502 | 24.6% |  |
| rules | color tags + FashionCLIP zero-shot | 1 |  |  | 0.505 | 23.7% |  |
| rules_clothing | color tags + FashionCLIP zero-shot | 1 |  |  | 0.512 | 24.3% |  |

**Model A / Model B (FashionCLIP)**

| model | encoder | seeds | val AUC | val FITB | **test AUC** | **test FITB** | notes |
|---|---|---|---|---|---|---|---|
| model_a | FashionCLIP (frozen) | 3 | 0.936 | 68.7% | 0.915 ± 0.001 | 67.0% ± 0.4 | MLP 512-256-128 on frozen FashionCLIP, triplet loss, same-type negatives |
| model_b | FashionCLIP (frozen) | 3 | 0.934 | 68.1% | 0.911 ± 0.003 | 66.5% ± 0.3 | Model A + learned diagonal mask per type pair (66 pairs), masks init to 1 |
| model_b_masklr10 | FashionCLIP (frozen) | 3 | 0.934 | 68.1% | 0.913 ± 0.002 | 66.8% ± 0.3 | Model B, masks at 10x learning rate and no weight decay |

**Published (Polyvore Outfits disjoint)**

| model | encoder | seeds | val AUC | val FITB | **test AUC** | **test FITB** | notes |
|---|---|---|---|---|---|---|---|
| Bi-LSTM + VSE (Han et al. 2017) | ResNet-18 (fine-tuned) + text |  |  |  | 0.620 | 39.4% | Vasileva et al. 2018, Table 5 |
| SiameseNet | ResNet-18 (fine-tuned) |  |  |  | 0.810 | 51.8% | Vasileva et al. 2018, Table 5 |
| Type-aware CSN T1:1 | ResNet-18 (fine-tuned) |  |  |  | 0.820 | 52.5% | Vasileva et al. 2018, Table 5 |
| Type-aware CSN + VSE + Sim + Metric (512-D) | ResNet-18 (fine-tuned) + text |  |  |  | 0.840 | 55.2% | Vasileva et al. 2018, Table 5 |
| CSA-Net (Lin et al. 2020) | ResNet-18 (fine-tuned) |  |  |  | 0.870 | 59.3% | CVPR 2020, Table 1 |
| OutfitTransformer (Sarkar et al. 2023) | ResNet-18 (fine-tuned) + text |  |  |  | 0.880 | 59.5% | WACV 2023, Tables 1–2 |
