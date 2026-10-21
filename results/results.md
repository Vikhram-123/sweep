# Outfit compatibility — Polyvore Outfits, disjoint test set

Trained models: mean ± std over seeds (column *seeds*).

| model | source | seeds | compat AUC | FITB acc | notes |
|---|---|---|---|---|---|
| random | ours | 1 | 0.502 | 24.6% |  |
| rules | ours | 1 | 0.505 | 23.7% |  |
| rules_clothing | ours | 1 | 0.512 | 24.3% |  |
| clip_cosine | ours | 1 | 0.753 | 52.4% |  |
| model_a | ours | 3 | 0.915 ± 0.001 | 67.0% ± 0.4 | MLP 512-256-128 on frozen FashionCLIP, triplet loss, same-type negatives |
| model_b | ours | 3 | 0.911 ± 0.003 | 66.5% ± 0.3 | Model A + learned diagonal mask per type pair (66 pairs), masks init to 1 |
| paper: Bi-LSTM + VSE (Han et al.) | published |  | 0.620 | 39.4% | Vasileva et al. 2018, Table 5; ResNet-18 + text |
| paper: SiameseNet | published |  | 0.810 | 51.8% | Vasileva et al. 2018, Table 5; ResNet-18 |
| paper: type-aware CSN T1:1 | published |  | 0.820 | 52.5% | Vasileva et al. 2018, Table 5; ResNet-18 |
| paper: type-aware CSN + VSE + Sim + Metric (512-D) | published |  | 0.840 | 55.2% | Vasileva et al. 2018, Table 5; ResNet-18 + text |
| CSA-Net (Lin et al. 2020) | published |  | 0.870 | 59.3% | CVPR 2020, Table 1; ResNet-18, 64-D |
| OutfitTransformer (Sarkar et al. 2023) | published |  | 0.880 | 59.5% | WACV 2023, Tables 1–2; ResNet-18 + text |
