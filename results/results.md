# Outfit compatibility — Polyvore Outfits, disjoint test set

| model | source | compat AUC | FITB acc | notes |
|---|---|---|---|---|
| random | ours | 0.502 | 24.6% |  |
| rules | ours | 0.505 | 23.7% |  |
| rules_clothing | ours | 0.512 | 24.3% |  |
| clip_cosine | ours | 0.753 | 52.4% |  |
| paper: Bi-LSTM + VSE (Han et al.) | published | 0.620 | 39.4% | Vasileva et al. 2018, Table 5 (Polyvore Outfits-D) |
| paper: SiameseNet | published | 0.810 | 51.8% | Vasileva et al. 2018, Table 5 (Polyvore Outfits-D) |
| paper: type-aware CSN T1:1 | published | 0.820 | 52.5% | Vasileva et al. 2018, Table 5 (Polyvore Outfits-D) |
| paper: type-aware CSN + VSE + Sim + Metric (512-D) | published | 0.840 | 55.2% | Vasileva et al. 2018, Table 5 (Polyvore Outfits-D) |
