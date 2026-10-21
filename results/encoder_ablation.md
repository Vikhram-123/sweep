# Encoder ablation — same Model A config, only the frozen image encoder changes

Polyvore Outfits disjoint **test**. Model A: mean ± std over 3 seeds (validation AUC shown for reference; nothing here was selected on test). Cosine = mean pairwise cosine of the frozen embeddings, no training.

| encoder (frozen) | cosine AUC | cosine FITB | Model A val AUC | **Model A test AUC** | **Model A test FITB** | gain from training (AUC) |
|---|---|---|---|---|---|---|
| ResNet-18, ImageNet-1k (`microsoft/resnet-18`) | 0.581 | 33.5% | 0.829 | 0.801 ± 0.001 | 49.5% ± 0.4 | +0.221 |
| CLIP ViT-B/32 (OpenAI) (`openai/clip-vit-base-patch32`) | 0.638 | 42.0% | 0.911 | 0.886 ± 0.001 | 61.9% ± 0.2 | +0.248 |
| FashionCLIP (CLIP ViT-B/32 fine-tuned on fashion) (`patrickjohncyh/fashion-clip`) | 0.753 | 52.4% | 0.936 | 0.915 ± 0.001 | 67.0% ± 0.4 | +0.162 |

**Published, Polyvore Outfits disjoint (ResNet-18 fine-tuned end to end):**

| method | encoder | test AUC | test FITB |
|---|---|---|---|
| SiameseNet (Vasileva et al. 2018) | ResNet-18 (fine-tuned) | 0.81 | 51.8% |
| Type-aware best (Vasileva et al. 2018) | ResNet-18 (fine-tuned) + text | 0.84 | 55.2% |
| CSA-Net (Lin et al. 2020) | ResNet-18 (fine-tuned) | 0.87 | 59.3% |
| OutfitTransformer (Sarkar et al. 2023) | ResNet-18 (fine-tuned) + text | 0.88 | 59.5% |

## What this shows

- **The encoder explains almost all of the gap to the published numbers.** With the papers' own
  backbone (ImageNet ResNet-18, frozen), Model A gets 0.801 AUC. That's on par with the paper's
  SiameseNet (0.81), which *fine-tunes* the same ResNet-18. So our training recipe isn't doing
  anything magic: with the published features it lands where the published baselines do. The
  published methods' extra +0.06–0.07 (CSA-Net, OutfitTransformer) comes from end-to-end fine-tuning
  and better architectures on top of a weak backbone.
- **Web-scale image–text pretraining is the big jump.** OpenAI CLIP, with no fashion-specific training,
  gives 0.886 AUC and 61.9% FITB, already above every published disjoint result (best 0.88 / 59.5%),
  with a frozen encoder and a 164k-parameter head. ResNet-18 → CLIP is worth +0.085 AUC and +12.4
  points of FITB.
- **Fashion fine-tuning adds a smaller, real gain on top.** FashionCLIP is CLIP ViT-B/32 fine-tuned on
  about 800k fashion products. Same architecture, same input, and it adds +0.029 AUC and +5.1 FITB
  over CLIP. Seed std is ≤ 0.001 AUC, so the gaps between all three encoders are far beyond noise.
- **Training matters more the less "fashion-aware" the raw space is.** Model A lifts CLIP by +0.248
  AUC but FashionCLIP by only +0.162, because FashionCLIP's raw cosine already captures more of
  "goes together" (0.753 vs 0.638 zero-shot).
- **Caveat:** CLIP's web-scraped pretraining data may include Polyvore pages, which can't be ruled out
  from here. The train/test leakage audit (`leakage.md`) rules out *dataset* leakage, not
  *pretraining* exposure.
