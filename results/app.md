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

## Steps 2–4 — 3-seed confirmation of the winner

| model | seeds | val AUC | val FITB | **test AUC** | **test FITB** |
|---|---|---|---|---|---|
| InfoNCE (step 1) | 3 | 0.943 | 70.1% | 0.921 ± 0.001 | 68.2% ± 0.3 |
| **listwise + attention pooling** | 3 | 0.945 | 70.4% | 0.926 ± 0.003 | 69.3% ± 0.1 |

Over 3 seeds, attention pooling **wins on validation on every seed** (val 0.945 / 70.4% vs 0.943 / 70.1%). The gain is small but holds on every seed.
Test is 0.926 ± 0.003 / 69.3% ± 0.1, vs 0.921 / 68.2%. It peaks at epochs 5–6 (InfoNCE peaked at epochs 1–2), with τ ≈ 0.06.
**Carried forward: listwise + attention.**

## Phase 2 — candidate encoders (frozen embeddings + cosine baseline)

Full catalog embedded (train 71,967 / valid 14,657 / test 70,035 items, 768-d). Zero-training cosine baseline (test):

| encoder | cosine AUC | cosine FITB |
|---|---|---|
| FashionCLIP ViT-B/32 (current) | 0.753 | 52.4% |
| Marqo-FashionSigLIP ViT-B/16 | 0.763 | 54.1% |
| ZooClaw-FashionSigLIP2 | 0.760 | 54.1% |

Both SigLIP encoders beat FashionCLIP out of the box (+0.01 AUC, +1.7 FITB). This is only a hint, because
the cosine baseline has no training. The encoder is chosen after training the step 2–4 winner
(listwise + attention) on each one, on validation.

### Phase 2 — listwise + attention on each encoder (screening, seed 0)

Identical config to the step 2–4 winner; only the frozen encoder changes.

| encoder | val AUC | val FITB | test AUC | test FITB |
|---|---|---|---|---|
| FashionCLIP ViT-B/32 | 0.945 | 71.1% | 0.929 | 69.3% |
| Marqo-FashionSigLIP ViT-B/16 | 0.948 | 71.6% | 0.933 | 71.0% |
| ZooClaw-FashionSigLIP2 ViT-B/16 @384 | 0.952 | 73.0% | 0.937 | 72.1% |

Both beat FashionCLIP on validation, so both get seeds 1–2.

**ZooClaw source check.** It is published by `srpone` ("SRP"), a verified Hugging Face org (7 members) tied to the ZooClaw
platform (GitHub links point to SerendipityOneInc), with an arXiv paper (2606.27708). It is fine-tuned from
`google/siglip2-base-patch16-384` and licensed Apache-2.0. It uses the standard transformers `SiglipModel`, with no
`auto_map`, no repo `.py` files and safetensors weights only, so it needs no `trust_remote_code`. **Caveat:** its training
data isn't disclosed, so Polyvore exposure can't be ruled out. Marqo-FashionSigLIP has the same unknown about its pretraining data.
