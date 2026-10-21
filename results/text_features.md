# Image + title text (step 5) — Model A, FashionCLIP, 3 seeds each

Text = FashionCLIP's *text* encoder applied to each item's name (its `title` if present, 28% of
items, otherwise its `url_name` product slug, 100% coverage). Descriptions were not used.
Selection metric: **validation AUC**.

| input to Model A | val AUC | val FITB | test AUC | test FITB |
|---|---|---|---|---|
| text cosine, no training | — | — | 0.657 | 41.7% |
| text only (512-d) | 0.834 | 50.6% | 0.819 ± 0.001 | 49.6% ± 0.5 |
| image only (plain Model A) | 0.936 | 68.7% | 0.915 ± 0.001 | 67.0% ± 0.4 |
| image + text, **averaged** (normalise(img + txt), 512-d) | 0.938 | 68.6% | 0.923 ± 0.003 | 68.0% ± 0.4 |
| image + text, **concatenated** ([img, txt], 1024-d) | **0.945** | **70.4%** | **0.926 ± 0.003** | **69.1% ± 0.3** |

**Winner on validation: concatenation**, +0.009 val AUC over image-only and +0.011 / +2.1 FITB on test.

## What it means

- **Text carries complementary information.** On its own it's much weaker than the image (0.819 vs
  0.915), but adding it still helps, so it isn't just a noisy copy of what the image already says.
  Product names state things that are hard to see in a 224-pixel product shot: material ("chiffon",
  "leather"), fit ("slim fit", "plus size"), occasion ("bridal", "beach") and brand.
- **Concatenation beats averaging** because averaging forces both vectors into one 512-d point before
  the model sees them. Image and text embeddings of the same item aren't identical even in CLIP's
  shared space (the "modality gap"), so the average blurs both. Concatenation keeps them separate
  and lets the first MLP layer learn how much to trust each, per dimension.
- The paper also found text helps (its "VSE" term), and OutfitTransformer gains +0.01 AUC from text.
  We see a similar-sized effect.

## Caveats

- **Possible shortcut.** Items in one Polyvore outfit are often from the same shop, and shops name
  products in similar styles (e.g. "Brand from China: …" or the same brand prefix). Some of the gain
  could come from naming style rather than fashion sense. It isn't measured here; a check would be
  to strip brand names and boilerplate and re-run.
- **Not deployable in the app as-is.** Items scanned from your closet video have no product titles.
  Using this model in Sweep would need a substitute: for example, text built from the app's own tags
  ("navy slim jeans"), or FashionCLIP zero-shot captions. That would change the text distribution.
  The image-only model is the one that matches what the app can actually feed it.
