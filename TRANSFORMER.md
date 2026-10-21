# The outfit transformer, explained

Code: `compat/transformer.py`. Config: `compat/configs/transformer.yaml`. Results: see the
"Outfit transformer" section of `results/results.md` and `MORNING.md`.

## The one-sentence version

Instead of judging an outfit by averaging how well each *pair* of items goes together, the
transformer looks at **all the items at once** and outputs a single "is this a real outfit?" score.

## Why pairwise scoring has a ceiling

Model A scores an outfit as the mean of pairwise scores: top–jeans, top–shoes, jeans–shoes, and so
on. That has two blind spots:

1. **No context.** Whether a red bag works depends on the rest of the outfit. With a black-and-white
   outfit it can be the one statement piece. With a red top and red shoes it might be one red thing
   too many. A pair score for "red bag + black jeans" can't know about the red top.
2. **Averaging hides the odd one out.** Five items that go together plus one that clashes still
   average out fine. A real stylist would notice the one clashing item immediately.

The transformer can, in principle, learn both: every item's representation is updated based on
every other item in the outfit before the final score is computed.

## The architecture, step by step

Take an outfit of, say, 5 items. Each item already has a frozen 512-d FashionCLIP embedding.

1. **Item tokens.** Each item's embedding (512-d image, or 1024-d image + title text) goes through
   one linear layer, which shrinks it to 128 numbers, followed by a LayerNorm. This is its "token",
   the same idea as a word token in a language model. (Why the LayerNorm matters: see "A bug worth
   telling" below.)

2. **Type embeddings.** We add a learned 128-d vector for the item's category (tops, shoes, bags,
   ...; 11 types). This tells the model "this token is a shoe", the same way BERT adds a
   segment embedding. It's learned, so the model decides what being a shoe should mean.

3. **No position embeddings.** A sentence has an order; an outfit doesn't ("top, jeans, shoes" is
   the same outfit as "shoes, top, jeans"). So we deliberately leave out positional encodings. Without them,
   self-attention is *permutation-equivariant*: shuffling the items just shuffles the outputs, and
   the final score is identical. This is tested in `tests/test_compat_train.py`.

4. **A [CLS] token.** We prepend one extra learned token that doesn't correspond to any item. Its
   job is to collect a summary of the whole outfit, the same trick BERT uses for sentence
   classification.

5. **Self-attention layers (2 layers, 4 heads).** In each layer, every token looks at every other
   token and decides how much to borrow from each:
   - each token makes a *query* ("what am I looking for?"), a *key* ("what do I offer?") and a
     *value* ("what information do I pass on?");
   - attention weight from token i to token j = softmax over j of (query_i · key_j / √d);
   - token i's new representation = weighted sum of the values, plus a small MLP, with residual
     connections and layer normalisation around both (pre-norm, which trains more stably).

   *4 heads* means this happens 4 times in parallel with different learned projections, so one head
   can focus on color, another on formality, and so on. In practice the heads specialise in ways
   you discover, not ways you assign. *2 layers* means items can react to items that have already
   reacted to others ("the bag notices that the top and shoes are both red").

6. **Padding mask.** Outfits have 2 to 16 items, and we batch them by padding short outfits with
   dummy tokens. The padding mask tells attention to give those tokens zero weight, so a 4-item
   outfit gets exactly the same score whether or not it's batched with a 12-item one (also tested).

7. **Score head.** We take the [CLS] token's final representation, normalise it, and pass it
   through a 2-layer MLP down to **one number** (a logit). Higher means "more like a real outfit".

Size: 349k parameters, about twice Model A (164k). It trains on CPU in seconds per epoch.

## How it's trained

- **Positives:** all 16,995 real training outfits.
- **Negatives (fakes):** for every real outfit, each epoch makes one fake:
  - half the time, *every* item is swapped for a random item of the same type. This is exactly how
    the test set's fake outfits were built, so it teaches the compatibility task;
  - half the time, *one* item is swapped. These are harder: the outfit is 80% real and one piece is
    wrong. This is exactly the fill-in-the-blank situation.

  Fakes are redrawn every epoch, so the model never sees the same fake twice.
- **Loss: binary cross-entropy.** Real outfits should get a high logit and fakes a low one. This is
  a different loss from Model A's triplet loss. Model A learns *distances* between items; this model
  learns a *classifier* over whole outfits.
- **Optimiser:** AdamW (Adam with decoupled weight decay), learning rate 3e-4, batches of 128
  outfits, gradient clipping at norm 1 for stability.
- **Early stopping** on validation compatibility AUC (patience 8), with 3 seeds, the same protocol as
  every other model here.

## How it's evaluated

- **Compatibility AUC:** score each test outfit directly; AUC of real vs fake.
- **FITB:** for each question, score the context plus each of the 4 candidates as a full outfit,
  then pick the highest. Unlike Model A, this lets the score depend on how the candidate interacts
  with the *combination* of context items.

## Results (Polyvore Outfits disjoint, 3 seeds, selection on validation)

| model | inputs | val AUC | val FITB | test AUC | test FITB |
|---|---|---|---|---|---|
| Model A (pairwise) | image | 0.936 | 68.7% | 0.915 ± 0.001 | 67.0% ± 0.4 |
| Transformer | image | 0.913 | 60.5% | 0.892 ± 0.003 | 58.7% ± 0.4 |
| Combined pairwise model | image + title text | **0.945** | **70.4%** | **0.928 ± 0.001** | **69.0% ± 0.4** |
| Transformer | image + title text | 0.922 | 60.4% | 0.904 ± 0.002 | 58.9% ± 0.4 |

**The pairwise models win, especially on FITB (−8 to −10 points for the transformer).** The
transformer is still respectable on AUC: on the same frozen features it beats every published
disjoint result except on FITB. Notes:

- **FITB is where it loses most.** Model A's training examples *are* FITB comparisons: "is this
  candidate closer to the anchor than that same-type alternative?", about 386k per epoch. The
  transformer gets one real/fake label per outfit (about 34k per epoch), and only half of its fakes
  differ by a single item. It sees far fewer of the fine-grained comparisons FITB tests.
- **The text runs hit the 60-epoch cap** (best epochs 46–59, validation still creeping up by about
  0.003 per 10 epochs), so they're slightly undertrained. The image-only runs early-stopped on their
  own (best epochs 28–35). More epochs would not plausibly close a 0.023 validation-AUC gap at that
  rate.
- The caveats below turned out to describe what happened.

## Why it might *not* beat pairwise scoring (honest caveats)

- **Data size.** 17k training outfits is small for attention. Pairwise models get roughly 20
  training pairs per outfit, so about 390k triplets. The transformer gets one example per outfit
  plus one fake. More capacity with fewer examples is a recipe for overfitting.
- **Frozen inputs.** Published transformers (OutfitTransformer) fine-tune the image encoder end to
  end. Ours only sees fixed FashionCLIP vectors, so it can only recombine features that already exist.
- **The test negatives are easy.** Random same-type swaps usually produce obviously incoherent
  outfits, which pairwise averaging already catches. The transformer's advantage (context, the odd
  one out) matters most for *subtle* fakes, which this benchmark doesn't contain many of.

## A bug worth telling: the first version sat at chance

The first full training run had loss stuck at 0.693 (= ln 2, i.e. predicting 50/50 for everything)
and validation AUC 0.50 for two epochs. How it was debugged:

1. **Can it overfit a tiny set?** 256 real + 256 fake outfits, no dropout: yes, 100% training
   accuracy. So the plumbing (masking, labels, gradients) was fine. The problem was optimisation,
   not a bug in the data path.
2. **Look at the scales.** At initialisation the item tokens had norm about **0.45**, because unit-norm
   CLIP vectors through a default-initialised linear layer come out small. The type embeddings had
   norm about **11** (`nn.Embedding` initialises from N(0, 1), so 128 dims give √128 ≈ 11). The
   type embedding was 25× louder than the item signal, and it carries *no* information here:
   fake outfits are built to have exactly the same types as real ones.
3. **Fix:** a LayerNorm after the item projection, so item tokens also have norm about 11. The same
   overfitting test then converges twice as fast, and the full run learns from the first epoch
   (validation AUC 0.80 → 0.85 in three epochs instead of 0.50 → 0.50).

The general lesson: when you add embeddings of different origins (here item + type; in BERT, token +
position + segment), check that they're on comparable scales, or one will drown out the other.

## Interview questions you should be able to answer

1. *Why no positional encoding, and what property does that give the model?* An outfit is a set.
   Without positions, self-attention is permutation-equivariant, so the [CLS] score is invariant to
   item order.
2. *What does attention let this model do that a mean of pairwise scores can't?* Context-dependent
   judgments ("this bag works with *this combination*") and spotting a single odd item that an
   average would wash out.
3. *How are the negatives built, and why two kinds?* Full same-type swaps match the test's fake
   outfits; single swaps are harder and match FITB. Both keep the type composition, so the model
   can't win by counting categories.
4. *Why BCE here but triplet loss for Model A?* Model A learns a metric space between items (a
   relative, ranking objective). The transformer outputs one score per outfit, so it's a binary
   classification of real vs fake.
5. *Your transformer trained at chance. How did you debug it?* First check whether it can overfit a
   tiny set: it could, so the code path was fine. Then compare activation scales: item tokens
   (norm 0.45) were drowned out by type embeddings (norm 11) that carry no real-vs-fake signal. A
   LayerNorm after the item projection fixed it.
6. *Why might a bigger model do worse here?* About 17k outfits is little data for attention; frozen
   inputs cap what it can learn; and easy negatives don't reward context-awareness.
