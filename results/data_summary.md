# Polyvore Outfits — disjoint split

Source: huggingface.co/datasets/mvasil/polyvore-outfits (authors' repo), hashes verified. Data lives in `data/polyvore/` (gitignored).

| split | outfits | unique items | items w/o image | mean size | min | max | compat outfits | compat positives | FITB questions |
|---|---|---|---|---|---|---|---|---|---|
| train | 16995 | 71967 | 0 | 5.06 | 2 | 16 | 33990 | 16995 | 16995 |
| valid | 3000 | 14657 | 0 | 5.10 | 2 | 14 | 6000 | 3000 | 3000 |
| test | 15145 | 70035 | 0 | 4.90 | 2 | 14 | 30290 | 15145 | 15145 |

**Item overlap between splits** (should be 0 for the disjoint split): {'train/valid': 3781, 'train/test': 84, 'valid/test': 34}

**Categories:** 11 coarse semantic types (used as the paper's "types"), 152 fine-grained category ids.

| type | train items |
|---|---|
| shoes | 13362 |
| jewellery | 12067 |
| bags | 11832 |
| tops | 9136 |
| bottoms | 7886 |
| all-body | 5751 |
| outerwear | 4153 |
| sunglasses | 3294 |
| accessories | 1692 |
| hats | 1676 |
| scarves | 1118 |

**Distinct type pairs co-occurring in train outfits:** 55. Most common: bags–shoes (13969), jewellery–shoes (13424), bags–jewellery (12418), shoes–tops (9981), bottoms–tops (9250), bottoms–shoes (9247), bags–tops (8776), bags–bottoms (8108)

**Test tasks**

- *Compatibility*: each line is a label (1 = real outfit, 0 = items drawn at random from the split) and the outfit's items. Metric: ROC AUC of outfit scores.
- *Fill-in-the-blank*: a partial outfit plus 4 candidates; exactly one came from the outfit. Metric: accuracy (chance = 25%). Correct-answer position in test: {0: 3745, 1: 3758, 2: 3855, 3: 3787}.

**What a scorer can and can't exploit**

- FITB candidates share the answer's type in 15144/15145 questions, so category alone can't pick the answer.
- Negative outfits have the same multiset of types as positives (identical distributions): each item is replaced by a random item of the same type. Category-pairing rules carry no signal; only visual compatibility does.
- Validation shares 3781 items with train, so validation scores are optimistic; they are used for early stopping only, never reported as results.
