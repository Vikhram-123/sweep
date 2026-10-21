"""Unit tests for the research training code (synthetic tensors only; no dataset needed)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from compat.data import Split  # noqa: E402
from compat.losses import squared_triplet_loss, triplet_loss  # noqa: E402
from compat.train import Bank, TripletSampler, mine_batch_negatives, same_type_knn  # noqa: E402


def toy_bank(n=40, d=8, n_types=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    emb = torch.nn.functional.normalize(torch.randn(n, d, generator=g), dim=-1)
    return Bank(index={f"i{k}": k for k in range(n)}, emb=emb, types=torch.arange(n) % n_types,
                type_names=[f"t{t}" for t in range(n_types)])


def test_squared_loss_matches_definition():
    a, p, n = torch.randn(5, 4), torch.randn(5, 4), torch.randn(5, 4)
    ref = torch.relu(((a - p) ** 2).sum(1) - ((a - n) ** 2).sum(1) + 0.3).mean()
    assert torch.allclose(squared_triplet_loss(a, p, n, 0.3), ref)
    assert torch.allclose(triplet_loss(a, p, n, 0.3), torch.nn.TripletMarginLoss(0.3, eps=0)(a, p, n), atol=1e-6)


def test_knn_is_same_type_excludes_self_and_near_duplicates():
    bank = toy_bank()
    bank.emb[1] = bank.emb[3]  # 1 and 3 are near-duplicates of the same type
    rows = np.arange(len(bank.emb))
    knn = same_type_knn(bank, rows, k=5, max_cos=0.95)
    types = bank.types.numpy()
    assert (knn >= 0).all() and (types[knn] == types[:, None]).all()
    assert not (knn == rows[:, None]).any()
    assert 3 not in knn[1] and 1 not in knn[3]
    # the first neighbour really is the most similar remaining same-type item
    sim = (bank.emb @ bank.emb.T).numpy()
    cand = [j for j in rows if types[j] == types[0] and j != 0 and sim[0, j] <= 0.95]
    assert knn[0, 0] == max(cand, key=lambda j: sim[0, j])


def test_sampler_knn_negatives_have_positive_type():
    bank = toy_bank()
    split = Split("train", {f"o{k}": [f"i{2 * k}", f"i{2 * k + 1}"] for k in range(20)})
    s = TripletSampler(split, bank, negatives="knn", knn_k=5)
    trip = s.epoch(np.random.default_rng(0))
    assert trip.shape == (len(s), 4)
    t = bank.types.numpy()
    assert (t[trip[:, 2]] == t[trip[:, 1]]).all() and (trip[:, 2] != trip[:, 1]).all()


@pytest.mark.parametrize("mode", ["batch_hard", "batch_semihard"])
def test_batch_mining_respects_type_and_outfit(mode):
    torch.manual_seed(0)
    B, d = 16, 4
    za, zp, zn = torch.randn(B, d), torch.randn(B, d), torch.randn(B, d)
    ptype = torch.arange(B) % 2
    outfit = torch.arange(B) // 2  # rows 2k, 2k+1 share an outfit
    rows = torch.stack([torch.arange(B), torch.arange(B) + 100, torch.arange(B) + 200, outfit], 1)
    mined = mine_batch_negatives(za, zp, zn, ptype, outfit, rows, "l2", mode)
    cand = torch.cat([zp, zn])
    cand_t, cand_o = torch.cat([ptype, ptype]), torch.cat([outfit, torch.full_like(outfit, -1)])
    for i in range(B):
        j = int(((cand - mined[i]).abs().sum(1) < 1e-6).nonzero()[0])
        assert cand_t[j] == ptype[i] and cand_o[j] != outfit[i] and j != i  # not its own positive
        d = (za[i] - cand).norm(dim=1)
        ok = (cand_t == ptype[i]) & (cand_o != outfit[i])
        ok[i] = False
        if mode == "batch_semihard":
            ok &= d > (za[i] - zp[i]).norm()
        if ok.any():
            assert d[j] == pytest.approx(float(d[ok].min()), abs=1e-5)


def test_transformer_is_order_invariant_and_ignores_padding():
    from compat.transformer import FakeOutfits, OutfitTransformer, pack

    torch.manual_seed(0)
    bank = toy_bank(n=30, d=8, n_types=3)
    model = OutfitTransformer(in_dim=8, n_types=3, d_model=16, heads=4, layers=2, ff=32, dropout=0.0).eval()
    outfit = np.array([0, 4, 8, 13])
    with torch.no_grad():
        s = model(*pack([outfit], bank, "cpu"))
        s_perm = model(*pack([outfit[[2, 0, 3, 1]]], bank, "cpu"))
        s_padded = model(*pack([outfit, np.arange(10)], bank, "cpu"))[0]  # batched with a longer outfit
    assert torch.allclose(s, s_perm, atol=1e-5)
    assert torch.allclose(s[0], s_padded, atol=1e-5)

    fakes = FakeOutfits([outfit, np.array([1, 2, 5])], bank)
    rng = np.random.default_rng(0)
    full, single = fakes.swap(outfit, rng, single=False), fakes.swap(outfit, rng, single=True)
    t = bank.types.numpy()
    assert (t[full] == t[outfit]).all() and (t[single] == t[outfit]).all()
    assert (single != outfit).sum() <= 1
