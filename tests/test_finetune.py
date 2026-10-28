import math

import pytest

torch = pytest.importorskip("torch")

from compat.finetune import Encoder, lr_factor  # noqa: E402


def test_lr_schedule_frozen_then_warmup_then_cosine():
    total, start, warm = 1000, 100, 50
    assert lr_factor(0, total, warm, start) == 0.0 and lr_factor(99, total, warm, start) == 0.0   # frozen
    assert lr_factor(100, total, warm, start) == pytest.approx(1 / 50)                          # warmup begins
    assert lr_factor(149, total, warm, start) == pytest.approx(1.0)                             # warmup done
    mid = start + warm + (total - start - warm) // 2
    assert lr_factor(mid, total, warm, start) == pytest.approx(0.5, abs=0.01)                   # cosine midpoint
    assert lr_factor(total, total, warm, start) == pytest.approx(0.0, abs=1e-6)                 # decays to 0
    xs = [lr_factor(s, total, warm, start) for s in range(start + warm, total)]
    assert all(a >= b - 1e-12 for a, b in zip(xs, xs[1:]))                                      # monotone decay


def _toy_encoder():
    blocks = [torch.nn.Linear(4, 4) for _ in range(5)]
    tail = [torch.nn.LayerNorm(4)]
    body = torch.nn.Sequential(*blocks, *tail)
    return Encoder(body, body, blocks, tail, 4, 32, (0.5,) * 3, (0.5,) * 3, lambda: None), blocks, tail


def test_partial_unfreeze_trains_only_last_blocks_and_tail():
    enc, blocks, tail = _toy_encoder()
    params = enc.set_trainable(2)
    assert all(not p.requires_grad for b in blocks[:3] for p in b.parameters())
    assert all(p.requires_grad for b in blocks[3:] + tail for p in b.parameters())
    assert len(params) == sum(len(list(m.parameters())) for m in blocks[3:] + tail)
    enc.set_trainable("all")
    assert all(p.requires_grad for p in enc.body.parameters())
    assert enc.set_trainable(0) == [] and not any(p.requires_grad for p in enc.body.parameters())


DEVICES = ["cpu",
           pytest.param("mps", marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS")),
           pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA"))]


@pytest.mark.parametrize("load_on", DEVICES)
def test_rng_state_round_trip_saved_on_cpu_loaded_on_any_device(tmp_path, load_on):
    """Resume loads checkpoints with map_location=<training device>, which moves the saved RNG tensors there."""
    import random

    import numpy as np

    from compat.finetune import rng_state, save_atomic, set_rng_state

    random.seed(1); np.random.seed(1); torch.manual_seed(1)
    state = rng_state()
    # A checkpoint from a CUDA machine also carries per-GPU states; here they must load on any machine.
    state["cuda"] = state["cuda"] or [torch.get_rng_state(), torch.get_rng_state()]
    save_atomic({"rng": state}, tmp_path / "last.pt")
    expected = (random.random(), float(np.random.rand()), torch.rand(3))

    random.seed(2); np.random.seed(2); torch.manual_seed(2)          # move every generator elsewhere
    ck = torch.load(tmp_path / "last.pt", map_location=load_on, weights_only=False)
    assert ck["rng"]["torch"].device.type == load_on                 # the condition that used to crash
    set_rng_state(ck["rng"])
    got = (random.random(), float(np.random.rand()), torch.rand(3))
    assert got[0] == expected[0] and got[1] == expected[1] and torch.equal(got[2], expected[2])
