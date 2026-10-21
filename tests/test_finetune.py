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
