import numpy as np
import pytest

from sweep.device import align_weights

torch = pytest.importorskip("torch")


def _misaligned_linear():
    lin = torch.nn.Linear(8, 4, bias=False)
    # Same layout a zero-copy safetensors load produces: weights at an odd offset in a byte buffer.
    raw = bytearray(2 + lin.weight.numel() * 4)
    w = torch.frombuffer(raw, dtype=torch.float32, offset=2, count=lin.weight.numel()).view(4, 8)
    w.copy_(torch.arange(32, dtype=torch.float32).view(4, 8))
    lin.weight.data = w
    return lin


def test_align_weights_copies_misaligned_cpu_tensors():
    lin = _misaligned_linear()
    assert lin.weight.data_ptr() % 4 != 0
    before = lin.weight.detach().numpy().copy()
    align_weights(lin, "cpu")
    assert lin.weight.data_ptr() % 64 == 0
    np.testing.assert_array_equal(lin.weight.detach().numpy(), before)
    with torch.inference_mode():  # batch-1 matrix-vector path that used to SIGBUS
        assert torch.isfinite(lin(torch.ones(1, 8))).all()


def test_align_weights_leaves_non_cpu_alone():
    lin = _misaligned_linear()
    ptr = lin.weight.data_ptr()
    align_weights(lin, "mps")
    assert lin.weight.data_ptr() == ptr
