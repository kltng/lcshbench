import numpy as np
from lcsh_benchmark.retrieval.backends import FakeBackend


def test_fake_backend_is_deterministic_and_unit_normed():
    be = FakeBackend(dim=16)
    v1 = be.encode(["hello", "world"])
    v2 = be.encode(["hello", "world"])
    assert v1.shape == (2, 16)
    assert np.allclose(v1, v2)                       # deterministic
    assert np.allclose(np.linalg.norm(v1, axis=1), 1.0)  # unit-normed
    assert be.name == "fake-16"
    assert be.cost_usd(["hello", "world"]) == 0.0    # local/free
