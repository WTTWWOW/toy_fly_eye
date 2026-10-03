import numpy as np
import torch

from toyeye.utils import set_seed


def test_set_seed_repeats():
    set_seed(1)
    a = (np.random.rand(3), torch.rand(3))
    set_seed(1)
    b = (np.random.rand(3), torch.rand(3))
    assert np.array_equal(a[0], b[0])
    assert torch.equal(a[1], b[1])
