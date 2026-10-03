"""Write these before any training: they catch a broken warp cheaply.

Marked xfail until make_drum and shift_ring are implemented; remove the
marks once they are.
"""

import numpy as np
import pytest

from toyeye.warp import photometric_error
from toyeye.world import make_drum

todo = pytest.mark.xfail(raises=NotImplementedError, reason="Stage 1 not implemented yet")


@todo
@pytest.mark.parametrize("velocity", [-1.5, -0.3, 0.7, 2.0])
def test_true_velocity_rebuilds_previous_frame(velocity):
    pattern = np.sin(2 * np.pi * np.arange(32) / 8) * 0.5 + 0.5
    movie = make_drum(pattern, velocity, steps=10)
    errs = [photometric_error(movie[t - 1], movie[t], velocity) for t in range(1, 10)]
    assert max(errs) < 1e-3


@todo
@pytest.mark.parametrize("velocity", [-1.5, 0.7, 2.0])
def test_zero_velocity_is_clearly_worse(velocity):
    pattern = np.sin(2 * np.pi * np.arange(32) / 8) * 0.5 + 0.5
    movie = make_drum(pattern, velocity, steps=10)
    true_err = photometric_error(movie[0], movie[1], velocity)
    zero_err = photometric_error(movie[0], movie[1], 0.0)
    assert zero_err > 10 * true_err + 1e-4
