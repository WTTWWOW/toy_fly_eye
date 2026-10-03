import numpy as np

from toyeye.probes import collapse_variance, direction_selectivity_index


def test_dsi_bounds():
    dsi = direction_selectivity_index(np.array([1.0, 1.0, 0.0]), np.array([0.0, 1.0, 1.0]))
    assert np.allclose(dsi, [1.0, 0.0, -1.0])


def test_collapse_variance_zero_for_constant():
    assert np.allclose(collapse_variance(np.ones((5, 7, 4))), 0.0)
