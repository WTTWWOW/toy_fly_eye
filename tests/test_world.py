import numpy as np
import pytest

from toyeye import N_RECEPTORS
from toyeye.world import make_drum, sine_grating, smooth_texture


def test_sine_grating_mean_and_range():
    g = sine_grating(wavelength=8, contrast=0.6)
    assert g.shape == (N_RECEPTORS,)
    assert np.isclose(g.mean(), 0.5)
    assert np.isclose(g.max() - g.min(), 0.6)


def test_smooth_texture_range_and_seed():
    a = smooth_texture(np.random.default_rng(0), smoothness=3.0, contrast=0.8)
    b = smooth_texture(np.random.default_rng(0), smoothness=3.0, contrast=0.8)
    assert np.array_equal(a, b)
    assert np.isclose(a.min(), 0.1) and np.isclose(a.max(), 0.9)


def test_smoother_texture_has_smaller_steps():
    rough = smooth_texture(np.random.default_rng(1), smoothness=0.5)
    smooth = smooth_texture(np.random.default_rng(1), smoothness=4.0)
    step = lambda p: np.abs(np.diff(np.r_[p, p[0]])).mean()  # includes the seam
    assert step(smooth) < step(rough)


def test_shape_and_first_frame():
    p = sine_grating(8)
    movie = make_drum(p, 0.7, steps=20)
    assert movie.shape == (20, N_RECEPTORS)
    assert np.allclose(movie[0], p)


def test_zero_velocity_is_static():
    p = smooth_texture(np.random.default_rng(2))
    assert np.allclose(make_drum(p, 0.0, 5), p)


@pytest.mark.parametrize("velocity", [-2, -1, 1, 3])
def test_integer_velocity_is_a_roll_with_wraparound(velocity):
    p = smooth_texture(np.random.default_rng(3))
    movie = make_drum(p, velocity, 40)  # long enough to wrap around the ring
    for t in range(40):
        assert np.allclose(movie[t], np.roll(p, velocity * t))


def test_half_step_is_midpoint():
    p = smooth_texture(np.random.default_rng(4))
    movie = make_drum(p, 0.5, 2)
    assert np.allclose(movie[1], 0.5 * (p + np.roll(p, 1)))


def test_full_revolution_returns_to_start():
    p = smooth_texture(np.random.default_rng(5))
    movie = make_drum(p, 0.25, 4 * N_RECEPTORS + 1)
    assert np.allclose(movie[-1], p)


@pytest.mark.parametrize("velocity", [-1.6, -0.4, 0.4, 1.6])
def test_stripes_tilt_with_direction(velocity):
    """Consecutive frames align better shifted the way the drum turns."""
    movie = make_drum(sine_grating(16), velocity, 10)
    err = lambda s: np.mean((np.roll(movie[:-1], s, axis=1) - movie[1:]) ** 2)
    assert np.sign(err(-1) - err(1)) == np.sign(velocity)
