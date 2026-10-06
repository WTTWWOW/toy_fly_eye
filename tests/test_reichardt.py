import numpy as np
import pytest

from toyeye import N_RECEPTORS
from toyeye.reichardt import lowpass, reichardt
from toyeye.world import make_drum, sine_grating, smooth_texture


def test_lowpass_keeps_a_constant_signal():
    movie = np.full((20, N_RECEPTORS), 0.3)
    assert np.allclose(lowpass(movie, tau=2.0), 0.3)


def test_lowpass_step_response_is_exponential():
    tau = 3.0
    step = np.r_[0.0, np.ones(30)][:, None]  # off at t = 0, on from t = 1
    t = np.arange(31)
    assert np.allclose(lowpass(step, tau)[:, 0], 1 - np.exp(-t / tau))


def test_lowpass_rejects_nonpositive_tau():
    with pytest.raises(ValueError):
        lowpass(np.zeros((5, 3)), tau=0)


def test_output_shape():
    movie = make_drum(sine_grating(8), 0.5, steps=40)
    assert reichardt(movie).shape == (40, N_RECEPTORS)


def test_static_pattern_gives_zero():
    movie = make_drum(smooth_texture(np.random.default_rng(0)), 0.0, steps=30)
    assert np.allclose(reichardt(movie), 0.0)


PATTERNS = {
    "grating": sine_grating(8),
    "texture": smooth_texture(np.random.default_rng(1), smoothness=2.0),
}


@pytest.mark.parametrize("name", PATTERNS)
@pytest.mark.parametrize("velocity", [-2, -1, -0.5, -0.25, 0.25, 0.5, 1, 2])
def test_sign_tracks_direction(name, velocity):
    movie = make_drum(PATTERNS[name], velocity, steps=200)
    mean_response = reichardt(movie)[20:].mean()  # skip the first few tau
    assert np.sign(mean_response) == np.sign(velocity)


@pytest.mark.parametrize("velocity", [-1.5, 0.3, 1.0])
def test_mirror_image_flips_the_sign(velocity):
    """Flipping the ring left-right turns rightward motion into leftward."""
    movie = make_drum(smooth_texture(np.random.default_rng(2)), velocity, steps=100)
    mirrored = movie[:, ::-1]
    assert np.isclose(reichardt(mirrored).mean(), -reichardt(movie).mean())


def test_grating_drives_every_column_equally():
    """Averaged over whole cycles, a drifting grating drives every column the same.

    Within a cycle the output ripples, because the mean brightness (0.5) leaks
    through the products; the opponent subtraction cancels it only on average.
    """
    wavelength, velocity = 8, 0.5
    period = round(wavelength / abs(velocity))  # time steps per temporal cycle
    movie = make_drum(sine_grating(wavelength), velocity, steps=50 + 10 * period)
    per_column = reichardt(movie)[50:].mean(axis=0)
    assert np.allclose(per_column, per_column.mean())
    assert per_column.mean() > 0
