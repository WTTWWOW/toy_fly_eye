"""The drum world: 1D patterns rotating around a ring of receptors.

The true velocity is known exactly, so this is the project's mini-Sintel.
Convention: velocity is in receptors per time step; positive means rightward,
i.e. towards higher receptor indices. Frame t is the pattern shifted right by
velocity * t: movie[t, x] = pattern(x - velocity * t).
"""

import numpy as np

from . import N_RECEPTORS


def sine_grating(wavelength: float, phase: float = 0.0, contrast: float = 1.0,
                 n: int = N_RECEPTORS) -> np.ndarray:
    """A sine grating sampled at n receptors, mean 0.5.

    `wavelength` is in receptors (pick a divisor of n so the grating is
    seamless on the ring), `phase` in radians, and `contrast` is the
    Michelson contrast, so values span 0.5 ± contrast / 2.
    """
    x = np.arange(n)
    return 0.5 + 0.5 * contrast * np.sin(2 * np.pi * x / wavelength + phase)


def smooth_texture(rng: np.random.Generator, smoothness: float = 3.0,
                   n: int = N_RECEPTORS, contrast: float = 1.0) -> np.ndarray:
    """A smooth random texture, periodic on the ring.

    White noise blurred by a Gaussian of standard deviation `smoothness`
    receptors (circular convolution, so there is no seam), then rescaled to
    span exactly 0.5 ± contrast / 2.
    """
    noise = rng.standard_normal(n)
    freqs = np.fft.rfftfreq(n)  # cycles per receptor
    kernel = np.exp(-0.5 * (2 * np.pi * freqs * smoothness) ** 2)
    smooth = np.fft.irfft(np.fft.rfft(noise) * kernel, n)
    smooth -= smooth.min()
    smooth /= smooth.max()
    return 0.5 + contrast * (smooth - 0.5)


def make_drum(pattern: np.ndarray, velocity: float, steps: int) -> np.ndarray:
    """Rotate `pattern` at `velocity` for `steps` frames.

    Returns a (steps, len(pattern)) brightness array. Sub-pixel shifts use
    linear interpolation, with wrap-around on the ring. Each frame is sampled
    from the original pattern, so interpolation blur does not accumulate.
    """
    pattern = np.asarray(pattern, dtype=float)
    n = len(pattern)
    x = np.arange(n)
    t = np.arange(steps)[:, None]
    src = x[None, :] - velocity * t  # where each receptor looks on the pattern
    left = np.floor(src)
    frac = src - left
    left = left.astype(int) % n
    right = (left + 1) % n
    return (1 - frac) * pattern[left] + frac * pattern[right]


def repeat_frames(movie: np.ndarray, times: int = 2) -> np.ndarray:
    """Show every frame `times` times, as flyvis does with Sintel (the bonus trap)."""
    return np.repeat(movie, times, axis=0)
