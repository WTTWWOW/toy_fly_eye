"""Hand-built Hassenstein–Reichardt detector.

Multiply each receptor's delayed (low-pass filtered) signal by its neighbour's
current signal, then subtract the mirror-image product. The sign tracks direction.
"""

import numpy as np


def lowpass(movie: np.ndarray, tau: float) -> np.ndarray:
    """First-order low-pass filter along time (axis 0)."""
    raise NotImplementedError


def reichardt(movie: np.ndarray, tau: float = 2.0) -> np.ndarray:
    """Per-column Reichardt output, shape (steps, n). Positive for rightward motion."""
    raise NotImplementedError
