"""Rebuild one frame from another given a velocity."""

import numpy as np


def shift_ring(frame: np.ndarray, shift: float) -> np.ndarray:
    """Sample `frame` at x + shift with linear interpolation and wrap-around."""
    raise NotImplementedError


def reconstruct_previous(current: np.ndarray, velocity: float) -> np.ndarray:
    """Rebuild the previous frame by sampling the current frame at x + v."""
    return shift_ring(current, velocity)


def photometric_error(previous: np.ndarray, current: np.ndarray, velocity: float) -> float:
    """Mean squared error between the previous frame and its reconstruction."""
    return float(np.mean((reconstruct_previous(current, velocity) - previous) ** 2))
