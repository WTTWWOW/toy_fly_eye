"""Electrophysiology-style probes. Stage 2 and the flagship reuse these unchanged."""

import numpy as np


def velocity_error(decoded: np.ndarray, true: np.ndarray) -> float:
    """Mean absolute error between decoded and true velocity."""
    return float(np.mean(np.abs(decoded - true)))


def linear_probe(hidden: np.ndarray, target: np.ndarray):
    """Fit a linear readout on frozen hidden states (for the temporal-prediction eye)."""
    raise NotImplementedError


def direction_selectivity_index(preferred: np.ndarray, null: np.ndarray) -> np.ndarray:
    """DSI = (preferred - null) / (preferred + null), per unit."""
    preferred, null = np.asarray(preferred, float), np.asarray(null, float)
    return (preferred - null) / (preferred + null + 1e-12)


def temporal_frequency_tuning(responses: np.ndarray, speeds: np.ndarray, wavelengths: np.ndarray):
    """Response against speed for several wavelengths, replotted against TF = speed / wavelength."""
    raise NotImplementedError


def collapse_variance(hidden: np.ndarray) -> np.ndarray:
    """Variance of each hidden unit across time and patterns; near zero means collapse."""
    return hidden.reshape(-1, hidden.shape[-1]).var(axis=0)
