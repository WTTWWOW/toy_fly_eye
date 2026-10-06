"""Hand-built Hassenstein–Reichardt detector.

Multiply each receptor's delayed (low-pass filtered) signal by its neighbour's
current signal, then subtract the mirror-image product. The sign tracks direction.

Column x sits between receptor x and its right neighbour x + 1 (wrapping
around the ring, so column n - 1 pairs receptor n - 1 with receptor 0):

    out[t, x] = D[t, x] * S[t, x + 1]  -  D[t, x + 1] * S[t, x]
                (rightward half)          (mirror image, leftward half)

where S is the current signal and D the delayed one. When the pattern moves
right, a feature reaches receptor x first and x + 1 a little later, so the
delayed copy at x lines up with the current signal at x + 1 and the first
product wins: the output is positive. Leftward motion favours the mirror
product and the output is negative. Velocities follow world.py: receptors
per time step, positive means rightward.
"""

import numpy as np


def lowpass(movie: np.ndarray, tau: float) -> np.ndarray:
    """First-order low-pass filter along time (axis 0).

    y[t] = y[t-1] + alpha * (x[t] - y[t-1]), with alpha = 1 - exp(-1 / tau),
    so a step input rises as 1 - exp(-t / tau). `tau` is the time constant in
    time steps and must be positive. The filter starts at y[0] = x[0], as if
    the first frame had always been there, which avoids a start-up transient.
    """
    if tau <= 0:
        raise ValueError(f"tau must be positive, got {tau}")
    movie = np.asarray(movie, dtype=float)
    alpha = 1 - np.exp(-1 / tau)
    out = np.empty_like(movie)
    out[0] = movie[0]
    for t in range(1, len(movie)):
        out[t] = out[t - 1] + alpha * (movie[t] - out[t - 1])
    return out


def reichardt(movie: np.ndarray, tau: float = 2.0) -> np.ndarray:
    """Per-column Reichardt output, shape (steps, n). Positive for rightward motion.

    `movie` is a (steps, n) brightness array from world.make_drum; `tau` is the
    time constant of the delay filter, in time steps.
    """
    current = np.asarray(movie, dtype=float)
    delayed = lowpass(current, tau)
    # np.roll(a, -1, axis=1)[:, x] is a[:, x + 1]: each receptor's right neighbour.
    current_right = np.roll(current, -1, axis=1)
    delayed_right = np.roll(delayed, -1, axis=1)
    return delayed * current_right - delayed_right * current
