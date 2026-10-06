"""Space-time plots, the main debugging tool for the drum world."""

import matplotlib.pyplot as plt
import numpy as np


def plot_spacetime(movie: np.ndarray, ax=None, title: str | None = None):
    """Show a (steps, n) movie with receptors across and time running down.

    Rightward motion (positive velocity) makes stripes tilt down to the right;
    leftward motion tilts them down to the left.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(3, 4))
    ax.imshow(movie, cmap="gray", vmin=0, vmax=1, aspect="auto",
              interpolation="nearest")
    ax.set_xlabel("receptor")
    ax.set_ylabel("time step")
    if title:
        ax.set_title(title)
    return ax
