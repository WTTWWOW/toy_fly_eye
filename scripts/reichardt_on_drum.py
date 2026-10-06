"""Stage 1, "build the classic detector": run the Reichardt detector on the drum.

Top row: space-time plots of a texture turning left and right, and the detector's
output for each (red = positive = rightward, blue = leftward).
Bottom: mean response against drum velocity for a grating and a texture. The
sign should follow the velocity, until the grating moves more than half a
wavelength per step and aliases (it then looks like it is turning the other way).
Saves the figure to results/<timestamp>_reichardt_on_drum/.

    uv run python scripts/reichardt_on_drum.py --seed 0
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from toyeye.plots import plot_spacetime
from toyeye.reichardt import reichardt
from toyeye.utils import make_run_dir, set_seed
from toyeye.world import make_drum, sine_grating, smooth_texture


def mean_response(pattern, velocity, steps, tau, skip):
    """Detector output averaged over columns and time, after `skip` settling steps."""
    return reichardt(make_drum(pattern, velocity, steps), tau)[skip:].mean()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tau", type=float, default=2.0, help="delay filter, time steps")
    parser.add_argument("--speed", type=float, default=0.5, help="receptors per step")
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--wavelength", type=float, default=8.0)
    parser.add_argument("--smoothness", type=float, default=2.0)
    args = parser.parse_args()

    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    grating = sine_grating(args.wavelength)
    texture = smooth_texture(rng, args.smoothness)

    fig = plt.figure(figsize=(10, 8))
    grid = fig.add_gridspec(2, 4, height_ratios=[1.3, 1])

    # Top row: stimulus and detector output, turning left then right.
    outputs = {v: reichardt(make_drum(texture, v, args.steps), args.tau)
               for v in (-args.speed, args.speed)}
    limit = max(np.abs(out).max() for out in outputs.values())
    for i, (v, out) in enumerate(outputs.items()):
        plot_spacetime(make_drum(texture, v, args.steps), fig.add_subplot(grid[0, 2 * i]),
                       f"texture, v = {v:+g}")
        ax = fig.add_subplot(grid[0, 2 * i + 1])
        ax.imshow(out, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto",
                  interpolation="nearest")
        ax.set_xlabel("column")
        ax.set_title(f"Reichardt, mean {out.mean():+.4f}")

    # Bottom: mean response against velocity, past the grating's aliasing limit.
    velocities = np.linspace(-1.5 * args.wavelength, 1.5 * args.wavelength, 97)
    ax = fig.add_subplot(grid[1, :])
    for name, pattern in [(f"grating (λ = {args.wavelength:g})", grating),
                          (f"texture (σ = {args.smoothness:g})", texture)]:
        curve = [mean_response(pattern, v, 400, args.tau, skip=50) for v in velocities]
        ax.plot(velocities, curve, label=name)
    for edge in (-args.wavelength / 2, args.wavelength / 2):
        ax.axvline(edge, color="gray", ls=":", lw=1)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("drum velocity (receptors per step)")
    ax.set_ylabel("mean response")
    ax.set_title(f"τ = {args.tau:g}; dotted lines: ± half a grating wavelength per step")
    ax.legend()
    fig.tight_layout()

    run_dir = make_run_dir("reichardt_on_drum", config=vars(args))
    fig.savefig(run_dir / "reichardt.png", dpi=150)
    print(run_dir / "reichardt.png")


if __name__ == "__main__":
    main()
