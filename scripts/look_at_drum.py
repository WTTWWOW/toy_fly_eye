"""Stage 1, "look at it": space-time plots of the drum world.

Draws a sine grating and a smooth random texture turning left, standing still
and turning right, and saves the figure to results/<timestamp>_look_at_drum/.

    uv run python scripts/look_at_drum.py --seed 0
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from toyeye.plots import plot_spacetime
from toyeye.utils import make_run_dir, set_seed
from toyeye.world import make_drum, sine_grating, smooth_texture


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--speed", type=float, default=0.5, help="receptors per step")
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--wavelength", type=float, default=8.0)
    parser.add_argument("--smoothness", type=float, default=2.0)
    args = parser.parse_args()

    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    patterns = {
        f"grating (λ = {args.wavelength:g})": sine_grating(args.wavelength),
        f"texture (σ = {args.smoothness:g})": smooth_texture(rng, args.smoothness),
    }
    velocities = [-args.speed, 0.0, args.speed]

    fig, axes = plt.subplots(len(patterns), len(velocities), figsize=(9, 8),
                             sharex=True, sharey=True)
    for row, (name, pattern) in zip(axes, patterns.items()):
        for ax, v in zip(row, velocities):
            plot_spacetime(make_drum(pattern, v, args.steps), ax, f"{name}, v = {v:+g}")
    fig.tight_layout()

    run_dir = make_run_dir("look_at_drum", config=vars(args))
    fig.savefig(run_dir / "spacetime.png", dpi=150)
    print(run_dir / "spacetime.png")


if __name__ == "__main__":
    main()
