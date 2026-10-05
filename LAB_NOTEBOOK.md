# Lab notebook

One entry per working session or run. Newest at the bottom.

Template:

```
## YYYY-MM-DD
- Commit:
- Command / notebook:
- Settings (seed, k, lr, steps, pattern, ...):
- Result:
- Next:
```

## 2026-10-03
- Created the repository and the Stage 0 scaffold.
- Next: create the `toyeye` environment, then implement `make_drum` and its plots.

## 2026-10-05
- Commit: Stage 1 drum world (branch `claude/project-thread-vxvn6a`)
- Command / notebook: `uv run python scripts/look_at_drum.py --seed 0`; `uv run pytest`
- Settings: seed 0, speed ±0.5 receptors/step, 64 steps, grating λ = 8, texture σ = 2
- Result: `sine_grating`, `smooth_texture`, `make_drum` implemented; 9 new world tests pass.
  Space–time plots tilt down-left for v < 0 and down-right for v > 0, as expected.
  Note: with linear interpolation applied twice (drum, then warp), a λ = 8 grating leaves
  a max MSE of about 2.7e-3 at the true velocity, above the 1e-3 bound in `test_warp.py`.
- Next: Reichardt detector; decide how to handle the warp-test threshold.
