# toy-fly-eye

**Teach a tiny eye to see motion.** A one-dimensional fly eye watches a rotating drum,
is trained with and without labels, and is probed like an electrophysiologist would.
The same probes are then pointed at the real connectome-constrained
[flyvis](https://github.com/TuragaLab/flyvis) models.

This is a warm-up for a final year project on self-supervised learning (SSL) and fly
vision. Everything here is meant to be reused there: the drum world becomes the unit
tests for every SSL loss, `probes.py` becomes the neural-evaluation pipeline, and the
timing run sizes the flagship's ensembles.

## Stages

| Stage | What | Hours | Where in the repo |
|---|---|---|---|
| 0 | Setup: repo, environment, lab notebook, fixed seeds | 2–4 | `envs/`, `LAB_NOTEBOOK.md`, `toyeye.utils.set_seed` |
| 1 | Toy eye: drum world, Reichardt detector, learnable eye, four training objectives, probes | 20–30 | `src/toyeye/`, `tests/`, `notebooks/` |
| 2 | flyvis: tutorials 1–4, pretrained ensemble, T4/T5 probes, Sintel bridge experiment, timing run | 12–20 | `stage2_flyvis/` |
| 3 (optional) | FlyGym: record a fly's-eye walk with efference copy | 10–15 | `stage3_flygym/` |

### Stage 1 checklist

- [ ] `world.make_drum(pattern, velocity, steps)` → `steps × 32` array, sub-pixel shifts, wrap-around
- [ ] Space–time plots: stripes tilt opposite ways for left vs right motion
- [ ] Hand-built Hassenstein–Reichardt detector whose sign tracks direction
- [ ] Warp unit tests pass (true v gives near-zero error, v = 0 clearly more)
- [ ] Tiny learnable eye: 1D conv across receptors + per-column GRU with shared weights
- [ ] Train four ways from the same initial weights: supervised oracle, temporal prediction (k = 1, 3, 5), photometric warp, latent prediction (optional)
- [ ] Bonus trap: repeat every frame twice, see what the warp eye learns, then fix it
- [ ] Run every probe in `probes.py` on all four eyes

## Repository layout

```
src/toyeye/        the reusable package
  world.py         drum stimuli: gratings, smooth textures, make_drum
  reichardt.py     hand-built Hassenstein–Reichardt detector
  warp.py          rebuild one frame from another given a velocity
  models.py        the tiny learnable eye (conv + shared GRU)
  losses.py        oracle, temporal-prediction, photometric-warp, latent-prediction
  probes.py        velocity error, copy baseline, DSI, TF tuning, generalisation, collapse
  utils.py         seeds, device, run directories
tests/             unit tests (run before any training)
notebooks/         exploration and figures
scripts/           training entry points
stage2_flyvis/     flyvis tutorials, probes and the bridge experiment
stage3_flygym/     FlyGym recording script (optional)
envs/              one conda environment per stage
results/           run outputs (git-ignored except for README)
LAB_NOTEBOOK.md    dated log of every run and its settings
```

## Getting started

```bash
conda env create -f envs/stage1.yml
conda activate toyeye
pip install -e .
pytest
```

Stage 1 runs on a laptop CPU. Stage 2's timing run needs a GPU (free Colab or a lab
machine); use `envs/stage2-flyvis.yml`. Stage 3 uses `envs/stage3-flygym.yml`.

## Conventions

- Fix seeds with `toyeye.utils.set_seed` at the top of every script and notebook.
- Log every run in `LAB_NOTEBOOK.md`: date, git commit, command, settings, result.
- Commit every step that works.
