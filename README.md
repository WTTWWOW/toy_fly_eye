# toy_fly_eye

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
| 0 | Setup: repo, environment, lab notebook, fixed seeds | 2–4 | `pyproject.toml`, `LAB_NOTEBOOK.md`, `toyeye.utils.set_seed` |
| 1 | Toy eye: drum world, Reichardt detector, learnable eye, four training objectives, probes | 20–30 | `src/toyeye/`, `tests/`, `notebooks/` |
| 2 | flyvis: tutorials 1–4, pretrained ensemble, T4/T5 probes, Sintel bridge experiment, timing run | 12–20 | `stage2_flyvis/` |
| 3 (optional) | FlyGym: record a fly's-eye walk with efference copy | 10–15 | `stage3_flygym/` |

### Stage 1 checklist

- [x] `world.make_drum(pattern, velocity, steps)` → `steps × 32` array, sub-pixel shifts, wrap-around
- [x] Space–time plots: stripes tilt opposite ways for left vs right motion
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
  plots.py         space–time plots
  utils.py         seeds, device, run directories
tests/             unit tests (run before any training)
notebooks/         exploration and figures
scripts/           training entry points
stage2_flyvis/     flyvis tutorials, probes and the bridge experiment
stage3_flygym/     FlyGym recording script (optional)
pyproject.toml     dependencies (uv); uv.lock pins exact versions
results/           run outputs (git-ignored except for README)
LAB_NOTEBOOK.md    dated log of every run and its settings
```

## Getting started (uv + local VS Code)

The environment is managed with [uv](https://docs.astral.sh/uv/). Install it once
(`curl -LsSf https://astral.sh/uv/install.sh | sh` on Mac/Linux,
`powershell -c "irm https://astral.sh/uv/install.ps1 | iex"` on Windows), then:

```bash
git clone https://github.com/WTTWWOW/toy_fly_eye
cd toy_fly_eye
uv sync          # creates .venv from uv.lock (Python 3.11, per .python-version)
uv run pytest    # all pass; the 7 xfailed are Stage 1 steps not built yet
```

`make setup`, `make test`, `make lab` (JupyterLab in `notebooks/`) and `make fmt` are
shortcuts for the same `uv` commands (`make` is optional; on Windows without WSL, run the
`uv` commands directly).

Then open the folder in VS Code, accept the recommended extensions, and make sure the
interpreter is `.venv` (Ctrl/Cmd+Shift+P, "Python: Select Interpreter"). The Testing panel
runs pytest, and `.vscode/launch.json` has debug configs for the current file and its tests.
In a notebook, pick the `.venv` kernel.

Add packages with `uv add <package>` (or `uv add --dev <package>` for tools) and commit
the updated `pyproject.toml` and `uv.lock`. Stage 2 and 3 dependencies are optional extras:
`uv sync --extra flyvis`, `uv sync --extra flygym`.

**Disk and GPU notes.** On Linux and Windows the default PyTorch wheel is the CUDA build,
so `.venv` is several GB (about 5 GB on Linux). If you only need CPU and want a small
install, follow uv's [PyTorch guide](https://docs.astral.sh/uv/guides/integration/pytorch/)
to point `torch` at the CPU index. Stage 1 runs on a laptop CPU; Stage 2's timing run needs
a GPU (free Colab or a lab machine).

## Other ways to open it

- **GitHub Codespaces**: Code > Codespaces > Create on the repo page. The dev container runs `uv sync` for you.
- **Dev container locally**: open the folder in VS Code and choose "Reopen in Container" (needs Docker).

Work on a branch and open a pull request for changes you want reviewed as diffs; the GitHub pull-request extension is in the recommended set.

## Conventions

- Fix seeds with `toyeye.utils.set_seed` at the top of every script and notebook.
- Log every run in `LAB_NOTEBOOK.md`: date, git commit, command, settings, result.
- Commit every step that works.
