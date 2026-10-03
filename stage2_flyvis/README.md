# Stage 2: hands-on with flyvis

- [ ] Tutorials 1–4: connectome, optic-flow task, flash responses, moving-edge responses ([flyvis](https://github.com/TuragaLab/flyvis); fall back to Colab if the install fights you)
- [ ] Load the pretrained ensemble and its best model ([training tutorial](https://turagalab.github.io/flyvis/examples/02_flyvision_optic_flow_task/))
- [ ] Probe T4/T5: four preferred directions per type, T4 ON, T5 OFF (reuse `toyeye.probes`)
- [ ] Bridge experiment: downsampled Sintel, photometric error for true vs zero flow with `grid_sample`, on original and repeated frames
- [ ] Time a few hundred supervised iterations on a GPU; record seconds per iteration in `LAB_NOTEBOOK.md`

Environment: `uv sync --extra flyvis`. Sintel setup: https://turagalab.github.io/flyvis/install/
