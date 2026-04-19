# latent_cbf

End-to-end pipeline for training a **latent control barrier function (CBF)** that keeps a diffusion policy safe on a Dubins-car obstacle-avoidance task. The CBF is learned in the latent space of a Dreamer-style world model using Hamilton–Jacobi reachability, and at deploy time it filters the actions proposed by the diffusion policy.

Pipeline at a glance:

1. Collect demonstrations in the Dubins environment.
2. Train a diffusion policy (behavior cloning) on those demos.
3. Roll out the diffusion policy from out-of-distribution starts to gather exploratory data.
4. Train a Dreamer world model on the combined trajectories.
5. Learn a safety value function (CBF) on top of the world model's latent state.
6. Deploy diffusion policy + CBF filter and evaluate.

## Repository layout

```
latent_cbf/
  src/latent_cbf/            # latent_cbf code (configs, controllers, scripts, dubins env)
  diffusion4robotics/        # diffusion-policy training package (data4robotics fork)
  PytorchReachability/       # HJ reachability package (PyHJ fork)
  dreamerv3_torch/           # Dreamer V3 world model package (fork)
  data/                      # default artifact tree (change ``DATA_ROOT`` in configs/paths.py to relocate)
```

All three sibling packages are installed editable from the root `pyproject.toml` (via `uv sync`).

## Prerequisites

- This code was tested on a Linux machine equipped with an RTX 4090.
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (manages Python 3.10 and the virtual environment).

## Install

From the repository root:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # or: pip install uv
uv sync
```

This creates `.venv/` (see `.python-version` for the Python pin), resolves dependencies from `uv.lock`, and installs `dreamerv3_torch`, `data4robotics`, `PyHJ`, and `robobuf` in editable mode along with everything else.

Use the environment either by prefixing commands with `uv run` or by activating the venv:

```bash
source .venv/bin/activate
```

If you need a different CUDA or CPU-only PyTorch build than the wheels `uv` selects from PyPI, follow [PyTorch’s install guide](https://pytorch.org/get-started/locally/) and adjust pins or index URLs in `pyproject.toml` / `uv lock` as needed.

## Configuration

All machine-local paths are centralized in `[src/latent_cbf/configs/paths.py](src/latent_cbf/configs/paths.py)`. Set `**DATA_ROOT**` there to the directory that should hold all generated files (the default is `<repository-root>/data`). Under that root the pipeline creates:

```
<DATA_ROOT>/
  trajs/        # HDF5 trajectory files
  buffers/      # diffusion + world-model replay buffers
  diffusion/    # diffusion policy checkpoints
  dreamer/      # world-model logs and RSSM checkpoints
  dreamer/PyHJ/ # reach-avoid value-function checkpoints
```

In command examples below, `<DATA_ROOT>` means the value of `DATA_ROOT` from that file.

### Logging

Training scripts log to [Weights & Biases](https://wandb.ai) under the project `latent_cbf`. Either `wandb login` once, or disable logging for a run with `WANDB_MODE=disabled`.

## Running the pipeline

All commands below are run from `src/latent_cbf/`:

```bash
cd src/latent_cbf
```

Each stage's command, purpose, expected runtime (on a 4090), and output are listed. Earlier stages must finish before later ones.

Either activate the environment once (`source /path/to/latent_cbf/.venv/bin/activate` from the repo root after `uv sync`) and use `python` as shown, or stay unactivated and prefix commands with `uv run python` (or `uv run bash …`) from anywhere under the repository; `uv` discovers the root `pyproject.toml` automatically.

### 1. Collect demonstrations

```bash
python scripts/collect_trajs.py --n_trajectories 200 --save_images --filename dubins_gap
python scripts/visualize_trajs.py   # optional: plot the demos
```

- Produces `<DATA_ROOT>/trajs/dubins_gap.h5`.
- ~2 min.

### 2. Convert to a diffusion-policy buffer

```bash
python scripts/combine_traj_to_buffer_dubins.py --input_file dubins_gap.h5
```

- Produces `<DATA_ROOT>/buffers/buffer.pkl` (robobuf format).
- ~30 s.

### 3. Train the diffusion policy

```bash
python scripts/train_diffusion.py --buffer-path "<DATA_ROOT>/buffers/buffer.pkl"
```

- Produces `<DATA_ROOT>/diffusion/dubins_diffusion_latest{iter}.ckpt` plus `agent_config.yaml`, `exp_config.yaml`, `ob_norm.json`, `ac_norm.json`.
- ~30 min.

### 4. Collect exploratory rollouts from the diffusion policy

```bash
bash scripts/collect_batch_diffusion.sh
```

- Starts the diffusion policy at OOD initial conditions across many checkpoints (500, 1000, …, 19500) to generate diverse trajectories for world-model training.
- Produces `<DATA_ROOT>/trajs/diffusion_trajectories_{iter}.h5`.
- ~0.5–1 hr. Individual checkpoints can be inspected with:

```bash
python scripts/visualize_trajs.py --filepath "<DATA_ROOT>/trajs/diffusion_trajectories_19500.h5"
```

### 5. Build the world-model training buffer

```bash
python scripts/combine_trajectories.py
```

- Concatenates every `*.h5` under `trajs/` into `<DATA_ROOT>/buffers/dreamer_buffer.h5`.

### 6. Train the Dreamer world model

```bash
python scripts/dreamer_offline.py
```

- Produces `<DATA_ROOT>/dreamer/rssm_ckpt.pt` and logs under `<DATA_ROOT>/dreamer/`.
- 2 hours. The car may not be visible in reconstructions until ~6–8 k iterations.

Optional — check the learned margin head on a trajectory:

```bash
python scripts/wm_trajectory_stats.py \
    --traj-h5 "<DATA_ROOT>/trajs/diffusion_trajectories_19500.h5"
```

### 7. Train the HJ safety value function

```bash
python scripts/wm_ddpg.py            # with gradient-penalty regularization (default)
python scripts/wm_ddpg.py --no_gp    # ablation: without gradient penalty
```

- Produces `<DATA_ROOT>/dreamer/PyHJ/{gp,nogp}/epoch_id_*/policy.pth`.

### 8. Evaluate with the CBF filter

Run the diffusion policy wrapped by the world-model CBF filter:

```bash
python scripts/collect_trajs.py \
    --controller diffusion --config diffusion_wm \
    --use_wm_prediction --wm_history_length 8 \
    --wm_checkpoint "<DATA_ROOT>/dreamer/rssm_ckpt.pt" \
    --filename wm_test_cbf --n_trajectories 5 --save_images \
    --filter_mode cbf   # or: lr, none
```

Visualize the results (picks `wm_test_{cbf,lr}.h5` automatically from `--filter_mode`):

```bash
python scripts/eval_dreamer.py --filter_mode cbf
```

Gifs and plots land in `src/latent_cbf/visualizations/`.

## Acknowledgements

This repository vendors forks of three upstream projects, each retaining its original license:

- [dreamerv3-torch](https://github.com/NM512/dreamerv3-torch) (MIT).
- [data4robotics](https://github.com/SudeepDasari/data4robotics) (see `diffusion4robotics/LICENSE.md`).
- [PyHJ](https://github.com/jamesjingqili/Lipschitz_Continuous_Reachability_Learning) — Jingqi Li's Lipschitz-continuous reachability learning code.

## License

See `[LICENSE](LICENSE)` at the repository root.

## Citation

If you use this code, please cite the upstream projects listed above. A BibTeX entry for `latent_cbf` will be added here when the accompanying paper is released.