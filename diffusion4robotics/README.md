# Dubins diffusion BC (minimal)

Behavior cloning with **DiffusionUnet** + **RobomimicResNet** on **dubins** buffers only (`obs_dim=1`, `ac_dim=1`). Training uses `train.py` and plain `defaults.yaml` (no Hydra).

**Note:** If CPU image augmentation is a bottleneck, use `--train-transform gpu_medium` (augmentation runs on the GPU in the training loop).

## Install

```bash
conda env create -f env.yml
conda activate data4robotics
pip install git+https://github.com/AGI-Labs/robobuf.git
pip install -e ./
```

## Train

From this directory (with the env activated):

```bash
nice -n 19 python train.py \
  --buffer-path /path/to/buf.pkl \
  --exp-name dubins_run \
  --wandb-name diffusion_singlecam \
  --max-iterations 20000 \
  --ac-chunk 16 \
  --train-transform medium \
  --cam-indexes 0 \
  --feature-dim 256
```

- `--cam-indexes`: one or more integers, e.g. `--cam-indexes 0` or `--cam-indexes 0 2`.
- `--defaults`: optional path to a YAML that overrides [defaults.yaml](defaults.yaml) (architecture / schedule).
- `--checkpoint-dir`: optional; overrides `checkpoint_dir` in [defaults.yaml](defaults.yaml) for this run.

Checkpoints, `exp_config.yaml`, and `agent_config.yaml` go under **`checkpoint_dir`** in [defaults.yaml](defaults.yaml) (default `/data/dubins/diffusion`), as `<checkpoint_dir>/<exp_name>_latest.ckpt`. Resuming: run again with the same `checkpoint_dir` so `exp_config.yaml` is found there.

## Robobuf conversion

Convert trajectories to robobuf format as in the upstream [data4robotics](https://github.com/SudeepDasari/data4robotics) README (pseudo-code for `buf.pkl`).

## Citation

If you use this code or representations from the original project:

```bibtex
@inproceedings{dasari2023datasets,
      title={An Unbiased Look at Datasets for Visuo-Motor Pre-Training},
      author={Dasari, Sudeep and Srirama, Mohan Kumar and Jain, Unnat and Gupta, Abhinav},
      booktitle={Conference on Robot Learning},
      year={2023},
      organization={PMLR}
    }
```

## Scope

This fork keeps only the **dubins** single-pipeline path. For other action spaces (e.g. absolute + R6), change `obs_dim` / `ac_dim` and buffer construction in `train.py` and rebuild buffers accordingly.
