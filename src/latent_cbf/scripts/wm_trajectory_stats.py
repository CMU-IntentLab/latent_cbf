#!/usr/bin/env python3
"""
Load a Dreamer RSSM checkpoint and evaluate margin-head predictions on saved trajectories.

Computes (margin_gp head only):
  - confusion-style rates (TP / FP / TN / FN) vs collision labels
  - per-trajectory max |Δℓ| (max_diffl_gp)
  - mean and std of |margin_gp| over all timesteps (all trajectories)
  - F1 scores: f1_safe and f1_alarm

Example (checkpoint from hyperparameters, same layout as dreamer_offline.py;
``<DATA_ROOT>`` is ``DATA_ROOT`` in ``configs.paths``):
  python wm_trajectory_stats.py \\
    --relu_weight 0.5 --zs_weight 0.05 \\
    --traj-h5 "<DATA_ROOT>/trajs/manual_none.h5"

Example (explicit file):
  python wm_trajectory_stats.py \\
    --rssm-ckpt "<DATA_ROOT>/dreamer/PyHJ/gp/relu_weight_0.5_gp_weight_10.0_zs_weight_0.05/rssm_ckpt_39999.pt" \\
    --traj-h5 "<DATA_ROOT>/trajs/manual_none.h5"
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import gym
import h5py
import numpy as np
import torch
from tqdm import tqdm

# Paths: repo root (dreamerv3_torch), package root (configs), this dir (dreamer_offline)
_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_PKG_ROOT = _SCRIPT_DIR.parent
_REPO_ROOT = _PKG_ROOT.parent.parent

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from dreamerv3_torch import tools  # noqa: E402
from dreamer_offline import Dreamer  # noqa: E402
from configs import Config, DreamerConfig  # noqa: E402
from configs.paths import DREAMER_DIR, TRAJS_DIR  # noqa: E402


def _build_agent(config: DreamerConfig, rssm_ckpt: pathlib.Path, device: torch.device) -> Dreamer:
    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()

    logdir = pathlib.Path(config.logdir).expanduser()
    config.steps //= config.action_repeat
    config.eval_every //= config.action_repeat
    config.log_every //= config.action_repeat
    logdir.mkdir(parents=True, exist_ok=True)
    step = 0
    logger = tools.DebugLogger(logdir, config.action_repeat * step)

    action_space = gym.spaces.Box(
        low=-config.turnRate, high=config.turnRate, shape=(1,), dtype=np.float32
    )
    bounds = np.array(
        [[config.x_min, config.x_max], [config.y_min, config.y_max], [0, 2 * np.pi]]
    )
    low, high = bounds[:, 0], bounds[:, 1]
    midpoint = (low + high) / 2.0
    interval = high - low
    gt_observation_space = gym.spaces.Box(
        np.float32(midpoint - interval / 2),
        np.float32(midpoint + interval / 2),
    )
    image_size = config.size[0]
    image_observation_space = gym.spaces.Box(
        low=0, high=255, shape=(image_size, image_size, 3), dtype=np.uint8
    )
    obs_observation_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
    observation_space = gym.spaces.Dict(
        {
            "state": gt_observation_space,
            "obs_state": obs_observation_space,
            "image": image_observation_space,
        }
    )
    config.num_actions = (
        action_space.n if hasattr(action_space, "n") else action_space.shape[0]
    )

    agent = Dreamer(observation_space, action_space, config, logger, None).to(device)
    ckpt = torch.load(rssm_ckpt, map_location=device)
    agent.load_state_dict(ckpt["agent_state_dict"])
    agent.eval()
    return agent


def margins_gp_from_traj(traj: h5py.Group, agent: Dreamer, hist: int) -> np.ndarray:
    obs = traj["observations"][:]
    actions = traj["actions"][:]
    states = traj["states"][:]
    states_inp = np.concatenate(
        [np.cos(states[:, [1]]), np.sin(states[:, [-1]])], axis=-1
    )

    margin_gps: list[float] = []

    for i in range(hist + 1, obs.shape[0]):
        is_first = torch.zeros(hist)
        is_first[0] = 1
        obs_batch = {
            "image": obs[None, i - hist : i],
            "obs_state": states_inp[None, i - hist : i],
            "action": actions[None, i - hist - 1 : i - 1, None],
            "is_first": is_first[None, :],
            "is_terminal": torch.zeros(1, hist),
        }
        data = agent._wm.preprocess(obs_batch)
        embed = agent._wm.encoder(data)
        st, _ = agent._wm.dynamics.observe(embed, data["action"], data["is_first"])
        feat = agent._wm.dynamics.get_feat(st)
        margin_gps.append(
            agent._wm.heads["margin_gp"](feat).squeeze().detach().cpu().numpy()[-1]
        )

    return np.tanh(np.array(margin_gps).flatten())


def _f1(tp: int, fp: int, fn: int) -> float:
    """Binary F1: positive class with counts TP, FP, FN (TN unused)."""
    denom = 2 * tp + fp + fn
    return float(2 * tp / denom) if denom > 0 else float("nan")


def confusion_rates_margin_gp(
    collisions: np.ndarray,
    gp_fails: np.ndarray,
) -> dict[str, float | int]:
    c = len(collisions)
    assert c == len(gp_fails)

    tp = np.sum(np.logical_and(np.logical_not(gp_fails), np.logical_not(collisions)))
    fp = np.sum(np.logical_and(np.logical_not(gp_fails), collisions))
    tn = np.sum(np.logical_and(gp_fails, collisions))
    fn = np.sum(np.logical_and(gp_fails, np.logical_not(collisions)))
    tp, fp, tn, fn = int(tp), int(fp), int(tn), int(fn)
    f1_safe = _f1(tp, fp, fn)
    f1_alarm = _f1(tn, fn, fp)
    return {
        "TP_pct": float(tp / c * 100),
        "FP_pct": float(fp / c * 100),
        "TN_pct": float(tn / c * 100),
        "FN_pct": float(fn / c * 100),
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "n": int(c),
        "f1_safe": f1_safe,
        "f1_alarm": f1_alarm,
    }


def resolve_rssm_ckpt(args: argparse.Namespace) -> pathlib.Path:
    """Either use --rssm-ckpt or build .../PyHJ/gp/relu_*_gp_*_zs_*/rssm_ckpt_{step}.pt."""
    if args.rssm_ckpt is not None:
        return args.rssm_ckpt.expanduser().resolve()
    if args.relu_weight is None or args.zs_weight is None:
        raise SystemExit(
            "Provide --rssm-ckpt, or both --relu_weight and --zs_weight "
            "(optional --gp_weight, default 10; --rssm-step, default 19999)."
        )
    run_dir = (
        args.dreamer_root.expanduser().resolve()
        / "PyHJ"
        / "gp"
        / f"relu_weight_{args.relu_weight}_gp_weight_{args.gp_weight}_zs_weight_{args.zs_weight}"
    )
    return (run_dir / f"rssm_ckpt_{args.rssm_step}.pt").resolve()
    return (run_dir / f"best_rssm_ckpt_{args.rssm_step}_0_00.pt").resolve()


def main() -> None:
    p = argparse.ArgumentParser(description="WM margin stats from trajectories + RSSM ckpt")
    p.add_argument(
        "--rssm-ckpt",
        type=pathlib.Path,
        default=None,
        help="Path to rssm_ckpt*.pt; if omitted, set --relu_weight and --zs_weight",
    )
    p.add_argument(
        "--relu_weight",
        type=float,
        default=None,
        help="Run folder relu_weight_* (with --zs_weight; matches dreamer_offline training)",
    )
    p.add_argument(
        "--zs_weight",
        type=float,
        default=None,
        help="Run folder zs_weight_* (with --relu_weight)",
    )
    p.add_argument(
        "--gp_weight",
        type=float,
        default=10.0,
        help="Run folder gp_weight_* when resolving path (default 10, same as dreamer_offline)",
    )
    p.add_argument(
        "--rssm-step",
        type=int,
        default=39999,
        help="Checkpoint rssm_ckpt_{step}.pt when resolving from weights",
    )
    p.add_argument(
        "--dreamer-root",
        type=pathlib.Path,
        default=DREAMER_DIR,
        help="Root containing PyHJ/gp/... run directories",
    )
    p.add_argument(
        "--traj-h5",
        type=pathlib.Path,
        default=TRAJS_DIR / "manual_none.h5",
        help="HDF5 file with a 'trajectories' group",
    )
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--hist", type=int, default=5, help="WM history length (must match training)")
    p.add_argument(
        "--collision-skip",
        type=int,
        default=5,
        help="Use collision labels from this index onward (matches notebook attrs['collision'][5:])",
    )
    p.add_argument(
        "--out-json",
        type=pathlib.Path,
        default=None,
        help="Optional path to write metrics as JSON",
    )
    p.add_argument(
        "--out-npz",
        type=pathlib.Path,
        default=None,
        help="Optional path to save max_diffl_gp array",
    )
    args = p.parse_args()

    rssm_path = resolve_rssm_ckpt(args)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    config = DreamerConfig()
    env_conf = Config()
    config.turnRate = env_conf.max_angular_velocity
    config.x_min = env_conf.environment.world_bounds[0]
    config.x_max = env_conf.environment.world_bounds[1]
    config.y_min = env_conf.environment.world_bounds[2]
    config.y_max = env_conf.environment.world_bounds[3]
    config.size = env_conf.environment.image_size
    config.device = str(device)
    config.logdir = str(rssm_path.parent)
    if args.relu_weight is not None:
        config.relu_weight = args.relu_weight
    if args.zs_weight is not None:
        config.zs_weight = args.zs_weight
    config.gp_weight = args.gp_weight

    if not rssm_path.is_file():
        raise FileNotFoundError(f"RSSM checkpoint not found: {rssm_path}")
    if not args.traj_h5.is_file():
        raise FileNotFoundError(f"Trajectory file not found: {args.traj_h5}")

    print("Loading agent from", rssm_path)
    agent = _build_agent(config, rssm_path, device)
    print("Agent loaded successfully")
    hist = args.hist

    margins_gp: list[np.ndarray] = []
    col_parts: list[np.ndarray] = []
    with h5py.File(args.traj_h5, "r") as f:
        trajs = f["trajectories"]
        # use tqdm to print the progress
        for traj_key in tqdm(trajs.keys(), total=len(trajs.keys()), desc="Processing trajectories"):
            t = trajs[traj_key]
            mg = margins_gp_from_traj(t, agent, hist)
            margins_gp.append(mg)
            col_parts.append(t.attrs["collision"][args.collision_skip :])

    diff_margins_gp = [np.diff(m) for m in margins_gp]
    max_diffl_gp = np.array([float(np.max(np.abs(d))) for d in diff_margins_gp])


    collisions = np.concatenate(col_parts)
    margins_flat = np.concatenate(margins_gp)
    gp_fails = margins_flat < 0.0
    abs_m = np.abs(margins_flat)
    mean_abs_margin_gp = float(np.mean(abs_m))
    max_abs_margin_gp = float(np.max(abs_m))
    std_abs_margin_gp = float(np.std(abs_m))

    if len(collisions) != len(gp_fails):
        raise ValueError(
            f"Collision length {len(collisions)} != margin length {len(gp_fails)}. "
            "Check --collision-skip and trajectory format."
        )

    conf = confusion_rates_margin_gp(collisions, gp_fails)

    print(
        "collisions total",
        len(collisions),
        "sum(collisions)",
        int(np.sum(collisions)),
        "sum(~collisions)",
        int(np.sum(np.logical_not(collisions))),
    )
    print("margin_gp  TP%     FP%     TN%     FN%   |  TP  FP  TN  FN")
    print(
        f"           {conf['TP_pct']:7.5f} {conf['FP_pct']:7.5f} {conf['TN_pct']:7.5f} {conf['FN_pct']:7.5f}   | "
        f"{conf['TP']:4d} {conf['FP']:4d} {conf['TN']:4d} {conf['FN']:4d}"
    )
    
    print(
        "F1_alarm (collision, pred fail):    "
        f"{conf['f1_alarm']:.3f}"
    )
    print(
        "max_diffl_gp   mean",
        f"{float(np.mean(max_diffl_gp)):.2f}",
        "std",
        f"{float(np.std(max_diffl_gp)):.2f}",
        "shape",
        max_diffl_gp.shape,
    )
    print(
        "|margin_gp| mean",
        f"{mean_abs_margin_gp:.2f}",
        "std",
        f"{std_abs_margin_gp:.2f}",
        "n_steps",
        len(margins_flat),
    )
    
    out_payload = {
        "rssm_ckpt": os.fspath(rssm_path),
        "run_dir": os.fspath(rssm_path.parent),
        "relu_weight": args.relu_weight,
        "zs_weight": args.zs_weight,
        "gp_weight": args.gp_weight,
        "rssm_step": args.rssm_step,
        "traj_h5": os.fspath(args.traj_h5),
        "hist": hist,
        "collision_skip": args.collision_skip,
        "margin_gp": conf,
        "mean_abs_margin_gp": mean_abs_margin_gp,
        "std_abs_margin_gp": std_abs_margin_gp,
        "margin_gp_timesteps": len(margins_flat),
        "max_diffl_gp_mean": float(np.mean(max_diffl_gp)),
        "max_diffl_gp_std": float(np.std(max_diffl_gp)),
    }
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as jf:
            json.dump(out_payload, jf, indent=2)
        print("Wrote", args.out_json)

    if args.out_npz:
        args.out_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.out_npz, max_diffl_gp=max_diffl_gp)
        print("Wrote", args.out_npz)


if __name__ == "__main__":
    main()
