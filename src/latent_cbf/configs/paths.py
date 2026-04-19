"""Data-root paths for checkpoints, buffers, and trajectories.

Edit ``DATA_ROOT`` below to point all pipeline artifacts at a directory on
this machine. Every other path in this module is derived from it.

All machine-local file paths used by the codebase should be defined here.
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

# Machine-local root for trajectories, buffers, checkpoints, and logs.
DATA_ROOT = (REPO_ROOT / "data").resolve()

# Top-level subtrees under the data root.
DIFFUSION_DIR = DATA_ROOT / "diffusion"
DREAMER_DIR = DATA_ROOT / "dreamer"
BUFFERS_DIR = DATA_ROOT / "buffers"
TRAJS_DIR = DATA_ROOT / "trajs"

# Canonical artifact paths.
DIFFUSION_CHECKPOINT = DIFFUSION_DIR / "dubins_diffusion_latest.ckpt"
DREAMER_BUFFER = BUFFERS_DIR / "dreamer_buffer.h5"
RSSM_CHECKPOINT = DREAMER_DIR / "rssm_ckpt.pt"
FILTER_GP = DREAMER_DIR / "PyHJ/gp/epoch_id_14/policy.pth"
FILTER_NOGP = DREAMER_DIR / "PyHJ/nogp/epoch_id_14/policy.pth"

# Repo-relative file paths (not under DATA_ROOT).
DIFFUSION4ROBOTICS_DEFAULTS = REPO_ROOT / "diffusion4robotics" / "defaults.yaml"
