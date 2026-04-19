"""
Controller configuration for DubinsEnv.

Only MPPI and diffusion policies are supported (including ``diffusion_wm`` for
world-model–filtered diffusion).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from .paths import DIFFUSION_CHECKPOINT, DIFFUSION_DIR

ControllerName = Literal["mppi", "diffusion", "diffusion_wm"]


@dataclass
class ControllerConfig:
    """Parameters for MPPI or diffusion controllers."""

    controller_type: ControllerName = "mppi"

    # --- MPPI (used when controller_type == "mppi") ---
    prediction_horizon: int = 10
    num_samples: int = 50
    temperature: float = 2.0
    lambda_param: float = 0.5
    noise_variance: float = 2.0
    adaptive_temperature: bool = True
    warm_start: bool = True
    goal_weight: float = 10.0
    obstacle_weight: float = 50.0
    control_weight: float = 0.1
    obstacle_safety_margin: float = 0.3
    goal_tolerance: float = 0.1

    # --- Diffusion / diffusion_wm (same checkpoint fields; WM is toggled via DreamerConfig) ---
    checkpoint_path: str = str(DIFFUSION_CHECKPOINT)
    config_path: str = f"{DIFFUSION_DIR}/"
    checkpoint_version: int = 1000
    device: str = "cuda:0"
    action_chunk_size: int = 8
    total_chunk_size: int = 16
    eval_diffusion_steps: int = 16

    seed: Optional[int] = None
