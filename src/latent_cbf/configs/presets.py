"""
Preset configurations for MPPI and diffusion pipelines.
"""

from __future__ import annotations

import numpy as np

from .base import Config
from .environment import ObstacleConfig
from .paths import DIFFUSION_DIR, RSSM_CHECKPOINT


def get_default_config() -> Config:
    """Default: Dubins task with MPPI (same defaults as :class:`ControllerConfig`)."""
    return Config()


def get_no_obstacles_config() -> Config:
    """Same as default but with no obstacles."""
    config = Config()
    config.environment.obstacles = []
    return config


def get_debug_config() -> Config:
    """Short runs with deterministic start and rich logging."""
    config = Config()
    config.environment.set_deterministic_reset(-1.2, 0.0, 0.0)
    config.experiment.verbose = True
    config.experiment.n_episodes = 1
    config.experiment.save_images = True
    config.experiment.save_video = True
    return config


def get_narrow_gap_config() -> Config:
    """MPPI with slightly tighter obstacles (narrower passage) for stress tests."""
    config = get_mppi_config()
    config.environment.obstacles = [
        ObstacleConfig(x=0.25, y=0.5, radius=0.35),
        ObstacleConfig(x=0.25, y=-0.5, radius=0.35),
    ]
    return config


def get_mppi_config() -> Config:
    """MPPI with sampling and cost weights tuned for Dubins gap navigation."""
    config = Config()
    config.controller.controller_type = "mppi"
    config.controller.prediction_horizon = 8
    config.controller.num_samples = 500
    config.controller.temperature = 3.0
    config.controller.lambda_param = 0.2
    config.controller.noise_variance = 3.0
    config.controller.goal_weight = 10.0
    config.controller.obstacle_weight = 10.0
    config.controller.control_weight = 0.01
    config.controller.obstacle_safety_margin = 0.05
    config.controller.goal_tolerance = 0.1
    config.controller.adaptive_temperature = True
    config.experiment.video_filename = "video/mppi_trajectory.mp4"
    config.environment.set_reset_bounds(
        x_range=(-1.5, -1.0),
        y_range=(-1.0, 1.0),
        theta_range=(-np.pi / 3, np.pi / 3),
    )
    return config


def _apply_diffusion_controller(config: Config, checkpoint_version: int) -> None:
    ctrl = config.controller
    ctrl.controller_type = "diffusion"
    ctrl.config_path = f"{DIFFUSION_DIR}/"
    ctrl.checkpoint_version = checkpoint_version
    ctrl.checkpoint_path = f"{DIFFUSION_DIR}/dubins_diffusion_latest{checkpoint_version}.ckpt"
    ctrl.device = "cuda"
    ctrl.action_chunk_size = 8
    ctrl.total_chunk_size = 16
    ctrl.eval_diffusion_steps = 16
    config.environment.max_angular_velocity = 2.0
    config.experiment.video_filename = f"video/diffusion_trajectory_ckpt{checkpoint_version}.mp4"


def get_diffusion_config(checkpoint_version: int = 1000) -> Config:
    """Diffusion BC policy with full-box random resets."""
    config = Config()
    _apply_diffusion_controller(config, checkpoint_version)
    config.environment.set_reset_bounds(
        x_range=(-1.5, 1.5),
        y_range=(-1.5, 1.5),
        theta_range=(-np.pi, np.pi),
    )
    return config


def get_diffusion_collection_config(checkpoint_version: int = 1000) -> Config:
    """Diffusion rollouts with the initial region data collection."""
    config = Config()
    _apply_diffusion_controller(config, checkpoint_version)
    config.environment.set_reset_bounds(
        x_range=(-1.5, -1.0),
        y_range=(-1.0, 1.0),
        theta_range=(-np.pi / 3, np.pi / 3),
    )
    return config


def get_diffusion_wm_config(
    checkpoint_version: int = 1000, wm_checkpoint_path: str = str(RSSM_CHECKPOINT)
) -> Config:
    """Diffusion + WM filter: same as collection resets, enables WM fields on config."""
    config = get_diffusion_config(checkpoint_version)
    config.controller.controller_type = "diffusion_wm"
    config.wm_config = {
        "use_wm_prediction": True,
        "wm_checkpoint_path": wm_checkpoint_path,
        "wm_history_length": 8,
    }
    config.environment.set_reset_bounds(
        x_range=(-1.5, -1.0),
        y_range=(-1.0, 1.0),
        theta_range=(-np.pi / 3, np.pi / 3),
    )
    config.experiment.video_filename = f"video/diffusion_wm_trajectory_ckpt{checkpoint_version}.mp4"
    return config
