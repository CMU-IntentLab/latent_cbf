"""
Preset configurations for common use cases
"""

import numpy as np
from .base import Config
from .environment import EnvironmentConfig, ObstacleConfig
from .controller import ControllerConfig
from .rendering import RenderingConfig
from .experiment import ExperimentConfig


def get_default_config() -> Config:
    """Get default configuration."""
    return Config()


def get_no_obstacles_config() -> Config:
    """Get configuration with no obstacles."""
    config = Config()
    config.environment.obstacles = []
    return config



def get_debug_config() -> Config:
    """Get configuration optimized for debugging."""
    config = Config()
    config.environment.set_deterministic_reset(-1.2, 0.0, 0.0)
    config.experiment.verbose = True
    config.experiment.n_episodes = 1
    config.experiment.save_images = True
    config.experiment.save_video = True
    return config


def get_mpc_config() -> Config:
    """Get configuration using MPC controller."""
    config = Config()
    config.controller.controller_type = "mpc"
    config.experiment.video_filename = "video/mpc_trajectory.mp4"
    return config

def get_random_config() -> Config:
    """Get random controller configuration for data collection."""
    config = get_default_config()
    config.controller.controller_type = "random"
    config.controller.seed = 0  # Set seed for reproducible random behavior
    # You might want different environment settings for random data collection
    config.environment.set_reset_bounds(
        x_range=(-1.5, 1.5),
        y_range=(-1.5, 1.5),
        theta_range=(-np.pi, np.pi)
    )
    
    return config

def get_mppi_config() -> Config:
    """Get configuration using MPPI controller."""
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
        x_range=(-1.5, -1  ),
        y_range=(-1., 1.),
        theta_range=(-np.pi/3, np.pi/3) 
    )
    return config


def get_diffusion_config(checkpoint_version: int = 1000) -> Config:
    """Get diffusion policy configuration."""
    config = Config()
    config.controller.controller_type = "diffusion"
    config.controller.config_path = "/data/dubins/diffusion/"
    config.controller.checkpoint_version = checkpoint_version
    config.controller.checkpoint_path = f"/data/dubins/diffusion/dubins_diffusion_latest{checkpoint_version}.ckpt"
    config.controller.device = "cuda"
    config.controller.action_chunk_size = 8
    config.controller.total_chunk_size = 16
    config.controller.eval_diffusion_steps = 16
    config.controller.max_angular_velocity = 2.0
    config.experiment.video_filename = f"video/diffusion_trajectory_ckpt{checkpoint_version}.mp4"
    config.environment.set_reset_bounds(
        x_range=(-1.5, 1.5),
        y_range=(-1.5, 1.5),
        theta_range=(-np.pi, np.pi) 
    )
    return config

def get_diffusion_collection_config(checkpoint_version: int = 1000) -> Config:
    """Get diffusion policy configuration."""
    config = Config()
    config.controller.controller_type = "diffusion"
    config.controller.config_path = "/data/dubins/diffusion/"
    config.controller.checkpoint_version = checkpoint_version
    config.controller.checkpoint_path = f"/data/dubins/diffusion/dubins_diffusion_latest{checkpoint_version}.ckpt"
    config.controller.device = "cuda"
    config.controller.action_chunk_size = 8
    config.controller.total_chunk_size = 16
    config.controller.eval_diffusion_steps = 16
    config.controller.max_angular_velocity = 2.0
    config.experiment.video_filename = f"video/diffusion_trajectory_ckpt{checkpoint_version}.mp4"
    config.environment.set_reset_bounds(
        x_range=(-1.5, -1),
        y_range=(-1., 1.),
        theta_range=(-np.pi/3, np.pi/3) 
    )
    return config


def get_diffusion_wm_config(checkpoint_version: int = 1000, wm_checkpoint_path: str = '/data/dubins/test/dreamer/rssm_ckpt.pt') -> Config:
    """Get diffusion policy configuration with world model prediction enabled."""
    config = get_diffusion_config(checkpoint_version)
    config.controller.controller_type = "diffusion_wm"

    # Enable world model prediction
    config.wm_config = {
        'use_wm_prediction': True,
        'wm_checkpoint_path': wm_checkpoint_path,
        'wm_history_length': 8
    }
    
    config.environment.set_reset_bounds(
        x_range=(-1.5, -1),
        y_range=(-1., 1.),
        theta_range=(-np.pi/3, np.pi/3) 
    )
    config.experiment.video_filename = f"video/diffusion_wm_trajectory_ckpt{checkpoint_version}.mp4"
    return config
