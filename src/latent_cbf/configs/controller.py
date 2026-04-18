"""
Controller configuration for DubinsEnv
"""

from dataclasses import dataclass


@dataclass
class ControllerConfig:
    """Configuration for controllers (both simple and MPC)."""
    
    # Controller type
    controller_type: str = "mppi"  # "simple", "mpc", "mppi", or "diffusion"

    # MPPI controller parameters
    num_samples: int = 50
    temperature: float = 2.0
    lambda_param: float = 0.5
    noise_variance: float = 2.0
    adaptive_temperature: bool = True
    
    # Diffusion controller parameters
    checkpoint_path: str = "/data/dubins/diffusion/dubins_diffusion_latest.ckpt"
    config_path: str = "/data/dubins/diffusion/"
    checkpoint_version: int = 1000  # For diffusion_test_latest{i}.ckpt format
    device: str = "cuda:0"
    action_chunk_size: int = 8
    total_chunk_size: int = 16
    eval_diffusion_steps: int = 16
    
    seed: int = None  # Random seed for reproducible behavior
    # Note: max_angular_velocity is now inherited from environment config
