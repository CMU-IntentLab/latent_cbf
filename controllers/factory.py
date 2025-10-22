"""
Controller factory for creating controllers from configuration
"""

from typing import Union
from .simple_controller import SimpleController
from .mpc_controller import MPCController
from .mppi_controller import MPPIController
from .diffusion_controller import DiffusionController
from .random_controller import RandomController


def create_controller_from_config(config) -> Union[SimpleController, MPCController, MPPIController, DiffusionController]:
    """
    Create a controller from configuration.
    
    Args:
        config: Configuration object with controller settings
        
    Returns:
        Configured controller instance
    """
    ctrl_config = config.controller
    
    if ctrl_config.controller_type == "mpc":
        return MPCController(
            prediction_horizon=ctrl_config.prediction_horizon,
            control_horizon=ctrl_config.control_horizon,
            dt=config.environment.dt,
            speed=config.environment.speed,
            max_angular_velocity=config.max_angular_velocity,
            goal_weight=ctrl_config.goal_weight,
            obstacle_weight=ctrl_config.obstacle_weight,
            control_weight=ctrl_config.control_weight,
            obstacle_safety_margin=ctrl_config.obstacle_safety_margin,
            goal_tolerance=ctrl_config.goal_tolerance
        )
    elif ctrl_config.controller_type == "random":
        return RandomController(
            max_angular_velocity=config.environment.max_angular_velocity,
            seed=getattr(ctrl_config, 'seed', None)
        )
    elif ctrl_config.controller_type == "mppi":
        return MPPIController(
            prediction_horizon=ctrl_config.prediction_horizon,
            num_samples=ctrl_config.num_samples,
            dt=config.environment.dt,
            speed=config.environment.speed,
            max_angular_velocity=config.max_angular_velocity,
            temperature=ctrl_config.temperature,
            lambda_param=ctrl_config.lambda_param,
            goal_weight=ctrl_config.goal_weight,
            obstacle_weight=ctrl_config.obstacle_weight,
            control_weight=ctrl_config.control_weight,
            obstacle_safety_margin=ctrl_config.obstacle_safety_margin,
            goal_tolerance=ctrl_config.goal_tolerance,
            noise_variance=ctrl_config.noise_variance,
            adaptive_temperature=ctrl_config.adaptive_temperature
        )
    elif ctrl_config.controller_type == "diffusion":
        return DiffusionController(
            checkpoint_path=ctrl_config.checkpoint_path,
            config_path=ctrl_config.config_path,
            device=ctrl_config.device,
            action_chunk_size=ctrl_config.action_chunk_size,
            total_chunk_size=ctrl_config.total_chunk_size,
            eval_diffusion_steps=ctrl_config.eval_diffusion_steps
        )
    else:  # simple controller (default)
        return SimpleController(
            goal_attraction_gain=ctrl_config.goal_attraction_gain,
            obstacle_repulsion_gain=ctrl_config.obstacle_repulsion_gain,
            obstacle_influence_radius=ctrl_config.obstacle_influence_radius,
            max_angular_velocity=config.max_angular_velocity,
            lookahead_distance=ctrl_config.lookahead_distance,
            proportional_gain=ctrl_config.proportional_gain,
            min_distance_threshold=ctrl_config.min_distance_threshold
        )
