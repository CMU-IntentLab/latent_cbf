"""
Controller factory for creating controllers from configuration.
"""

from __future__ import annotations

from typing import Optional, Union

from configs import Config
from configs.dreamer_conf import DreamerConfig

from .simple_controller import SimpleController
from .mpc_controller import MPCController
from .mppi_controller import MPPIController
from .diffusion_controller import DiffusionController, FilteredDiffusionController
from .random_controller import RandomController


def _make_diffusion_controller(ctrl_config) -> DiffusionController:
    return DiffusionController(
        checkpoint_path=ctrl_config.checkpoint_path,
        config_path=ctrl_config.config_path,
        device=ctrl_config.device,
        action_chunk_size=ctrl_config.action_chunk_size,
        total_chunk_size=ctrl_config.total_chunk_size,
        eval_diffusion_steps=ctrl_config.eval_diffusion_steps
    )


def create_controller_from_config(
    config: Config,
    wm_config: Optional[DreamerConfig] = None,
) -> Union[
    SimpleController,
    MPCController,
    MPPIController,
    DiffusionController,
    FilteredDiffusionController,
]:
    """
    Create a controller from configuration.

    For ``diffusion`` / ``diffusion_wm``, builds a :class:`DiffusionController` and
    optionally wraps it with :class:`FilteredDiffusionController` when ``wm_config``
    is given and ``wm_config.use_wm_prediction`` is True.

    MPC/MPPI fields may be set on ``config.controller`` by presets (dynamic attributes);
    missing attributes fall back to the same defaults as the controller classes.
    """
    ctrl_config = config.controller
    ctype = ctrl_config.controller_type

    if ctype == "mpc":
        return MPCController(
            prediction_horizon=ctrl_config.prediction_horizon,
            control_horizon=ctrl_config.control_horizon,
            dt=config.environment.dt,
            speed=config.environment.speed,
            max_angular_velocity=config.environment.max_angular_velocity,
            goal_weight=ctrl_config.goal_weight,
            obstacle_weight=ctrl_config.obstacle_weight,
            control_weight=ctrl_config.control_weight,
            obstacle_safety_margin=ctrl_config.obstacle_safety_margin,
            goal_tolerance=ctrl_config.goal_tolerance
        )
    if ctype == "mppi":
        return MPPIController(
            prediction_horizon=ctrl_config.prediction_horizon,
            num_samples=ctrl_config.num_samples,
            dt=config.environment.dt,
            speed=config.environment.speed,
            max_angular_velocity=config.environment.max_angular_velocity,
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
    if ctype in ("diffusion", "diffusion_wm"):
        base = _make_diffusion_controller(ctrl_config)
        if wm_config is not None and getattr(wm_config, "use_wm_prediction", False):
            return FilteredDiffusionController(base, wm_config)
        return base

    # default: simple controller
    return SimpleController(
        goal_attraction_gain=ctrl_config.goal_attraction_gain,
            obstacle_repulsion_gain=ctrl_config.obstacle_repulsion_gain,
            obstacle_influence_radius=ctrl_config.obstacle_influence_radius,
            max_angular_velocity=config.environment.max_angular_velocity,
            lookahead_distance=ctrl_config.lookahead_distance
    )
