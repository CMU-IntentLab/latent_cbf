"""
Config package for DubinsEnv and controllers.

Modules cover environment, controller (MPPI / diffusion), rendering, experiment
settings, Dreamer training config, and named presets.
"""

from .environment import EnvironmentConfig
from .obstacle import ObstacleConfig
from .controller import ControllerConfig
from .rendering import RenderingConfig
from .experiment import ExperimentConfig
from .base import Config
from .dreamer_conf import DreamerConfig

from .presets import (
    get_default_config,
    get_no_obstacles_config,
    get_debug_config,
    get_narrow_gap_config,
    get_mppi_config,
    get_diffusion_config,
    get_diffusion_wm_config,
    get_diffusion_collection_config,
)

__all__ = [
    "EnvironmentConfig",
    "ObstacleConfig",
    "ControllerConfig",
    "RenderingConfig",
    "ExperimentConfig",
    "Config",
    "get_default_config",
    "get_no_obstacles_config",
    "get_debug_config",
    "get_narrow_gap_config",
    "get_mppi_config",
    "get_diffusion_config",
    "get_diffusion_wm_config",
    "get_diffusion_collection_config",
    "DreamerConfig",
]
