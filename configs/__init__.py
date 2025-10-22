"""
Config package for DubinsEnv and Controller

This package provides modular configuration classes for:
- Environment parameters
- Controller parameters  
- Obstacle specifications
- Rendering settings
- Experiment settings
- Preset configurations
"""

from .environment import EnvironmentConfig
from .obstacle import ObstacleConfig
from .controller import ControllerConfig
from .rendering import RenderingConfig
from .experiment import ExperimentConfig
from .base import Config
from .dreamer_conf import DreamerConfig

# Import presets
from .presets import (
    get_default_config,
    get_no_obstacles_config,
    get_debug_config,
    get_mpc_config,
    get_random_config,
    get_mppi_config,
    get_diffusion_config,
    get_diffusion_wm_config,
)

__all__ = [
    # Core config classes
    'EnvironmentConfig',
    'ObstacleConfig', 
    'ControllerConfig',
    'RenderingConfig',
    'ExperimentConfig',
    'Config',
    # Preset functions
    'get_default_config',
    'get_no_obstacles_config',
    'get_narrow_gap_config',
    'get_debug_config',
    'get_mpc_config',
    'get_random_config',
    'get_mppi_config',
    'get_diffusion_config',
    'get_diffusion_wm_config',

    'DreamerConfig'
]
