"""
Environment configuration for DubinsEnv
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np
from .obstacle import ObstacleConfig


@dataclass
class EnvironmentConfig:
    """Configuration for the Dubins car environment."""
    
    # Image and world settings
    image_size: Tuple[int, int] = (128, 128)
    world_bounds: Tuple[float, float, float, float] = (-1.5, 1.5, -1.5, 1.5)  # (x_min, x_max, y_min, y_max)
    
    # Dynamics parameters
    max_angular_velocity: float = 2.
    speed: float = 1.0
    dt: float = 0.05
    
    # Episode settings
    max_episode_steps: int = 200
    
    # Goal settings
    goal_position: Tuple[float, float] = (1.3, 0.0)
    goal_radius: float = 0.05
    
    # Agent settings
    collision_radius: float = 0.0
    
    # Reset distribution settings
    reset_x_range: Tuple[float, float] = (-1.5, 0.0)  # (min_x, max_x) for random reset
    reset_y_range: Tuple[float, float] = (-1.5, 1.5)  # (min_y, max_y) for random reset
    reset_theta_range: Tuple[float, float] = (-np.pi/3, np.pi/3)  # (min_theta, max_theta) for random reset
    
    # Deterministic reset (overrides random if set)
    deterministic_start: bool = False
    initial_position: Optional[Tuple[float, float, float]] = None  # (x, y, theta)
    
    # Rendering settings
    render_mode: Optional[str] = "rgb_array"  # "human", "rgb_array", or None
    
    # Obstacles
    obstacles: List[ObstacleConfig] = field(default_factory=lambda: [
        ObstacleConfig(x=0.25, y=0.65, radius=0.5),
        ObstacleConfig(x=0.25, y=-0.65, radius=0.5)
    ])
    
    def get_obstacles_list(self) -> List[Tuple[float, float, float]]:
        """Get obstacles in the format expected by the environment."""
        return [obs.to_tuple() for obs in self.obstacles]
    
    @property
    def x_min(self) -> float:
        return self.world_bounds[0]
    
    @property
    def x_max(self) -> float:
        return self.world_bounds[1]
    
    @property
    def y_min(self) -> float:
        return self.world_bounds[2]
    
    @property
    def y_max(self) -> float:
        return self.world_bounds[3]
    
    def get_reset_bounds(self) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        """Get reset distribution bounds as (x_range, y_range, theta_range)."""
        return (self.reset_x_range, self.reset_y_range, self.reset_theta_range)
    
    def set_reset_bounds(self, x_range: Tuple[float, float], y_range: Tuple[float, float], theta_range: Tuple[float, float]):
        """Set reset distribution bounds."""
        self.reset_x_range = x_range
        self.reset_y_range = y_range
        self.reset_theta_range = theta_range
    
    def set_deterministic_reset(self, x: float, y: float, theta: float):
        """Set deterministic initial position and enable deterministic mode."""
        self.deterministic_start = True
        self.initial_position = (x, y, theta)
    
    def set_random_reset(self):
        """Disable deterministic mode to use random reset distribution."""
        self.deterministic_start = False
        self.initial_position = None
