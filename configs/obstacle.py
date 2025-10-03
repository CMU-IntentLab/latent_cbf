"""
Obstacle configuration for DubinsEnv
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class ObstacleConfig:
    """Configuration for a single circular obstacle."""
    x: float
    y: float
    radius: float
    
    def to_tuple(self) -> Tuple[float, float, float]:
        """Convert to tuple format expected by environment."""
        return (self.x, self.y, self.radius)
