"""
Base configuration class that combines all config modules
"""

from dataclasses import dataclass, fields, asdict
from .environment import EnvironmentConfig
from .controller import ControllerConfig
from .rendering import RenderingConfig
from .experiment import ExperimentConfig
from typing import Any, Dict, Optional

@dataclass
class Config:
    """Main configuration class that combines all config modules."""

    environment: EnvironmentConfig = None
    controller: ControllerConfig = None
    rendering: RenderingConfig = None
    experiment: ExperimentConfig = None
    wm_config: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Initialize default configs if not provided."""
        if self.environment is None:
            self.environment = EnvironmentConfig()
        if self.controller is None:
            self.controller = ControllerConfig()
        if self.rendering is None:
            self.rendering = RenderingConfig()
        if self.experiment is None:
            self.experiment = ExperimentConfig()
    
    @property
    def max_angular_velocity(self) -> float:
        """Get max_angular_velocity from environment (single source of truth)."""
        return self.environment.max_angular_velocity
    
    @max_angular_velocity.setter
    def max_angular_velocity(self, value: float):
        """Set max_angular_velocity in environment (single source of truth)."""
        self.environment.max_angular_velocity = value

    def save_to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        result = {}
        
        # Get all dataclass fields of this Config class
        for field in fields(self):
            field_value = getattr(self, field.name)
            
            if field_value is not None:
                # Use asdict for dataclass objects, which handles nested structures
                if hasattr(field_value, '__dataclass_fields__'):
                    result[field.name] = asdict(field_value)
                else:
                    result[field.name] = field_value
        
        return result
