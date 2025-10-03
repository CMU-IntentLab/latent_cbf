"""
Rendering configuration for DubinsEnv
"""

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class RenderingConfig:
    """Configuration for rendering and visualization."""
    
    # Colors (RGB tuples or color names)
    colors: Dict[str, str] = field(default_factory=lambda: {
        'background': 'white',
        'agent': 'blue',
        'goal': 'green',
        'obstacle': 'red',
        'trajectory': 'lightblue'
    })
    
    # Agent visual parameters
    agent_length_scale: float = 17.5  # Length multiplier for agent rendering
    agent_width_scale: float = 10.0   # Width multiplier for agent rendering
    
    # Anti-aliasing
    anti_aliasing_scale: int = 4
    
    # Video generation
    video_fps: float = 10.0
    video_quality: str = "medium"  # "low", "medium", "high"
