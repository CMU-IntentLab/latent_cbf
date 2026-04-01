"""
Experiment configuration for DubinsEnv
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class ExperimentConfig:
    """Configuration for experiments and data collection."""
    
    # Output settings
    output_dir: str = "results"
    video_filename: str = "trajectory.mp4"
    data_filename: str = "trajectories.h5"
    
    # Experiment settings
    n_episodes: int = 10
    max_episode_length: int = 200
    save_images: bool = False
    save_video: bool = True
    
    # Debugging settings
    deterministic_start: bool = False
    initial_position: Optional[tuple] = None
    seed: Optional[int] = None
    
    # Logging settings
    verbose: bool = True
    log_level: str = "INFO"
    
    # Additional metadata
    experiment_name: str = "dubins_experiment"
    description: str = ""
    tags: Dict[str, Any] = field(default_factory=dict)
