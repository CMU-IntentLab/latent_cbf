"""
Random Controller for Dubins Car Environment

This controller generates random actions for data collection purposes.
"""

import numpy as np
from typing import Dict, Any, List


class RandomController:
    """
    Random controller that generates random angular velocities for data collection.
    
    This is useful for collecting diverse trajectory data without any specific
    control strategy, which can be valuable for training world models or
    other learning-based approaches.
    """
    
    def __init__(self, max_angular_velocity: float = 2.0, seed: int = None):
        """
        Initialize the random controller.
        
        Args:
            max_angular_velocity: Maximum angular velocity for the Dubins car
            seed: Random seed for reproducible behavior (optional)
        """
        self.max_angular_velocity = max_angular_velocity
        if seed is not None:
            np.random.seed(seed)
    
    def compute_action(self, info: Dict[str, Any], obstacles: List[Dict], observation: np.ndarray = None) -> float:
        """
        Compute a random action.
        
        Args:
            info: Environment info dict (not used by random controller)
            obstacles: List of obstacles (not used by random controller)
            observation: Current observation (not used by random controller)
            
        Returns:
            Random angular velocity action for the Dubins car
        """
        # Generate random angular velocity in [-max_angular_velocity, max_angular_velocity]
        action = np.random.uniform(-self.max_angular_velocity, self.max_angular_velocity)
        return float(action)
    
    def reset(self):
        """Reset the controller state (no state to reset for random controller)."""
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """Get controller information."""
        return {
            'controller_type': 'random',
            'max_angular_velocity': self.max_angular_velocity
        }