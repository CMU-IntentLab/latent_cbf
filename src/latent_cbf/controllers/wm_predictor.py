"""
World Model Predictor for Safety Margin Evaluation

This module provides a wrapper around the trained Dreamer world model to predict
safety margins for action sequences. It loads a trained world model checkpoint
and provides methods to evaluate margin heads on predicted latent states.
"""

import torch
import torch.nn as nn
import numpy as np
import gym
from typing import Dict, List, Tuple, Optional, Any
import os
from pathlib import Path

from dreamerv3_torch.models import WorldModel
from configs.dreamer_conf import DreamerConfig


class WMPredictor:
    """
    World Model Predictor for evaluating safety margins on action sequences.
    
    This class loads a trained Dreamer world model and provides methods to:
    1. Encode current observations to latent states
    2. Rollout action sequences using the world model
    3. Evaluate margin heads on predicted latent states
    """
    
    def __init__(self, checkpoint_path: str, config: DreamerConfig, device: str = 'cuda'):
        """
        Initialize the World Model Predictor.
        
        Args:
            checkpoint_path: Path to the trained world model checkpoint
            config: DreamerConfig containing model architecture parameters
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.config = config
        self.history_length = config.wm_history_length
        
        # Initialize history buffers
        self.observation_history = []
        self.action_history = []
        self.state_history = []
        
        # Create observation and action spaces (matching dreamer_offline.py)
        self.action_space = gym.spaces.Box(
            low=-config.turnRate, high=config.turnRate, shape=(1,), dtype=np.float32
        )
    
        # dreamer wm takes in tuples of (o_t+1, a_t)
        self.action_history.append(self.action_space.sample()*0)

        # Define observation space components
        low = np.array([config.x_min, config.y_min, -np.pi])
        high = np.array([config.x_max, config.y_max, np.pi])
        midpoint = (low + high) / 2.0
        interval = high - low
        
        gt_observation_space = gym.spaces.Box(
            np.float32(midpoint - interval/2),
            np.float32(midpoint + interval/2),
        )
        
        image_size = config.size[0] if hasattr(config, 'size') else 128
        image_observation_space = gym.spaces.Box(
            low=0, high=255, shape=(image_size, image_size, 3), dtype=np.uint8
        )
        
        obs_observation_space = gym.spaces.Box(
            low=-1, high=1, shape=(2,), dtype=np.float32
        )
        
        self.observation_space = gym.spaces.Dict({
            'state': gt_observation_space,
            'obs_state': obs_observation_space,
            'image': image_observation_space
        })
        
        # Set number of actions
        config.num_actions = self.action_space.shape[0]
        
        # Initialize world model
        self.wm = WorldModel(self.observation_space, self.action_space, 0, config)
        self.wm.to(self.device)
        self.wm.eval()
        
        # Load checkpoint
        self._load_checkpoint(checkpoint_path)
        
        # Disable gradients for inference
        for param in self.wm.parameters():
            param.requires_grad = False
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load the world model from checkpoint."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"Loading world model from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        # Load the world model state dict
        if 'agent_state_dict' in checkpoint:
            # Full agent checkpoint
            agent_state = checkpoint['agent_state_dict']
            # Extract world model state dict (keys starting with '_wm.')
            wm_state = {k[14:]: v for k, v in agent_state.items() if k.startswith('_wm.')}
            self.wm.load_state_dict(wm_state)
        else:
            # Direct world model checkpoint
            self.wm.load_state_dict(checkpoint)
        
        print("World model loaded successfully")
    
    
    def add_to_history(self, image: np.ndarray, state: np.ndarray, action: float):
        """
        Add observation, state, and action to history buffer.
        
        Args:
            image: Current RGB image observation (H, W, 3)
            state: Current state [x, y, theta] or [theta]
            action: Current action
        """
        print('ADDING TO HISTORY')
        print('state', state)
        self.observation_history.append(image.copy())
        self.state_history.append(state.copy())
        self.action_history.append(action[0])
        # Maintain history length
        if len(self.observation_history) > self.history_length:
            self.observation_history.pop(0)
            self.state_history.pop(0)
        if len(self.action_history) > self.history_length:
            self.action_history.pop(0)
    
    def has_sufficient_history(self) -> bool:
        """Check if we have sufficient history for prediction."""
        return len(self.observation_history) >= self.history_length
    
    def reset_history(self):
        """Reset the history buffers."""
        self.observation_history = []
        self.action_history = []
        self.state_history = []
        self.action_history.append(self.action_space.sample()*0)
        
    def predict_margins(self, actions: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Predict safety margins for a sequence of actions using history.
        
        Args:
            image: Current RGB image observation (H, W, 3)
            state: Current state [x, y, theta] or [theta]
            actions: Action sequence to evaluate (T,)
            
        Returns:
            Dictionary containing:
            - 'margin_gp': Margin GP predictions for each timestep (T,)
            - 'margin_nogp': Margin NoGP predictions for each timestep (T,)
            - 'latent_states': Predicted latent states (T, feature_dim)
            - 'history_used': Number of history timesteps used
        """
        # Check if we have sufficient history
        if not self.has_sufficient_history():
            return {
                'margin_gp': np.zeros(len(actions)),
                'margin_nogp': np.zeros(len(actions)),
                'latent_states': np.zeros((len(actions), 1)),
                'history_used': 0
            }
        
        with torch.no_grad():
            # Prepare historical sequence
            history_length = len(self.observation_history)
            # Prepare batch data for entire history sequence
            
            # Collect all observations and actions first
            all_images = self.observation_history
            all_states = self.state_history
            print('all_states', all_states)
            all_actions = self.action_history  # Dummy action for current state            

            
            # Process all observations in batch
            batch_images = torch.stack([torch.from_numpy(img).float() / 255.0 
                                      for img in all_images], dim=0).to(self.device)  # (T, C, H, W)
            
            batch_states = torch.stack([torch.from_numpy(s.astype(np.float32)) 
                                      for s in all_states], dim=0).to(self.device)  # (T, 3)
            
            # Create obs_state (cos/sin of theta) for all states
            thetas = batch_states[:, 2] if batch_states.shape[1] > 2 else batch_states[:, 0]
            batch_obs_state = torch.stack([
                torch.cos(thetas), 
                torch.sin(thetas)
            ], dim=1).to(self.device)  # (T, 2)
            
            # Create batch observation dictionary
            batch_obs = {
                'image': batch_images,
                'state': batch_states,
                'obs_state': batch_obs_state
            }
            
            # Prepare historical actions tensor
            historical_actions_tensor = torch.tensor(np.array(all_actions), dtype=torch.float32).unsqueeze(-1).to(self.device)  # (T, 1)

            # Prepare is_first flags (all False except first)
            is_first = torch.zeros(history_length, 1, dtype=torch.bool, device=self.device)
            is_first[0] = True
            is_first = is_first.unsqueeze(0)
            
            for k, v in batch_obs.items():
                batch_obs[k] = v.unsqueeze(0)
            historical_actions_tensor = historical_actions_tensor.unsqueeze(0)

            # Encode all observations in batch
            embed = self.wm.encoder(batch_obs)


            # Get initial state from historical sequence
            post, prior = self.wm.dynamics.observe(embed, historical_actions_tensor, is_first)
            # Use the last posterior state as initial state for imagination
            initial_state = {k: v[0, -1:] for k, v in post.items()}
            
            # Prepare future actions
            action_data = self._prepare_actions(actions)
        
            # Rollout action sequence
            predicted_states = self.wm.dynamics.imagine_with_action(action_data, initial_state)
            # Get features from predicted states
            features = self.wm.dynamics.get_feat(predicted_states)
            
            # Evaluate margin heads
            margin_gp_preds = self.wm.heads['margin_gp'](features)
            margin_nogp_preds = self.wm.heads['margin_nogp'](features)
            # Convert to numpy
            results = {
                'margin_gp': margin_gp_preds.cpu().numpy().squeeze(),  # (T,)
                'margin_nogp': margin_nogp_preds.cpu().numpy().squeeze(),  # (T,)
                'latent_states': features.cpu().numpy().squeeze(),  # (T, feature_dim)
                'history_used': history_length
            }
            
            return results
    
