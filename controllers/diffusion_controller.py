"""
Diffusion Policy Controller for Dubins Car Environment

This controller uses a trained diffusion policy model to control the Dubins car.
The model takes current image and theta as input, generates 16 action steps,
and executes the first 8 before requerying.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Tuple
import cv2
from torchvision import transforms
from PIL import Image
import os
import sys
import hydra
import json
from pathlib import Path
from collections import defaultdict
from omegaconf import DictConfig, OmegaConf, ListConfig  # 👈 important import
from scipy.spatial.transform import Rotation as R

# Add diffusion4robotics to path
sys.path.append('/home/kensuke/diffusion4robotics')
from data4robotics import misc
from data4robotics.models.diffusion_unet import DiffusionUnetAgent
from configs import get_diffusion_config
import omegaconf.listconfig



def rmat_to_quat(rot_mat, degrees=False):
    quat = R.from_matrix(torch.Tensor(rot_mat)).as_quat()
    return quat


def vec_to_reorder_mat(vec):
    X = np.zeros((len(vec), len(vec)))
    for i in range(X.shape[0]):
        ind = int(abs(vec[i])) - 1
        X[i, ind] = np.sign(vec[i])
    return X

    



class DiffusionController:
    """
    Diffusion Policy Controller for Dubins Car Environment
    
    Args:
        checkpoint_path: Path to trained diffusion model checkpoint
        config_path: Path to model configuration directory
        device: Device to run model on ('cuda' or 'cpu')
        action_chunk_size: Number of actions to execute before requerying (default: 8)
        total_chunk_size: Total actions generated per query (default: 16)
        eval_diffusion_steps: Number of diffusion steps for inference
    """
    
    def __init__(self, 
                 checkpoint_path: str,
                 config_path: str,
                 device: str = 'cuda',
                 action_chunk_size: int = 8,
                 total_chunk_size: int = 16,
                 eval_diffusion_steps: int = 16):
        
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.action_chunk_size = action_chunk_size
        self.total_chunk_size = total_chunk_size
        self.eval_diffusion_steps = eval_diffusion_steps
        
        # Load configuration and model using RealPolicy approach
        self._setup_model_and_configs(checkpoint_path, config_path)
        
        # Initialize action buffer and state tracking
        self.action_buffer = []
        self.step_count = 0
        self.prev_img = defaultdict(lambda: None)
        self.prev_obs = None
        
        print(f"Diffusion Controller initialized:")
        print(f"  - Device: {self.device}")
        print(f"  - Action chunk size: {action_chunk_size}")
        print(f"  - Total chunk size: {total_chunk_size}")
        print(f"  - Eval diffusion steps: {eval_diffusion_steps}")
        print(f"  - Model: {self.model_name}")
    
    def _setup_model_and_configs(self, checkpoint_path: str, config_path: str):
        """Setup model and configurations using RealPolicy approach"""
        # Load configuration files
        agent_yaml_path = Path(config_path, "agent_config.yaml")
        exp_config_path = Path(config_path, "exp_config.yaml")
        obs_config_path = Path(config_path, "obs_config.yaml")
        ob_dict_path = Path(config_path, "ob_norm.json")
        ac_dict_path = Path(config_path, "ac_norm.json")
        
        # Load normalization parameters
        with open(ob_dict_path, 'r') as f:
            data = json.load(f)
            self.ob_max = np.array(data['maximum'])
            self.ob_min = np.array(data['minimum'])
        
        with open(ac_dict_path, 'r') as f:
            data = json.load(f)
            self.ac_max = np.array(data['maximum'])
            self.ac_min = np.array(data['minimum'])
        
        # Load configurations
        agent_config = OmegaConf.load(agent_yaml_path)
        exp_config = OmegaConf.load(exp_config_path)
        obs_config = OmegaConf.load(obs_config_path)
        
        # Load model
        self.agent = hydra.utils.instantiate(agent_config)
        
        # Determine model name and load checkpoint
        if hasattr(exp_config.params, 'exp_name') and exp_config.params.exp_name is not None:
            model_name = exp_config.params.exp_name
        else:
            model_name = exp_config.exp_name
        
        load_dict = torch.load(Path(checkpoint_path), map_location=self.device)
        self.agent.load_state_dict(load_dict["model"])
        self.agent = self.agent.eval().to(self.device)
        
        # Load transforms
        self.transform = hydra.utils.instantiate(obs_config["transform"])
        
        # Store important parameters
        self.model_name = model_name
        self.img_chunk = exp_config.params.img_chunk if hasattr(exp_config.params, 'img_chunk') else 1
        self.obs_keys = ['state']  # For Dubins: we use 'state' key containing theta
        
        print(f"Model loaded: {model_name}")
        print(f"Image chunk: {self.img_chunk}")
        print(f"State normalization: min={self.ob_min}, max={self.ob_max}")
        print(f"Action normalization: min={self.ac_min}, max={self.ac_max}")
    
    def _proc_image(self, img):
        """Process image for the model"""
        # Assume input is RGB image (H, W, 3) with values 0-255
        rgb_img = img[:, :, :3]
        rgb_img = torch.from_numpy(rgb_img).float().permute((2, 0, 1)) / 255
        return self.transform(rgb_img)[None].to(self.device)
    
    def _get_images_and_states(self, obs):
        """
        Return images and states given observations (adapted from RealPolicy)
        """
        images = {}
        state = np.empty(0)
        
        # Extract state information (for Dubins: theta from state key)
        for key in self.obs_keys:
            if key in obs:
                state = np.append(state, obs[key])
        
        # Process image (for Dubins: single camera 'cam_0')
        if 'cam0' in obs:
            img = obs['cam0']
            cur_img = self._proc_image(img)
            
            if self.prev_img['cam0'] is None:
                self.prev_img['cam0'] = torch.clone(cur_img)
            
            if self.img_chunk == 2:
                images['cam0'] = torch.cat((cur_img, self.prev_img['cam0']), dim=0).unsqueeze(0)
            else:
                images['cam0'] = cur_img
            
            self.prev_img['cam0'] = cur_img
        
        # Normalize state
        state = (state - self.ob_min) / (self.ob_max - self.ob_min)
        state = state * 2 - 1
        
        if self.prev_obs is None:
            self.prev_obs = state
        
        self.prev_obs = state
        state = torch.from_numpy(state.astype(np.float32))[None].to(self.device)
        
        return images, state
    
    
    def compute_action(self, info: Dict[str, Any], obstacles: list, observation: np.ndarray = None) -> float:
        """
        Compute action using diffusion policy (adapted from RealPolicy)
        
        Args:
            info: Environment info dict containing agent state information
            obstacles: List of obstacles (not used by diffusion policy)
            observation: Current RGB image observation (optional, will be fetched if not provided)
            
        Returns:
            Angular velocity action for the Dubins car
        """
        # If observation is not provided, we need to get it from the environment
        # For now, we'll use a placeholder or try to get it from info
        if observation is None:
            # Try to get observation from info dict, or use a placeholder
            if 'observation' in info:
                img_obs = info['observation']
            else:
                # Create a placeholder image - this should be replaced with actual observation
                print("Warning: No observation provided, using placeholder")
                img_obs = np.zeros((128, 128, 3), dtype=np.uint8)
        else:
            img_obs = observation
        
        # Prepare observation in the format expected by the model
        obs = {
            'cam0': img_obs,  # RGB image (128, 128, 3)
            'state': np.array([info['agent_orientation']])  # Extract theta only [theta]
        }
        
        # Get processed images and states
        images, state = self._get_images_and_states(obs)
        # Get action from buffer or generate new ones
        if self.action_buffer:
            ac = self.action_buffer.pop(0)

        else:
            # Generate new action chunk
            with torch.no_grad():
                action = self.agent.get_actions(images, state)
            # Extract actions and convert to numpy
            ac_list = action[0].cpu().numpy().astype(np.float32)[:self.total_chunk_size]
            self.action_buffer = list(ac_list)
            ac = self.action_buffer.pop(0)
        
        # Denormalize action: (ac + 1) / 2 * (max - min) + min
        ac = (ac + 1) / 2
        ac = ac * (self.ac_max - self.ac_min) + self.ac_min
        
        self.step_count += 1
        
        # Ensure we return a scalar for Dubins car (angular velocity)
        if isinstance(ac, np.ndarray):
            ac = ac.item()
        
        # Clip to valid range
        max_angular_velocity = 2.0  # From your config
        ac = np.clip(ac, -max_angular_velocity, max_angular_velocity)
        
        return float(ac)
    
    
    def reset(self):
        """Reset the controller state"""
        self.action_buffer = []
        self.step_count = 0
        self.prev_img = defaultdict(lambda: None)
        self.prev_obs = None
    
    def get_info(self) -> Dict[str, Any]:
        """Get controller information"""
        return {
            'controller_type': 'diffusion',
            'action_buffer_size': len(self.action_buffer),
            'step_count': self.step_count,
            'device': str(self.device),
            'action_chunk_size': self.action_chunk_size,
            'total_chunk_size': self.total_chunk_size
        }


def create_diffusion_controller_from_config(config) -> DiffusionController:
    """
    Create diffusion controller from config object
    
    Args:
        config: Configuration object containing diffusion controller parameters
        
    Returns:
        Initialized DiffusionController
    """
    return DiffusionController(
        checkpoint_path=config.controller.checkpoint_path,
        config_path=config.controller.config_path,
        device=config.controller.device,
        action_chunk_size=config.controller.action_chunk_size,
        total_chunk_size=config.controller.total_chunk_size,
        eval_diffusion_steps=config.controller.eval_diffusion_steps
    )


# Example usage and testing
if __name__ == "__main__":
    # Example usage
    config = get_diffusion_config()
    
    try:
        # Create controller
        controller = create_diffusion_controller_from_config(config)
        print("Diffusion controller created successfully!")
        
        # Test with dummy data
        dummy_info = {
            'observation': np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8),
            'state': np.array([0.0, 0.0, 0.0])
        }
        
        action = controller.compute_action(dummy_info, [])
        print(f"Generated action: {action}")
        
    except Exception as e:
        print(f"Error creating diffusion controller: {e}")
        print("Make sure the checkpoint and config paths are correct")
