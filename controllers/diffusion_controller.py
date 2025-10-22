"""
Diffusion Policy Controller for Dubins Car Environment

This controller uses a trained diffusion policy model to control the Dubins car.
The model takes current image and theta as input, generates 16 action steps,
and executes the first 8 before requerying.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
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
import pathlib
dreamer = os.path.abspath(os.path.join(os.path.dirname(__file__), '../dreamerv3-torch'))
sys.path.append(dreamer)
sys.path.append(str(pathlib.Path(__file__).parent))
from models import WorldModel
import gym
from PyHJ.utils.net.common import Net
from PyHJ.utils.net.continuous import Actor, Critic
from gymnasium import spaces
from PyHJ.policy import avoid_DDPGPolicy_annealing as DDPGPolicy
from PyHJ.data import Batch
from PyHJ.exploration import GaussianNoise

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
                 eval_diffusion_steps: int = 16,
                 wm_predictor=None):
        
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
        
        # Initialize prediction logging
        self.prediction_log = []
        self.current_action_chunk = None
        self.current_predictions = None
        
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
            
            images['cam0'] = cur_img
                    
        # Normalize state
        state = (state - self.ob_min) / (self.ob_max - self.ob_min)
        state = state * 2 - 1
        
        state = torch.from_numpy(state.astype(np.float32))[None].to(self.device)
        
        return images, state
    
    
    def compute_action(self, info: Dict[str, Any], observation: np.ndarray = None) -> float:
        """
        Compute action using diffusion policy (adapted from RealPolicy)
        
        Args:
            info: Environment info dict containing agent state information
            observation: Current RGB image observation (optional, will be fetched if not provided)
            
        Returns:
            Angular velocity action for the Dubins car
        """
        
        obs = {
            'cam0': observation,  # RGB image (128, 128, 3)
            'state': np.array([info['agent_orientation']])  # Extract theta only [theta]
        }
        images, state = self._get_images_and_states(obs)
        # Get action from buffer or generate new ones
        if self.action_buffer:
            ac = self.action_buffer.pop(0)
        else:
            # Generate new action chunk
            with torch.no_grad():
                action = self.agent.get_actions(images, state)
            # Extract actions and convert to numpy
            ac_list = action[0].cpu().numpy().astype(np.float32)[:self.action_chunk_size]
            # Denormalize action: (ac + 1) / 2 * (max - min) + min
            ac_list = (ac_list + 1) / 2
            ac_list = ac_list * (self.ac_max - self.ac_min) + self.ac_min
            self.action_buffer = list(ac_list)    
            ac = self.action_buffer.pop(0)
        
        
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


class FilteredDiffusionController:
    def __init__(self, controller, wm_config):
        self.controller = controller
        self.wm_config = wm_config
        self.observation_history = []
        self.state_history = []
        self.action_history = []
        self.device = self.controller.device
        self.init_wm()
        self.init_filter()

        
        
    
    def init_filter(self):
        config = self.wm_config
        config.state_shape = (1,1,544,)
        config.action_shape = (1,)
        config.max_action = 1

        actor_activation = torch.nn.ReLU
        critic_activation = torch.nn.ReLU
        critic_net = Net(
            config.state_shape,
            config.action_shape,
            hidden_sizes=config.critic_net,
            activation=critic_activation,
            concat=True,
            device=config.device
        )
        critic = Critic(critic_net, device=config.device).to(config.device)
        critic_optim = torch.optim.AdamW(critic.parameters(), lr=config.critic_lr, weight_decay=config.weight_decay_pyhj)
        env_action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32) # joint action space
        actor_net = Net(config.state_shape, hidden_sizes=config.control_net, activation=actor_activation, device=config.device)
        actor = Actor(
            actor_net, config.action_shape, max_action=config.max_action, device=config.device
        ).to(config.device)
        actor_optim = torch.optim.AdamW(actor.parameters(), lr=config.actor_lr)

        policy = DDPGPolicy(
        critic,
        critic_optim,
        tau=config.tau,
        gamma=config.gamma_pyhj,
        exploration_noise=GaussianNoise(sigma=config.exploration_noise),
        reward_normalization=config.rew_norm,
        estimation_step=config.n_step,
        action_space=env_action_space,
        actor=actor,
        actor_optim=actor_optim,
        actor_gradient_steps=config.actor_gradient_steps,
        )
        if config.no_gp:
            print("Loading no GP policy")
            policy_ckpt = torch.load(config.filter_directory_nogp)
        else:
            print("Loading GP policy")
            policy_ckpt = torch.load(config.filter_directory_gp)
        policy.load_state_dict(policy_ckpt)
        self.policy = policy
    
    def init_wm(self):
        config = self.wm_config
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
        
        
        print('wm', self.wm)
        # Load checkpoint
        self._load_checkpoint(self.wm_config.wm_checkpoint_path)
        print("World Model initialized successfully")
        
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

    def eval_Q(self, state, action):
        tmp_obs = np.array(state)
        tmp_batch = Batch(obs = tmp_obs, info = Batch())
        tmp = self.policy.critic_old(tmp_batch.obs, action)
        return tmp.cpu().detach().numpy().flatten()
    
    def eval_V(self, state):
        tmp_obs = np.array(state)
        tmp_batch = Batch(obs = tmp_obs, info = Batch())
        tmp = self.policy.critic_old(tmp_batch.obs, self.policy(tmp_batch, model="actor_old").act)
        return tmp.cpu().detach().numpy().flatten()
    
    def eval_policy(self, state):
        tmp_obs = np.array(state)
        tmp_batch = Batch(obs = tmp_obs, info = Batch())
        return self.policy(tmp_batch, model="actor_old").act
        
    def compute_action(self, info: Dict[str, Any], observation: np.ndarray = None) -> float:        
        raw_ac = self.controller.compute_action(info, observation)
            
        self.action_history.append(raw_ac)
        self.observation_history.append(observation)
        self.state_history.append(info['agent_orientation'])
        H = 5
        if len(self.action_history) > H+1:
            # (o_{t+1}, a_t, r_t)
            obs_chunk = np.array(self.observation_history[-H:])
            states = np.array(self.state_history[-H:])
            ac_chunk = np.array(self.action_history[-H-1:-1])
            state_chunk = np.concatenate([np.cos(states[:, None]), np.sin(states[:, None])], axis=-1)

            
            
            is_first = torch.zeros(H)
            is_first[0] = 1
            obs_batch = {
                'image': obs_chunk[None, :],
                'obs_state': state_chunk[None, :],
                'action': ac_chunk[None, :, None],
                'is_first': is_first[None, :, None],
                'is_terminal': torch.zeros(1, H, 1)
            }
            data = self.wm.preprocess(obs_batch)
            embed = self.wm.encoder(data)
            states, _ = self.wm.dynamics.observe(embed, data["action"], data["is_first"])
            feat = self.wm.dynamics.get_feat(states)
            feat_latest= feat[:,-1].detach().cpu().numpy()
            acs =np.array([[raw_ac]])/self.controller.ac_max
            print(self.eval_V(feat_latest))

            if self.wm_config.filter_mode == 'cbf':
                B = 25
                sample_acs = np.linspace(-1, 1, B)[:, None]
                feat_tiled = np.tile(feat_latest, (B+1, 1))

                sample_acs = np.concatenate([acs, sample_acs], axis=0)
                qvals = self.eval_Q(feat_tiled, sample_acs)

                dec = qvals/qvals.max()
                valid_idx = np.where(dec > self.wm_config.cbf_gamma)[0]

                
                if 0 in valid_idx:
                    safe_ac = sample_acs[0]
                else:
                    print('filtering')
                    valid_acs = sample_acs[valid_idx]
                    closest_idx = np.argmin(np.abs(valid_acs - acs[0]))
                    safe_ac = valid_acs[closest_idx]
                    print('filtered ac', safe_ac, 'from', acs[0])
                ac = safe_ac * self.controller.ac_max

            elif self.wm_config.filter_mode == 'lr':
                qval = self.eval_Q(feat_latest, acs)
                if qval < self.wm_config.lr_thresh:
                    safe_ac = self.eval_policy(feat_latest).detach().cpu().numpy().flatten()
                    print('filtered ac', safe_ac, 'from', acs[0])
                else:
                    safe_ac = acs[0]
                ac = safe_ac * self.controller.ac_max
            else:
                ac = raw_ac
            
        else:
            ac = raw_ac
        return ac

    def reset(self):
        self.observation_history = []
        self.state_history = []
        self.action_history = []
        self.controller.reset()
    def get_info(self) -> Dict[str, Any]:
        return self.controller.get_info()
    




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
