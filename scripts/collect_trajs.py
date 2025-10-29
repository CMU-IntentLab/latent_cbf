"""
Trajectory Collection Script

Collects N trajectories from the Dubins car environment using various controllers
and saves them to HDF5 files for analysis, visualization, or training purposes.
"""

import sys
import os
import numpy as np
import h5py
import torch
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import argparse

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our modules
from configs import (Config, get_default_config, get_mpc_config, get_random_config, get_no_obstacles_config, 
                get_debug_config, get_mppi_config, get_diffusion_config, get_diffusion_wm_config)
from configs.dreamer_conf import DreamerConfig
from scripts.run_experiment import create_env_from_config, create_controller_from_config
from dubins_env import DubinsEnv
from controllers.wm_predictor import WMPredictor
from controllers import FilteredDiffusionController

class TrajectoryCollector:
    """
    Collects trajectories from the Dubins car environment.
    """
    
    def __init__(self, config: Config, output_dir: str = "data", wm_config: DreamerConfig = None):
        """
        Initialize trajectory collector.
        
        Args:
            config: Configuration for environment and controller
            output_dir: Directory to save trajectory files
            wm_config: DreamerConfig for world model prediction (optional)
        """
        self.config = config
        self.output_dir = output_dir
        self.wm_config = wm_config
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize environment and controller
        self.env = create_env_from_config(config)
        controller = create_controller_from_config(config)
        # Initialize world model predictor if enabled
        if wm_config and wm_config.use_wm_prediction:
            self.controller = FilteredDiffusionController(controller, wm_config)
            print("FilteredDiffusionController initialized successfully")
        else:
            self.controller = controller
            print("Controller initialized successfully")
        # Trajectory storage
        self.trajectories = []
        self.metadata = {
            'config': config.save_to_dict(),
            'collection_timestamp': datetime.now().isoformat(),
            'environment_info': {
                'world_bounds': config.environment.world_bounds,
                'goal_position': config.environment.goal_position,
                'obstacles': config.environment.get_obstacles_list(),
                'dt': config.environment.dt,
                'speed': config.environment.speed
            },
            'controller_info': {
                'type': config.controller.controller_type,
                'max_angular_velocity': config.environment.max_angular_velocity
            }
        }
    
    def collect_single_trajectory(self, episode_idx: int, 
                                 seed: Optional[int] = None,
                                 initial_state: Optional[List[float]] = None,
                                 max_steps: Optional[int] = None,
                                 save_images: bool = False) -> Dict[str, Any]:
        """
        Collect a single trajectory.
        
        Args:
            episode_idx: Episode index for identification
            seed: Random seed for episode
            initial_state: Optional initial state [x, y, theta]
            max_steps: Maximum steps for episode (uses config default if None)
            save_images: Whether to save image observations
            
        Returns:
            Trajectory data dictionary
        """
        # Set up episode parameters
        if max_steps is None:
            max_steps = self.config.experiment.max_episode_length
        
        # Reset environment
        reset_options = {}
        if initial_state is not None:
            reset_options['initial_state'] = initial_state
        
        long_traj = False

        while not long_traj: 
            obs, info = self.env.reset( options=reset_options if reset_options else None)

            # Reset controller if it has a reset method (MPC)
            if hasattr(self.controller, 'reset'):
                self.controller.reset()
            
            # Data storage for this trajectory
            states = [np.array([info['agent_position'][0], info['agent_position'][1], info['agent_orientation']])]
            actions = []
            rewards = []
            collisions = []
            observations = [obs] if save_images else []
            infos = [info.copy()]
            
            # Run episode
            for step in range(max_steps):
                # Compute action
                if self.config.controller.controller_type == "mpc":
                    action = self.controller.compute_action(
                        info, 
                        self.config.environment.get_obstacles_list(), 
                        np.array(self.config.environment.goal_position)
                    )
                elif self.config.controller.controller_type == "mppi":
                    action = self.controller.compute_action(
                        info, 
                        self.config.environment.get_obstacles_list(), 
                        np.array(self.env.goal_position)
                    )
                elif self.config.controller.controller_type == "diffusion":
                    # For diffusion controller, pass the current observation
                    print("Computing action for diffusion controller")
                    action = self.controller.compute_action(
                        info, 
                        obs  # Pass the current observation
                    )
                else:
                    action = self.controller.compute_action(info, self.config.environment.get_obstacles_list())
                
                # Store action
                actions.append(float(action))
                
                # Step environment
                obs, reward, terminated, truncated, info = self.env.step(action)
                
                # Store data
                states.append(np.array([info['agent_position'][0], info['agent_position'][1], info['agent_orientation']]))
                rewards.append(float(reward))
                infos.append(info.copy())
                collisions.append(info.get('collision', False))
                
                if save_images:
                    observations.append(obs)
                
                # Check termination
                if terminated or truncated:
                    break
                #exit()
            
            # Compile trajectory data
            trajectory_data = {
                'episode_idx': episode_idx,
                'seed': seed,
                'initial_state': states[0],
                'states': np.array(states),  # Shape: (T+1, 3) where T is number of steps
                'actions': np.array(actions),  # Shape: (T,)
                'rewards': np.array(rewards),  # Shape: (T,)
                'observations': np.array(observations) if save_images else None,  # Shape: (T+1, H, W, 3) or None
                'success': infos[-1].get('goal_reached', False),
                'collision': np.array(collisions),
                'steps': len(actions),
                'total_reward': np.sum(rewards),
                'final_distance_to_goal': infos[-1]['goal_distance'],
                'infos': infos
            }
            
           
            
            if trajectory_data['rewards'].shape[0] > 20:
                long_traj = True
                
        return trajectory_data
    
    def collect_trajectories(self, 
                           n_trajectories: int,
                           seeds: Optional[List[int]] = None,
                           initial_states: Optional[List[List[float]]] = None,
                           save_images: bool = False,
                           verbose: bool = True) -> List[Dict[str, Any]]:
        """
        Collect multiple trajectories.
        
        Args:
            n_trajectories: Number of trajectories to collect
            seeds: Optional list of seeds (if None, uses sequential seeds)
            initial_states: Optional list of initial states
            save_images: Whether to save image observations
            verbose: Whether to print progress
            
        Returns:
            List of trajectory data dictionaries
        """
        if seeds is None:
            seeds = list(range(n_trajectories))
        
        if initial_states is not None and len(initial_states) != n_trajectories:
            raise ValueError(f"Number of initial states ({len(initial_states)}) must match n_trajectories ({n_trajectories})")
        
        trajectories = []
        
        if verbose:
            print(f"Collecting {n_trajectories} trajectories...")
            print(f"Controller: {self.config.controller.controller_type}")
            print(f"Environment: {len(self.config.environment.obstacles)} obstacles")
            print(f"Save images: {save_images}")
            print("-" * 50)
        
        for i in range(n_trajectories):
            seed = seeds[i] if i < len(seeds) else i
            initial_state = initial_states[i] if initial_states is not None else None
            
            trajectory = self.collect_single_trajectory(
                episode_idx=i,
                seed=seed,
                initial_state=initial_state,
                save_images=save_images
            )
            
            trajectories.append(trajectory)
            
            if verbose and (i + 1) % max(1, n_trajectories // 10) == 0:
                success_rate = np.mean([t['success'] for t in trajectories])
                avg_steps = np.mean([t['steps'] for t in trajectories])
                print(f"Progress: {i+1:3d}/{n_trajectories} | "
                      f"Success rate: {success_rate:.1%} | "
                      f"Avg steps: {avg_steps:.1f}")
        
        if verbose:
            success_rate = np.mean([t['success'] for t in trajectories])
            collision_rate = np.mean([t['collision'].any() for t in trajectories])
            avg_steps = np.mean([t['steps'] for t in trajectories])
            avg_reward = np.mean([t['total_reward'] for t in trajectories])
            
            print("-" * 50)
            print(f"Collection complete!")
            print(f"Success rate: {success_rate:.1%}")
            print(f"Collision rate: {collision_rate:.1%}")
            print(f"Average steps: {avg_steps:.1f}")
            print(f"Average reward: {avg_reward:.2f}")
        
        return trajectories
    
    def save_to_hdf5(self, trajectories: List[Dict[str, Any]], filename: str, 
                     compress: bool = True) -> str:
        """
        Save trajectories to HDF5 file.
        
        Args:
            trajectories: List of trajectory data
            filename: Output filename (without extension)
            compress: Whether to use compression
            
        Returns:
            Full path to saved file
        """
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        full_filename = filename + ".h5" #f"{filename}_{timestamp}.h5"
        filepath = os.path.join(self.output_dir, full_filename)
        
        # Compression settings
        compression = 'gzip' if compress else None
        compression_opts = 9 if compress else None
        
        with h5py.File(filepath, 'w') as f:
            # Save metadata
            metadata_group = f.create_group('metadata')
            
            # Save configuration as JSON string
            metadata_group.attrs['config'] = str(self.metadata['config'])
            metadata_group.attrs['collection_timestamp'] = self.metadata['collection_timestamp']
            metadata_group.attrs['n_trajectories'] = len(trajectories)
            
            # Environment info
            env_group = metadata_group.create_group('environment')
            env_info = self.metadata['environment_info']
            env_group.attrs['world_bounds'] = env_info['world_bounds']
            env_group.attrs['goal_position'] = env_info['goal_position']
            env_group.attrs['dt'] = env_info['dt']
            env_group.attrs['speed'] = env_info['speed']
            
            # Obstacles
            if env_info['obstacles']:
                obstacles_data = np.array([(obs[0], obs[1], obs[2]) for obs in env_info['obstacles']])
                env_group.create_dataset('obstacles', data=obstacles_data,
                                       compression=compression, compression_opts=compression_opts)
            
            # Controller info
            ctrl_group = metadata_group.create_group('controller')
            ctrl_info = self.metadata['controller_info']
            ctrl_group.attrs['type'] = ctrl_info['type']
            ctrl_group.attrs['max_angular_velocity'] = ctrl_info['max_angular_velocity']
            
            # Save trajectory data
            traj_group = f.create_group('trajectories')
            
            # Aggregate statistics
            stats_group = f.create_group('statistics')
            success_rate = np.mean([t['success'] for t in trajectories])
            collision_rate = np.mean([t['collision'].any() for t in trajectories])
            avg_steps = np.mean([t['steps'] for t in trajectories])
            avg_reward = np.mean([t['total_reward'] for t in trajectories])
            
            stats_group.attrs['success_rate'] = success_rate
            stats_group.attrs['collision_rate'] = collision_rate
            stats_group.attrs['average_steps'] = avg_steps
            stats_group.attrs['average_reward'] = avg_reward
            
            # Individual trajectory data
            for i, traj in enumerate(trajectories):
                traj_subgroup = traj_group.create_group(f'trajectory_{i:04d}')
                
                # Basic info
                traj_subgroup.attrs['episode_idx'] = traj['episode_idx']
                traj_subgroup.attrs['seed'] = traj['seed'] if traj['seed'] is not None else -1
                traj_subgroup.attrs['success'] = traj['success']
                traj_subgroup.attrs['collision'] = traj['collision']
                traj_subgroup.attrs['steps'] = traj['steps']
                traj_subgroup.attrs['total_reward'] = traj['total_reward']
                traj_subgroup.attrs['final_distance_to_goal'] = traj['final_distance_to_goal']
                
                # Time series data
                traj_subgroup.create_dataset('initial_state', data=traj['initial_state'])
                traj_subgroup.create_dataset('states', data=traj['states'],
                                           compression=compression, compression_opts=compression_opts)
                traj_subgroup.create_dataset('actions', data=traj['actions'],
                                           compression=compression, compression_opts=compression_opts)
                traj_subgroup.create_dataset('rewards', data=traj['rewards'],
                                           compression=compression, compression_opts=compression_opts)
                
                traj_subgroup.create_dataset('failures', data=traj['collision'],
                                           compression=compression, compression_opts=compression_opts)
                if 'margin_gp' in traj.keys():
                    traj_subgroup.create_dataset('margin_gp', data=traj['margin_gp'],
                                            compression=compression, compression_opts=compression_opts)
                if 'margin_nogp' in traj.keys():
                    traj_subgroup.create_dataset('margin_nogp', data=traj['margin_nogp'],
                                               compression=compression, compression_opts=compression_opts)
                # Images (if available)
                if traj['observations'] is not None:
                    traj_subgroup.create_dataset('observations', data=traj['observations'],
                                               compression=compression, compression_opts=compression_opts)
        
        print(f"Trajectories saved to: {filepath}")
        return filepath
    
    def close(self):
        """Clean up resources."""
        if hasattr(self, 'env'):
            self.env.close()


def load_trajectories_from_hdf5(filepath: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Load trajectories from HDF5 file.
    
    Args:
        filepath: Path to HDF5 file
        
    Returns:
        Tuple of (trajectories_list, metadata_dict)
    """
    trajectories = []
    metadata = {}
    
    with h5py.File(filepath, 'r') as f:
        # Load metadata
        metadata['collection_timestamp'] = f['metadata'].attrs['collection_timestamp']
        metadata['n_trajectories'] = f['metadata'].attrs['n_trajectories']
        
        # Environment info
        env_info = {}
        env_group = f['metadata']['environment']
        env_info['world_bounds'] = tuple(env_group.attrs['world_bounds'])
        env_info['goal_position'] = tuple(env_group.attrs['goal_position'])
        env_info['dt'] = env_group.attrs['dt']
        env_info['speed'] = env_group.attrs['speed']
        
        if 'obstacles' in env_group:
            obstacles_data = env_group['obstacles'][:]
            env_info['obstacles'] = [(obs[0], obs[1], obs[2]) for obs in obstacles_data]
        else:
            env_info['obstacles'] = []
        
        metadata['environment_info'] = env_info
        
        # Controller info
        ctrl_group = f['metadata']['controller']
        metadata['controller_info'] = {
            'type': ctrl_group.attrs['type'],
            'max_angular_velocity': ctrl_group.attrs['max_angular_velocity']
        }
        
        # Statistics
        if 'statistics' in f:
            stats_group = f['statistics']
            metadata['statistics'] = {
                'success_rate': stats_group.attrs['success_rate'],
                'collision_rate': stats_group.attrs['collision_rate'],
                'average_steps': stats_group.attrs['average_steps'],
                'average_reward': stats_group.attrs['average_reward']
            }
        
        # Load trajectories
        traj_group = f['trajectories']
        for traj_name in sorted(traj_group.keys()):
            traj_data = traj_group[traj_name]
            
            trajectory = {
                'episode_idx': traj_data.attrs['episode_idx'],
                'seed': traj_data.attrs['seed'] if traj_data.attrs['seed'] != -1 else None,
                'success': traj_data.attrs['success'],
                'collision': traj_data.attrs['collision'],
                'steps': traj_data.attrs['steps'],
                'total_reward': traj_data.attrs['total_reward'],
                'final_distance_to_goal': traj_data.attrs['final_distance_to_goal'],
                'initial_state': traj_data['initial_state'][:],
                'states': traj_data['states'][:],
                'actions': traj_data['actions'][:],
                'rewards': traj_data['rewards'][:]
            }
            
            # Load observations if available
            if 'observations' in traj_data:
                trajectory['observations'] = traj_data['observations'][:]
            else:
                trajectory['observations'] = None
            
            trajectories.append(trajectory)
    
    return trajectories, metadata


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Collect trajectories from Dubins car environment')
    parser.add_argument('--n_trajectories', type=int, default=10, help='Number of trajectories to collect')
    parser.add_argument('--controller', type=str, default='mppi', choices=['simple', 'mpc', 'random', 'mppi', 'diffusion'], 
                       help='Controller type to use')
    parser.add_argument('--config', type=str, default='mppi', 
                       choices=['default', 'mpc', 'random','mppi',
                               'diffusion', 'diffusion_wm', 'no_obstacles', 'narrow_gap', 'debug'],
                       help='Configuration preset to use')
    parser.add_argument('--output_dir', type=str, default='/data/dubins/trajs', help='Output directory')
    parser.add_argument('--filename', type=str, default='trajectories', help='Output filename (without extension)')
    parser.add_argument('--save_images', action='store_true', help='Save image observations')
    parser.add_argument('--compress', action='store_true', default=True, help='Use HDF5 compression')
    parser.add_argument('--verbose', action='store_true', default=True, help='Print progress')
    parser.add_argument('--checkpoint', type=int, default=1000, help='Checkpoint version')
    parser.add_argument('--use_wm_prediction', action='store_true', help='Enable world model prediction')
    parser.add_argument('--wm_checkpoint', type=str, default='/data/dubins/test/dreamer/rssm_ckpt.pt', 
                       help='Path to world model checkpoint')
    parser.add_argument('--wm_history_length', type=int, default=8, 
                       help='Number of timesteps of history to use for WM prediction')
    
    args = parser.parse_args()
    
    # Select configuration
    config_map = {
        'default': get_default_config,
        'mpc': get_mpc_config,
        'random': get_random_config,
        'mppi': get_mppi_config,
        'diffusion': get_diffusion_config,
        'diffusion_wm': get_diffusion_wm_config,
        'no_obstacles': get_no_obstacles_config,
        'debug': get_debug_config
    }
    
    if args.controller == 'diffusion':
        config = config_map[args.config](args.checkpoint)
    else:
        config = config_map[args.config]()
    config.controller.controller_type = args.controller
    
    # Create WM config if enabled
    wm_config = None
    if args.use_wm_prediction:
        wm_config = DreamerConfig()
        wm_config.use_wm_prediction = True
        wm_config.wm_checkpoint_path = args.wm_checkpoint
        # Set other required config values
        wm_config.turnRate = config.environment.max_angular_velocity
        wm_config.x_min = config.environment.world_bounds[0]
        wm_config.x_max = config.environment.world_bounds[1]
        wm_config.y_min = config.environment.world_bounds[2]
        wm_config.y_max = config.environment.world_bounds[3]
        wm_config.size = [128, 128]  # Image size
        # Set history length from command line argument
        wm_config.wm_history_length = args.wm_history_length
    
    # Create collector and collect trajectories
    collector = TrajectoryCollector(config, args.output_dir, wm_config)
    
    try:
        trajectories = collector.collect_trajectories(
            n_trajectories=args.n_trajectories,
            save_images=args.save_images,
            verbose=args.verbose
        )
        
        # Save to HDF5
        filepath = collector.save_to_hdf5(trajectories, args.filename, args.compress)
        
        print(f"\nCollection complete! Saved {len(trajectories)} trajectories to {filepath}")
        
    finally:
        collector.close()


if __name__ == "__main__":
    main()
