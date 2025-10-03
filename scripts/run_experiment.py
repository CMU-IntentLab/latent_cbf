"""
Experiment runner using the configuration system.

This script demonstrates how to use the config system to run experiments
with different environment and controller configurations.
"""

import sys
import os
import cv2
import numpy as np
from typing import List, Dict, Any

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our modules
from configs import (Config, get_default_config, get_no_obstacles_config, 
                   get_debug_config, get_mpc_config, get_mppi_config,
                   get_diffusion_config)
from dubins_env import DubinsEnv
from controllers import SimpleController, MPCController, MPPIController, DiffusionController


def create_env_from_config(config: Config) -> DubinsEnv:
    """
    Create a DubinsEnv from configuration.
    
    Args:
        config: Configuration object
        
    Returns:
        Configured DubinsEnv instance
    """
    env_config = config.environment
    
    return DubinsEnv(
        image_size=env_config.image_size,
        world_bounds=env_config.world_bounds,
        max_angular_velocity=env_config.max_angular_velocity,
        speed=env_config.speed,
        dt=env_config.dt,
        max_episode_steps=env_config.max_episode_steps,
        obstacles=env_config.get_obstacles_list(),
        goal_position=env_config.goal_position,
        goal_radius=env_config.goal_radius,
        collision_radius=env_config.collision_radius,
        render_mode=env_config.render_mode,
        config=config  # Pass the full config for reset distribution settings
    )


def create_controller_from_config(config: Config):
    """
    Create a controller from configuration (Simple, MPC, MPPI, or Diffusion).
    
    Args:
        config: Configuration object
        
    Returns:
        Configured controller instance (SimpleController, MPCController, MPPIController, or DiffusionController)
    """
    ctrl_config = config.controller
    
    if ctrl_config.controller_type == "mpc":
        return MPCController(
            prediction_horizon=ctrl_config.prediction_horizon,
            control_horizon=ctrl_config.control_horizon,
            dt=config.environment.dt,
            speed=config.environment.speed,
            max_angular_velocity=config.environment.max_angular_velocity,
            goal_weight=ctrl_config.goal_weight,
            obstacle_weight=ctrl_config.obstacle_weight,
            control_weight=ctrl_config.control_weight,
            obstacle_safety_margin=ctrl_config.obstacle_safety_margin,
            goal_tolerance=ctrl_config.goal_tolerance
        )
    elif ctrl_config.controller_type == "mppi":
        return MPPIController(
            prediction_horizon=ctrl_config.prediction_horizon,
            num_samples=ctrl_config.num_samples,
            dt=config.environment.dt,
            speed=config.environment.speed,
            max_angular_velocity=config.environment.max_angular_velocity,
            temperature=ctrl_config.temperature,
            lambda_param=ctrl_config.lambda_param,
            goal_weight=ctrl_config.goal_weight,
            obstacle_weight=ctrl_config.obstacle_weight,
            control_weight=ctrl_config.control_weight,
            obstacle_safety_margin=ctrl_config.obstacle_safety_margin,
            goal_tolerance=ctrl_config.goal_tolerance,
            noise_variance=ctrl_config.noise_variance,
            adaptive_temperature=ctrl_config.adaptive_temperature
        )
    elif ctrl_config.controller_type == "diffusion":
        return DiffusionController(
            checkpoint_path=ctrl_config.checkpoint_path,
            config_path=ctrl_config.config_path,
            device=ctrl_config.device,
            action_chunk_size=ctrl_config.action_chunk_size,
            total_chunk_size=ctrl_config.total_chunk_size,
            eval_diffusion_steps=ctrl_config.eval_diffusion_steps
        )
    else:  # simple controller
        return SimpleController(
            goal_attraction_gain=ctrl_config.goal_attraction_gain,
            obstacle_repulsion_gain=ctrl_config.obstacle_repulsion_gain,
            obstacle_influence_radius=ctrl_config.obstacle_influence_radius,
            max_angular_velocity=config.environment.max_angular_velocity,
            lookahead_distance=ctrl_config.lookahead_distance
        )


def run_episode(env: DubinsEnv, controller, config: Config) -> Dict[str, Any]:
    """
    Run a single episode with the given environment and controller.
    
    Args:
        env: Environment instance
        controller: Controller instance
        config: Configuration object
        
    Returns:
        Episode results dictionary
    """
    exp_config = config.experiment
    env_config = config.environment
    
    # Reset environment
    if exp_config.initial_position is not None:
        obs, info = env.reset(
            seed=exp_config.env_seed,
            options={'initial_state': exp_config.initial_position}
        )
    else:
        obs, info = env.reset(seed=exp_config.env_seed)
    
    # Track episode data
    observations = [obs] if exp_config.save_video else []
    trajectory = [info['agent_position'].copy()]
    rewards = []
    actions = []
    
    if exp_config.verbose:
        print(f"Initial position: {info['agent_position']}, orientation: {info['agent_orientation']:.3f}")
        print(f"Goal position: {env_config.goal_position}")
        print(f"Initial goal distance: {info['goal_distance']:.3f}")
    
    # Reset controller if it has a reset method (MPC)
    if hasattr(controller, 'reset'):
        controller.reset()
    
    # Run episode
    for step in range(exp_config.max_test_steps):
        # Compute action using controller
        if config.controller.controller_type in ["mpc", "mppi"]:
            action = controller.compute_action(info, env_config.get_obstacles_list(), 
                                             np.array(env_config.goal_position))
        else:
            action = controller.compute_action(info, env_config.get_obstacles_list())
        actions.append(action)
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        trajectory.append(info['agent_position'].copy())
        
        if exp_config.save_video:
            observations.append(obs)
        
        # Logging
        if exp_config.verbose and (step % exp_config.log_interval == 0 or step < 5):
            print(f"Step {step+1:3d}: pos=({info['agent_position'][0]:5.2f}, {info['agent_position'][1]:5.2f}), "
                  f"θ={info['agent_orientation']:5.2f}, goal_dist={info['goal_distance']:.3f}, "
                  f"action={action:5.2f}, reward={reward:6.2f}")
        
        # Check termination
        if terminated or truncated:
            if exp_config.verbose:
                if info['goal_reached']:
                    print(f"🎉 Goal reached in {step+1} steps!")
                elif info['collision']:
                    print(f"💥 Collision occurred at step {step+1}")
                else:
                    print(f"🚫 Episode ended at step {step+1} (boundary exit)")
            break
    
    # Compile results
    results = {
        'success': info.get('goal_reached', False),
        'collision': info.get('collision', False),
        'steps': step + 1,
        'final_distance': info['goal_distance'],
        'total_reward': sum(rewards),
        'trajectory': trajectory,
        'actions': actions,
        'rewards': rewards
    }
    
    # Save video if requested
    if exp_config.save_video and observations:
        save_video(observations, exp_config.video_filename, config.rendering.video_fps)
        if exp_config.verbose:
            print(f"Video saved as '{exp_config.video_filename}' with {len(observations)} frames")
    
    return results


def save_video(observations: List[np.ndarray], filename: str, fps: float = 10.0):
    """
    Save observations as a video file.
    
    Args:
        observations: List of RGB image arrays
        filename: Output video filename
        fps: Frames per second
    """
    if not observations:
        return
    
    height, width, channels = observations[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    
    for obs in observations:
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
        video_writer.write(frame_bgr)
    
    video_writer.release()


def run_experiment(config: Config) -> Dict[str, Any]:
    """
    Run a complete experiment with the given configuration.
    
    Args:
        config: Configuration object
        
    Returns:
        Experiment results
    """
    if config.experiment.verbose:
        print(f"Running experiment with {len(config.environment.obstacles)} obstacles")
        print(f"Environment bounds: {config.environment.world_bounds}")
        print(f"Goal position: {config.environment.goal_position}")
        print(f"Controller type: {config.controller.controller_type}")
        if config.controller.controller_type == "mpc":
            print(f"MPC parameters: horizon={config.controller.prediction_horizon}, "
                  f"goal_weight={config.controller.goal_weight}, "
                  f"obstacle_weight={config.controller.obstacle_weight}")
        elif config.controller.controller_type == "mppi":
            print(f"MPPI parameters: horizon={config.controller.prediction_horizon}, "
                  f"samples={config.controller.num_samples}, "
                  f"temperature={config.controller.temperature}, "
                  f"goal_weight={config.controller.goal_weight}, "
                  f"obstacle_weight={config.controller.obstacle_weight}")
        else:
            print(f"Simple controller gains: goal={config.controller.goal_attraction_gain}, "
                  f"obstacle={config.controller.obstacle_repulsion_gain}")
        print("-" * 60)
    
    # Create environment and controller
    env = create_env_from_config(config)
    controller = create_controller_from_config(config)
    
    # Run episodes
    all_results = []
    for episode in range(config.experiment.num_test_episodes):
        if config.experiment.verbose and config.experiment.num_test_episodes > 1:
            print(f"\nEpisode {episode + 1}/{config.experiment.num_test_episodes}")
        
        # Update video filename for multiple episodes
        if config.experiment.num_test_episodes > 1:
            base_name = config.experiment.video_filename.rsplit('.', 1)[0]
            extension = config.experiment.video_filename.rsplit('.', 1)[1]
            config.experiment.video_filename = f"{base_name}_ep{episode+1}.{extension}"
        
        results = run_episode(env, controller, config)
        all_results.append(results)
    
    env.close()
    
    # Compile summary statistics
    successes = sum(1 for r in all_results if r['success'])
    collisions = sum(1 for r in all_results if r['collision'])
    avg_steps = np.mean([r['steps'] for r in all_results])
    avg_reward = np.mean([r['total_reward'] for r in all_results])
    avg_final_distance = np.mean([r['final_distance'] for r in all_results])
    
    summary = {
        'success_rate': successes / len(all_results),
        'collision_rate': collisions / len(all_results),
        'average_steps': avg_steps,
        'average_reward': avg_reward,
        'average_final_distance': avg_final_distance,
        'individual_results': all_results
    }
    
    if config.experiment.verbose:
        print("\n" + "="*60)
        print("EXPERIMENT SUMMARY")
        print("="*60)
        print(f"Episodes: {len(all_results)}")
        print(f"Success rate: {summary['success_rate']:.1%}")
        print(f"Collision rate: {summary['collision_rate']:.1%}")
        print(f"Average steps: {summary['average_steps']:.1f}")
        print(f"Average reward: {summary['average_reward']:.2f}")
        print(f"Average final distance: {summary['average_final_distance']:.3f}")
    
    return summary


def compare_configurations():
    """Compare different configurations side by side."""
    configs = {
        "Default": get_default_config(),
        "No Obstacles": get_no_obstacles_config(),
        "Narrow Gap": get_narrow_gap_config()
    }
    
    print("CONFIGURATION COMPARISON")
    print("="*80)
    
    results = {}
    for name, config in configs.items():
        print(f"\nTesting {name} configuration...")
        config.experiment.verbose = False  # Reduce output for comparison
        config.experiment.save_video = False  # Don't save videos for comparison
        config.experiment.num_test_episodes = 3  # Multiple episodes for statistics
        
        summary = run_experiment(config)
        results[name] = summary
        
        print(f"{name}: Success={summary['success_rate']:.1%}, "
              f"Steps={summary['average_steps']:.1f}, "
              f"Reward={summary['average_reward']:.1f}")
    
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    for name, summary in results.items():
        print(f"{name:15}: {summary['success_rate']:5.1%} success, "
              f"{summary['average_steps']:5.1f} steps, "
              f"{summary['average_reward']:6.1f} reward")


if __name__ == "__main__":
    # Example 1: MPC Controller
    print("Example 1: MPC Controller")

    # run 5 times with different random seeds
    for i in range(5):
        config = get_mpc_config()
        config.experiment.verbose = True
        config.experiment.save_video = True
        config.experiment.env_seed = i
        config.experiment.video_filename = f"video/mpc_config_{i}.mp4"
        results = run_experiment(config)
        print(f"Results for seed {i}: {results}")

