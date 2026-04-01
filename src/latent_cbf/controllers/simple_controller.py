"""
Simple Controller for DubinsEnv

A basic controller that uses privileged information to navigate to the goal
while avoiding obstacles. Uses a combination of:
1. Attractive force towards goal
2. Repulsive force from obstacles
3. Dubins car kinematic constraints
"""

import numpy as np
import math
from typing import Dict, Any, Tuple


class SimpleController:
    """
    A simple controller for the Dubins car environment.
    Uses artificial potential fields with obstacle avoidance.
    """
    
    def __init__(self,
                 goal_attraction_gain: float = 2.0,
                 obstacle_repulsion_gain: float = 3.0,
                 obstacle_influence_radius: float = 0.8,
                 max_angular_velocity: float = 3.3,
                 lookahead_distance: float = 0.3):
        """
        Initialize the controller.
        
        Args:
            goal_attraction_gain: Strength of attraction to goal
            obstacle_repulsion_gain: Strength of repulsion from obstacles
            obstacle_influence_radius: Distance at which obstacles start affecting the car
            max_angular_velocity: Maximum angular velocity (should match env)
            lookahead_distance: How far ahead to look for obstacle avoidance
        """
        self.k_goal = goal_attraction_gain
        self.k_obs = obstacle_repulsion_gain
        self.obs_radius = obstacle_influence_radius
        self.max_omega = max_angular_velocity
        self.lookahead = lookahead_distance
        
    def compute_action(self, info: Dict[str, Any], obstacles: list) -> float:
        """
        Compute control action based on current state and environment info.
        
        Args:
            info: Info dictionary from environment step
            obstacles: List of obstacles [(x, y, radius), ...]
            
        Returns:
            Angular velocity command (scalar)
        """
        # Extract state information
        agent_pos = info['agent_position']
        agent_theta = info['agent_orientation']
        goal_pos = np.array([1.3, 0.0])  # From your test setup
        
        # Current heading direction
        heading_vec = np.array([np.cos(agent_theta), np.sin(agent_theta)])
        
        # 1. Goal attraction force
        goal_vec = goal_pos - agent_pos
        goal_distance = np.linalg.norm(goal_vec)
        
        if goal_distance > 0.01:  # Avoid division by zero
            goal_direction = goal_vec / goal_distance
        else:
            goal_direction = np.array([0.0, 0.0])
        
        # 2. Obstacle repulsion force
        repulsion_force = np.array([0.0, 0.0])
        
        for obs_x, obs_y, obs_r in obstacles:
            obs_pos = np.array([obs_x, obs_y])
            
            # Distance from agent to obstacle surface
            obs_vec = agent_pos - obs_pos
            obs_distance = np.linalg.norm(obs_vec)
            surface_distance = obs_distance - obs_r
            
            # Also check lookahead point for proactive avoidance
            lookahead_pos = agent_pos + self.lookahead * heading_vec
            lookahead_obs_vec = lookahead_pos - obs_pos
            lookahead_obs_distance = np.linalg.norm(lookahead_obs_vec)
            lookahead_surface_distance = lookahead_obs_distance - obs_r
            
            # Use the closer of current position or lookahead position
            if lookahead_surface_distance < surface_distance:
                effective_distance = lookahead_surface_distance
                effective_vec = lookahead_obs_vec
            else:
                effective_distance = surface_distance
                effective_vec = obs_vec
            
            # Apply repulsion if within influence radius
            if effective_distance < self.obs_radius and effective_distance > 0:
                # Normalize direction away from obstacle
                repulsion_direction = effective_vec / np.linalg.norm(effective_vec)
                
                # Repulsion strength inversely proportional to distance
                repulsion_strength = self.k_obs * (1.0 / effective_distance - 1.0 / self.obs_radius)
                repulsion_force += repulsion_strength * repulsion_direction
        
        # 3. Combine forces to get desired direction
        desired_force = self.k_goal * goal_direction + repulsion_force
        
        if np.linalg.norm(desired_force) > 0.01:
            desired_direction = desired_force / np.linalg.norm(desired_force)
        else:
            desired_direction = heading_vec
        
        # 4. Convert desired direction to angular velocity
        # Calculate angle between current heading and desired direction
        cross_product = np.cross(heading_vec, desired_direction)
        dot_product = np.dot(heading_vec, desired_direction)
        
        # Angle to desired direction (signed)
        angle_error = math.atan2(cross_product, dot_product)
        
        # Simple proportional control for angular velocity
        angular_velocity = 3.0 * angle_error  # Proportional gain
        
        # Clamp to maximum angular velocity
        angular_velocity = np.clip(angular_velocity, -self.max_omega, self.max_omega)
        
        return angular_velocity
    
    def compute_action_simple(self, info: Dict[str, Any]) -> float:
        """
        Simplified version that only uses goal information (no obstacles).
        
        Args:
            info: Info dictionary from environment step
            
        Returns:
            Angular velocity command (scalar)
        """
        agent_pos = info['agent_position']
        agent_theta = info['agent_orientation']
        goal_pos = np.array([1.3, 0.0])
        
        # Vector to goal
        goal_vec = goal_pos - agent_pos
        goal_angle = math.atan2(goal_vec[1], goal_vec[0])
        
        # Current heading
        heading_vec = np.array([np.cos(agent_theta), np.sin(agent_theta)])
        goal_direction = goal_vec / (np.linalg.norm(goal_vec) + 1e-8)
        
        # Calculate angle error
        cross_product = np.cross(heading_vec, goal_direction)
        dot_product = np.dot(heading_vec, goal_direction)
        angle_error = math.atan2(cross_product, dot_product)
        
        # Proportional control
        angular_velocity = 2.0 * angle_error
        angular_velocity = np.clip(angular_velocity, -self.max_omega, self.max_omega)
        
        return angular_velocity


def test_controller():
    """Test the controller with the DubinsEnv."""
    import sys
    sys.path.append('/home/kensuke/WM_CBF/wm_cbf_dubin')
    from dubins_env_outline import DubinsEnv, create_circular_obstacles
    import cv2
    
    # Create environment
    obstacles = create_circular_obstacles([[0, 0.65, 0.5], [0, -0.65, 0.5]])
    
    env = DubinsEnv(
        image_size=(128, 128),
        obstacles=obstacles,
        goal_position=(1.3, 0.0),
        render_mode="rgb_array"
    )
    
    # Create controller
    controller = SimpleController(
        goal_attraction_gain=2.0,
        obstacle_repulsion_gain=4.0,
        obstacle_influence_radius=0.8,
        max_angular_velocity=3.3,
        lookahead_distance=0.4
    )
    
    # Test episode
    obs, info = env.reset(seed=42)
    observations = [obs]
    
    print("Testing controller...")
    print(f"Initial position: {info['agent_position'], info['agent_orientation']}")
    print(f"Goal position: (1.3, 0.0)")
    print(f"Initial goal distance: {info['goal_distance']:.3f}")
    
    for step in range(100):
        # Use controller to compute action
        action = controller.compute_action(info, obstacles)
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        observations.append(obs)
        
        print(f"Step {step+1}: pos=({info['agent_position'][0]:.2f}, {info['agent_position'][1]:.2f}), "
              f"goal_dist={info['goal_distance']:.3f}, action={action:.2f}, reward={reward:.2f}")
        
        if terminated or truncated:
            if info['goal_reached']:
                print(f"🎉 Goal reached in {step+1} steps!")
            elif info['collision']:
                print(f"💥 Collision occurred at step {step+1}")
            else:
                print(f"🚫 Episode ended at step {step+1}")
            break
    
    # Save video
    if len(observations) > 1:
        height, width, channels = observations[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter('controller_test.mp4', fourcc, 10.0, (width, height))
        
        for obs in observations:
            frame_bgr = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)
        
        video_writer.release()
        print(f"Video saved as 'controller_test.mp4' with {len(observations)} frames")
    
    env.close()


def test_simple_controller():
    """Test the simplified controller (no obstacle avoidance)."""
    import sys
    sys.path.append('/home/kensuke/WM_CBF/wm_cbf_dubin')
    from dubins_env_outline import DubinsEnv, create_circular_obstacles
    
    # Create environment without obstacles
    env = DubinsEnv(
        image_size=(128, 128),
        obstacles=[],  # No obstacles
        goal_position=(1.3, 0.0),
        render_mode="rgb_array"
    )
    
    controller = SimpleController()
    
    obs, info = env.reset(seed=42)
    print("Testing simple controller (no obstacles)...")
    
    for step in range(50):
        action = controller.compute_action_simple(info)
        obs, reward, terminated, truncated, info = env.step(action)
        
        if step % 10 == 0:
            print(f"Step {step}: goal_distance={info['goal_distance']:.3f}")
        
        if terminated or truncated:
            print(f"Episode ended at step {step+1}, goal_reached: {info['goal_reached']}")
            break
    
    env.close()


if __name__ == "__main__":
    print("Testing controller with obstacles...")
    test_controller()
    
    print("\n" + "="*50 + "\n")
    
    #print("Testing simple controller without obstacles...")
   # test_simple_controller()
