"""
DubinsEnv - A Gym Environment for Dubins Car with Image Observations

This outline provides a comprehensive structure for a Dubins car environment
that returns image observations, suitable for vision-based reinforcement learning.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from PIL import Image, ImageDraw
import math
from typing import Tuple, Dict, Any, Optional, Union


class DubinsEnv(gym.Env):
    """
    A Dubins car environment with image observations.
    
    The agent controls a car that can only move forward and turn with a limited
    turning radius. The environment returns RGB images as observations.
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(self, 
                 image_size: Tuple[int, int] = (64, 64),
                 world_bounds: Tuple[float, float, float, float] = (-1.5, 1.5, -1.5, 1.5),
                 max_angular_velocity: float = 3.3,
                 speed: float = 1.0,
                 dt: float = 0.05,
                 max_episode_steps: int = 200,
                 obstacles: Optional[list] = None,
                 goal_position: Optional[Tuple[float, float]] = None,
                 goal_radius: float = 0.05,
                 collision_radius: float = 0.0,
                 render_mode: Optional[str] = None,
                 config: Optional[object] = None):
        """
        Initialize the Dubins car environment.
        
        Args:
            image_size: Size of the rendered image (width, height)
            world_bounds: World boundaries (x_min, x_max, y_min, y_max)
            max_speed: Maximum forward speed
            max_angular_velocity: Maximum turning rate
            dt: Time step for integration
            max_episode_steps: Maximum steps per episode
            obstacles: List of circular obstacles [(x, y, radius), ...]
            goal_position: Target position (x, y)
            goal_radius: Radius around goal for success
            collision_radius: Agent collision radius
            render_mode: Rendering mode ("human", "rgb_array", or None)
            config: Configuration object containing reset distribution settings
        """
        super().__init__()
        
        # Environment parameters
        self.image_size = image_size
        self.world_bounds = world_bounds  # (x_min, x_max, y_min, y_max)
        self.max_angular_velocity = max_angular_velocity
        self.speed = speed
        self.dt = dt
        self.max_episode_steps = max_episode_steps
        self.collision_radius = collision_radius
        self.goal_radius = goal_radius
        self.render_mode = render_mode
        self.config = config  # Store config for reset distribution settings
        
        # World setup
        self.x_min, self.x_max, self.y_min, self.y_max = world_bounds
        self.obstacles = obstacles or []
        self.goal_position = goal_position or (self.x_max - 1.0, self.y_max - 1.0)
        
        # State variables
        self.state = None  # [x, y, theta] - position and orientation
        self.step_count = 0
        
        # Define action space:  angular_velocity
        self.action_space = spaces.Box(
            low=-self.max_angular_velocity,
            high=self.max_angular_velocity,
            dtype=np.float32
        )
        
        # Define observation space: RGB image
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(image_size[1], image_size[0], 3),  # (height, width, channels)
            dtype=np.uint8
        )
        
        # Rendering setup
        self._setup_rendering()
    
    def _setup_rendering(self):
        """Initialize rendering parameters and colors."""
        self.colors = {
            'background': 'white',
            'agent': 'blue',
            'goal': 'green',
            'obstacle': 'red',
            'trajectory': 'lightblue'
        }
        self.agent_size = 0.3  # Visual size of the agent
        self.trajectory = []  # Store agent trajectory for visualization
    
    def reset(self, 
              seed: Optional[int] = None, 
              options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options for reset
            
        Returns:
            observation: Initial image observation
            info: Additional information dictionary
        """
        super().reset(seed=seed)
        
        # Reset step counter
        self.step_count = 0
        
        # Initialize agent state
        self.state = self._sample_initial_state(options)
        
        # Clear trajectory
        self.trajectory = [self.state[:2].copy()]
        
        # Generate initial observation
        observation = self._get_observation()
        
        # Prepare info dictionary
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: float) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action scalar (angular_velocity)
            
        Returns:
            observation: New image observation
            reward: Reward for this step
            terminated: Whether episode is terminated (goal reached/collision)
            truncated: Whether episode is truncated (max steps reached)
            info: Additional information dictionary
        """
        # Clip actions to valid range
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # Update state using Dubins car dynamics
        self._update_state(action)
        
        # Update trajectory
        self.trajectory.append(self.state[:2].copy())
        
        # Increment step counter
        self.step_count += 1
        
        # Calculate reward
        reward = self._calculate_reward(action)
        
        # Check termination conditions
        terminated = self._is_terminated()
        truncated = self.step_count >= self.max_episode_steps
        
        # Generate observation
        observation = self._get_observation()
        
        # Prepare info dictionary
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def render(self) -> Optional[np.ndarray]:
        """
        Render the environment.
        
        Returns:
            RGB array if render_mode is "rgb_array", None otherwise
        """
        if self.render_mode is None:
            return None
            
        # Generate high-quality image
        img_array = self._render_image()
        
        if self.render_mode == "human":
            # Display image (implementation depends on your display backend)
            self._display_image(img_array)
        elif self.render_mode == "rgb_array":
            return img_array
    
    def close(self):
        """Clean up rendering resources."""
        # Close any open display windows or resources
        pass
    
    # ==================== PRIVATE METHODS ====================
    
    def _sample_initial_state(self, options: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Sample a valid initial state for the agent.
        
        Args:
            options: Reset options that may specify initial conditions
            
        Returns:
            Initial state [x, y, theta]
        """
        # Check for explicit initial state in options
        if options and 'initial_state' in options:
            return np.array(options['initial_state'], dtype=np.float32)
        
        # Use config-based reset distribution if available
        if self.config and hasattr(self.config, 'environment'):
            env_config = self.config.environment
            
            # Check for deterministic start
            if env_config.deterministic_start and env_config.initial_position is not None:
                return np.array(env_config.initial_position, dtype=np.float32)
            
            # Use config reset bounds
            x_range, y_range, theta_range = env_config.get_reset_bounds()
            x_min, x_max = x_range
            y_min, y_max = y_range
            theta_min, theta_max = theta_range
        else:
            # Fallback to world bounds and default theta range
            x_min, x_max = self.x_min, self.x_max
            y_min, y_max = self.y_min, self.y_max
            theta_min, theta_max = -np.pi/2, np.pi/2

        
        # Sample random position avoiding obstacles
        max_attempts = 100
        for _ in range(max_attempts):
            x = self.np_random.uniform(x_min, x_max)
            y = self.np_random.uniform(y_min, y_max)
            theta = self.np_random.uniform(theta_min, theta_max)
            
            state = np.array([x, y, theta], dtype=np.float32)
            
            if not self._check_collision(state):
                return state
        
        # Fallback to a safe default position
        return np.array([x_min + 0.1, y_min + 0.1, 0.0], dtype=np.float32)
    
    def _update_state(self, action: float):
        """
        Update agent state using Dubins car dynamics with RK4 integration.
        
        Args:
            action: Control input angular_velocity (scalar)
        """
        omega = action[0]  # Action is now a scalar
        x, y, theta = self.state
        
        # Define the dynamics function f(state, omega)
        def dynamics(state, omega):
            x, y, theta = state
            dx_dt = self.speed * np.cos(theta)
            dy_dt = self.speed * np.sin(theta)
            dtheta_dt = omega
            return np.array([dx_dt, dy_dt, dtheta_dt])
        
        # RK4 integration
        k1 = dynamics(self.state, omega)
        k2 = dynamics(self.state + 0.5 * self.dt * k1, omega)
        k3 = dynamics(self.state + 0.5 * self.dt * k2, omega)
        k4 = dynamics(self.state + self.dt * k3, omega)
        
        # Update state using RK4 formula
        state_new = self.state + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        # Normalize angle to [-pi, pi]
        state_new[2] = np.arctan2(np.sin(state_new[2]), np.cos(state_new[2]))
        
        # Update state
        self.state = state_new.astype(np.float32)
    
    def _calculate_reward(self, action: float) -> float:
        """
        Calculate reward for the current step.
        
        Args:
            action: Action taken this step
            
        Returns:
            Reward value
        """
        reward = 0.0
        
        # Distance to goal reward
        goal_distance = np.linalg.norm(self.state[:2] - np.array(self.goal_position))
        reward += -0.1 * goal_distance  # Negative distance encourages approaching goal
        
        # Goal reached reward
        if goal_distance <= self.goal_radius:
            reward += 100.0
        
        # Collision penalty
        if self._check_collision(self.state):
            reward += -100.0
        
        # Action penalty (encourage smooth control)
        omega = action[0]  # Action is now a scalar
        reward += -0.01 * (omega**2)
        
        # Time penalty (encourage efficiency)
        reward += -0.1
        
        return reward
    
    def _is_terminated(self) -> bool:
        """
        Check if episode should be terminated.
        
        Returns:
            True if episode is terminated
        """
        # Goal reached
        goal_distance = np.linalg.norm(self.state[:2] - np.array(self.goal_position))
        if goal_distance <= self.goal_radius * 4:
            return True
        
        # Only terminate if the car exits the x-y bounds
        x, y = self.state[0], self.state[1]
        if not (self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max):
            return True
        return False
    
    def _check_collision(self, state: np.ndarray) -> bool:
        """
        Check if agent collides with obstacles.
        
        Args:
            state: Agent state [x, y, theta]
            
        Returns:
            True if collision detected
        """
        agent_pos = state[:2]
        
        for obs_x, obs_y, obs_radius in self.obstacles:
            obs_pos = np.array([obs_x, obs_y])
            distance = np.linalg.norm(agent_pos - obs_pos)
            
            if distance <= (self.collision_radius + obs_radius):
                return True
        
        return False
    
    def _get_observation(self) -> np.ndarray:
        """
        Generate image observation of current state.
        
        Returns:
            RGB image array
        """
        return self._render_image()
    
    def _get_info(self) -> Dict[str, Any]:
        """
        Get additional information about current state.
        
        Returns:
            Information dictionary
        """
        goal_distance = np.linalg.norm(self.state[:2] - np.array(self.goal_position))
        
        return {
            'agent_position': self.state[:2].copy(),
            'agent_orientation': self.state[2],
            'goal_distance': goal_distance,
            'goal_reached': goal_distance <= self.goal_radius*4,
            'collision': self._check_collision(self.state),
            'step_count': self.step_count
        }
    
    def _render_image(self) -> np.ndarray:
        """
        Render the current state as an RGB image.
        
        Returns:
            RGB image array of shape (height, width, 3)
        """
        # Use anti-aliasing for high-quality rendering
        scale = 4
        h_size = (self.image_size[0] * scale, self.image_size[1] * scale)
        
        # Create high-resolution image
        img = Image.new('RGB', h_size, self.colors['background'])
        draw = ImageDraw.Draw(img)
        
        # Helper function to convert world coordinates to pixel coordinates
        def world_to_pixel(world_coord):
            x, y = world_coord
            px = int((x - self.x_min) / (self.x_max - self.x_min) * h_size[0])
            py = int((self.y_max - y) / (self.y_max - self.y_min) * h_size[1])
            return (px, py)
        
        # Draw obstacles
        for obs_x, obs_y, obs_radius in self.obstacles:
            center_px = world_to_pixel((obs_x, obs_y))
            radius_px = obs_radius / (self.x_max - self.x_min) * h_size[0]
            
            draw.ellipse(
                [(center_px[0] - radius_px, center_px[1] - radius_px),
                 (center_px[0] + radius_px, center_px[1] + radius_px)],
                fill=self.colors['obstacle'],
                width=2 * scale
            )
        
        # Draw goal
        goal_px = world_to_pixel(self.goal_position)
        goal_radius_px = self.goal_radius / (self.x_max - self.x_min) * h_size[0]
        
        draw.ellipse(
            [(goal_px[0] - goal_radius_px, goal_px[1] - goal_radius_px),
             (goal_px[0] + goal_radius_px, goal_px[1] + goal_radius_px)],
            fill=self.colors['goal'],
            width=2 * scale
        )
        
        
        
        # Draw agent using the existing comet function
        agent_pos_px = world_to_pixel(self.state[:2])
        agent_angle = self.state[2]
        
        self._draw_agent(draw, agent_pos_px, agent_angle, scale)
        
        # Resize to target size with high-quality resampling
        img = img.resize(self.image_size, Image.Resampling.LANCZOS)
        
        return np.array(img)
    
    def _draw_agent(self, draw, center_px, angle_rad, scale):
        """
        Draw the agent as a directional shape (comet/arrow).
        
        Args:
            draw: PIL ImageDraw object
            center_px: Agent center in pixel coordinates
            angle_rad: Agent orientation in radians
            scale: Scaling factor for anti-aliasing
        """
        # Flip angle for correct visual orientation
        angle_rad = -angle_rad
        
        # Agent visual parameters (matching dubin_multiobs_render.py)
        length = 17.5 * scale / 2
        width = 10 * scale / 2
        radius = width / 2
        
        # Calculate tip position
        tip_x = center_px[0] + length * math.cos(angle_rad)
        tip_y = center_px[1] + length * math.sin(angle_rad)
        
        # Calculate base corners
        perp_angle = angle_rad + math.pi / 2
        p1_x = center_px[0] + radius * math.cos(perp_angle)
        p1_y = center_px[1] + radius * math.sin(perp_angle)
        p2_x = center_px[0] - radius * math.cos(perp_angle)
        p2_y = center_px[1] - radius * math.sin(perp_angle)
        
        # Draw triangular body
        draw.polygon([(tip_x, tip_y), (p1_x, p1_y), (p2_x, p2_y)], 
                    fill=self.colors['agent'])
        
        # Draw circular base
        start_angle_deg = math.degrees(angle_rad) + 90
        end_angle_deg = math.degrees(angle_rad) - 90
        bounding_box = [
            (center_px[0] - radius, center_px[1] - radius),
            (center_px[0] + radius, center_px[1] + radius)
        ]
        draw.pieslice(bounding_box, start=start_angle_deg, end=end_angle_deg, 
                     fill=self.colors['agent'])
    
    def _display_image(self, img_array: np.ndarray):
        """
        Display image for human rendering mode.
        
        Args:
            img_array: RGB image array to display
        """
        # Implementation depends on your preferred display method
        # Options: matplotlib, opencv, pygame, etc.
        try:
            import matplotlib.pyplot as plt
            plt.imshow(img_array)
            plt.axis('off')
            plt.pause(0.01)
        except ImportError:
            print("Matplotlib not available for human rendering")


# ==================== UTILITY FUNCTIONS ====================

def create_circular_obstacles(obstacles_specs: list) -> list:
    """
    Create circular obstacles from specifications.
    
    Args:
        obstacles_specs: List of obstacle specifications [[x, y, radius], ...]
        
    Returns:
        List of obstacles [(x, y, radius), ...]
    """
    obstacles = []
    for spec in obstacles_specs:
        if len(spec) != 3:
            raise ValueError(f"Each obstacle spec must have 3 elements [x, y, radius], got {len(spec)}")
        x, y, radius = spec
        obstacles.append((x, y, radius))
    
    return obstacles


def test_environment():
    """Test the DubinsEnv environment."""
    import cv2
    
    # Create environment with some obstacles
    obstacles = create_circular_obstacles([[0, 0.65, 0.5], [0, -0.65, 0.5]])
    
    env = DubinsEnv(
        image_size=(128, 128),
        obstacles=obstacles,
        goal_position=(1.3, 0.0),
        render_mode="rgb_array"
    )
    
    # Test reset
    obs, info = env.reset(seed=42)
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial info: {info}")
    
    # Collect observations for video
    observations = [obs]
    
    # Test a few steps
    for i in range(50):  # Increased steps for better video
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        observations.append(obs)
        print(f"Step {i+1}:")
        print(f"  Action: {action}")
        print(f"  Reward: {reward:.3f}")
        print(f"  Terminated: {terminated}")
        print(f"  Info: {info}")
        
        if terminated or truncated:
            break
    
    # Save observations as video
    height, width, channels = observations[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter('test_traj.mp4', fourcc, 10.0, (width, height))
    
    for obs in observations:
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
        video_writer.write(frame_bgr)
    
    video_writer.release()
    print(f"Video saved as 'test_traj.mp4' with {len(observations)} frames")
    
    env.close()


if __name__ == "__main__":
    test_environment()
