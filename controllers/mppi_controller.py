"""
Model Predictive Path Integral (MPPI) Controller

MPPI is a stochastic optimal control algorithm that samples multiple trajectories,
weights them by their cost, and computes a weighted average control sequence.
It's particularly effective for handling uncertainty and generating robust policies.
"""

import numpy as np
import math
from typing import Dict, Any, List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


class MPPIController:
    """
    Model Predictive Path Integral (MPPI) controller for the Dubins car environment.
    
    MPPI samples multiple control sequences, evaluates their costs, and computes
    a weighted average that approximates the optimal control distribution.
    """
    
    def __init__(self,
                 prediction_horizon: int = 10,
                 num_samples: int = 1000,
                 dt: float = 0.05,
                 speed: float = 1.0,
                 max_angular_velocity: float = 3.3,
                 temperature: float = 1.0,
                 lambda_param: float = 1.0,
                 goal_weight: float = 10.0,
                 obstacle_weight: float = 50.0,
                 control_weight: float = 0.1,
                 obstacle_safety_margin: float = 0.3,
                 goal_tolerance: float = 0.1,
                 noise_variance: float = 1.0,
                 warm_start: bool = True,
                 adaptive_temperature: bool = False):
        """
        Initialize the MPPI controller.
        
        Args:
            prediction_horizon: Number of steps to predict ahead
            num_samples: Number of control sequence samples to generate
            dt: Time step for integration
            speed: Constant forward speed of Dubins car
            max_angular_velocity: Maximum angular velocity
            temperature: Temperature parameter for importance weighting (lower = more greedy)
            lambda_param: Regularization parameter for control cost
            goal_weight: Weight for goal reaching cost
            obstacle_weight: Weight for obstacle avoidance cost
            control_weight: Weight for control effort regularization
            obstacle_safety_margin: Safety margin around obstacles
            goal_tolerance: Distance tolerance for goal reaching
            noise_variance: Variance of control noise for sampling
            warm_start: Whether to use previous solution for initialization
            adaptive_temperature: Whether to adapt temperature based on performance
        """
        self.T = prediction_horizon
        self.K = num_samples
        self.dt = dt
        self.speed = speed
        self.max_omega = max_angular_velocity
        
        # MPPI parameters
        self.temperature = temperature
        self.lambda_param = lambda_param
        self.adaptive_temperature = adaptive_temperature
        
        # Cost weights
        self.w_goal = goal_weight
        self.w_obstacle = obstacle_weight
        self.w_control = control_weight
        
        # Constraints
        self.safety_margin = obstacle_safety_margin
        self.goal_tol = goal_tolerance
        
        # Noise and sampling
        self.noise_variance = noise_variance
        self.warm_start = warm_start
        
        # MPPI state
        self.prev_controls = None
        self.prev_weights = None
        self.temperature_history = []
        self.performance_history = []
        
        # Adaptive parameters
        self.min_temperature = 0.1
        self.max_temperature = 10.0
        self.temperature_decay = 0.99
        self.performance_window = 10
        
    def sample_control_sequences(self, initial_state: np.ndarray, 
                                prev_controls: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Sample control sequences for MPPI (fully vectorized).
        
        Args:
            initial_state: Current state [x, y, theta]
            prev_controls: Previous control sequence for warm starting
            
        Returns:
            Control sequences of shape (K, T) where K is num_samples, T is horizon
        """
        if prev_controls is not None and self.warm_start:
            # Warm start: use previous solution as mean
            if len(prev_controls) >= self.T:
                mean_controls = prev_controls[:self.T]
            else:
                # Pad with last control
                mean_controls = np.full(self.T, prev_controls[-1] if len(prev_controls) > 0 else 0.0)
        else:
            # Cold start: use simple heuristic
            goal_direction = self._get_goal_direction(initial_state)
            heuristic_control = self._compute_heuristic_control(initial_state, goal_direction)
            mean_controls = np.full(self.T, heuristic_control)
        
        # Vectorized sampling: generate all samples at once
        noise = np.random.normal(0, np.sqrt(self.noise_variance), (self.K, self.T))
        control_sequences = mean_controls[np.newaxis, :] + noise
        
        # Vectorized clipping to valid range
        control_sequences = np.clip(control_sequences, -self.max_omega, self.max_omega)
        
        return control_sequences
    
    def predict_trajectories(self, initial_state: np.ndarray, 
                           control_sequences: np.ndarray) -> np.ndarray:
        """
        Predict trajectories for all control sequences (vectorized for speed).
        
        Args:
            initial_state: Initial state [x, y, theta]
            control_sequences: Control sequences of shape (K, T)
            
        Returns:
            Trajectories of shape (K, T+1, 3)
        """
        K, T = control_sequences.shape
        trajectories = np.zeros((K, T + 1, 3))
        trajectories[:, 0] = initial_state
        
        # Vectorized trajectory prediction
        for t in range(T):
            # Current states for all samples
            current_states = trajectories[:, t]  # Shape: (K, 3)
            x, y, theta = current_states[:, 0], current_states[:, 1], current_states[:, 2]
            
            # Current controls for all samples
            omega = control_sequences[:, t]  # Shape: (K,)
            
            # Vectorized Dubins car dynamics
            x_new = x + self.speed * np.cos(theta) * self.dt
            y_new = y + self.speed * np.sin(theta) * self.dt
            theta_new = theta + omega * self.dt
            
            # Normalize angles (vectorized)
            theta_new = np.arctan2(np.sin(theta_new), np.cos(theta_new))
            
            # Update trajectories
            trajectories[:, t + 1, 0] = x_new
            trajectories[:, t + 1, 1] = y_new
            trajectories[:, t + 1, 2] = theta_new
        
        return trajectories
    
    def compute_trajectory_costs(self, trajectories: np.ndarray, 
                               control_sequences: np.ndarray,
                               goal_pos: np.ndarray,
                               obstacles: List[Tuple[float, float, float]]) -> np.ndarray:
        """
        Compute costs for all trajectories (fully vectorized for speed).
        
        Args:
            trajectories: Predicted trajectories of shape (K, T+1, 3)
            control_sequences: Control sequences of shape (K, T)
            goal_pos: Goal position [x_goal, y_goal]
            obstacles: List of obstacles [(x, y, radius), ...]
            
        Returns:
            Costs for each trajectory of shape (K,)
        """
        K, T_plus_1, _ = trajectories.shape
        T = T_plus_1 - 1
        
        # Initialize cost arrays
        goal_costs = np.zeros(K)
        obstacle_costs = np.zeros(K)
        control_costs = np.zeros(K)
        mppi_costs = np.zeros(K)
        
        # 1. Goal reaching cost (vectorized)
        # All positions except initial state
        all_positions = trajectories[:, 1:, :2]  # Shape: (K, T, 2)
        
        # Distance to goal for all positions
        goal_distances = np.linalg.norm(all_positions - goal_pos[np.newaxis, np.newaxis, :], axis=2)  # Shape: (K, T)
        
        # Sum over time steps
        goal_costs += self.w_goal * np.sum(goal_distances**2, axis=1)
        
        # Terminal goal cost (higher weight)
        final_positions = trajectories[:, -1, :2]  # Shape: (K, 2)
        final_goal_distances = np.linalg.norm(final_positions - goal_pos, axis=1)  # Shape: (K,)
        goal_costs += self.w_goal * 5.0 * final_goal_distances**2
        
        # 2. Obstacle avoidance cost (vectorized)
        if obstacles:
            for obs_x, obs_y, obs_radius in obstacles:
                obs_pos = np.array([obs_x, obs_y])
                safe_distance = obs_radius + self.safety_margin
                
                # Distance to obstacle for all positions
                obs_distances = np.linalg.norm(all_positions - obs_pos[np.newaxis, np.newaxis, :], axis=2)  # Shape: (K, T)
                
                # Violation mask
                violation_mask = obs_distances < safe_distance
                
                # Exponential penalty for violations
                violations = safe_distance - obs_distances
                exp_penalties = self.w_obstacle * np.exp(violations * 5.0)
                exp_penalties = np.where(violation_mask, exp_penalties, 0)
                
                # Soft constraint for non-violations
                soft_violations = np.maximum(0, safe_distance * 1.5 - obs_distances)
                soft_penalties = self.w_obstacle * 1. * soft_violations**2
                soft_penalties = np.where(~violation_mask, soft_penalties, 0)
                
                # Sum over time steps
                obstacle_costs += np.sum(exp_penalties + soft_penalties, axis=1)
        
        # 3. Control effort cost (vectorized)
        control_costs += self.w_control * np.sum(control_sequences**2, axis=1)
        
        # 4. Control smoothness cost (vectorized)
        if T > 1:
            control_changes = control_sequences[:, 1:] - control_sequences[:, :-1]  # Shape: (K, T-1)
            control_costs += self.w_control * 0.5 * np.sum(control_changes**2, axis=1)
        
        # 5. MPPI control cost (vectorized)
        if self.prev_controls is not None and self.lambda_param > 0:
            if len(self.prev_controls) >= T:
                prev_seq = self.prev_controls[:T]
            else:
                prev_seq = np.full(T, self.prev_controls[-1] if len(self.prev_controls) > 0 else 0.0)
            
            control_deviations = control_sequences - prev_seq[np.newaxis, :]  # Shape: (K, T)
            mppi_costs += self.lambda_param * np.sum(control_deviations**2, axis=1)
        
        # Total cost
        total_costs = goal_costs + obstacle_costs + control_costs + mppi_costs
        
        return total_costs
    
    def compute_weights(self, costs: np.ndarray) -> np.ndarray:
        """
        Compute importance weights for control sequences.
        
        Args:
            costs: Trajectory costs of shape (K,)
            
        Returns:
            Importance weights of shape (K,)
        """
        # Shift costs for numerical stability
        min_cost = np.min(costs)
        shifted_costs = costs - min_cost
        
        # Compute weights using softmax with temperature
        exp_costs = np.exp(-shifted_costs / self.temperature)
        weights = exp_costs / np.sum(exp_costs)
        
        # Handle numerical issues
        weights = np.nan_to_num(weights, nan=1.0/self.K)
        
        return weights
    
    def update_temperature(self, weights: np.ndarray, performance: float):
        """
        Adaptively update temperature based on performance.
        
        Args:
            weights: Current importance weights
            performance: Current performance metric (e.g., success rate)
        """
        if not self.adaptive_temperature:
            return
        
        self.temperature_history.append(self.temperature)
        self.performance_history.append(performance)
        
        # Keep only recent history
        if len(self.temperature_history) > self.performance_window:
            self.temperature_history = self.temperature_history[-self.performance_window:]
            self.performance_history = self.performance_history[-self.performance_window:]
        
        # Adapt temperature based on weight entropy
        if len(weights) > 1:
            entropy = -np.sum(weights * np.log(weights + 1e-10))
            max_entropy = np.log(len(weights))
            normalized_entropy = entropy / max_entropy
            
            # If weights are too concentrated (low entropy), increase temperature
            # If weights are too uniform (high entropy), decrease temperature
            target_entropy = 0.5  # Target 50% of max entropy
            
            if normalized_entropy < target_entropy - 0.1:
                self.temperature *= 1.05  # Increase exploration
            elif normalized_entropy > target_entropy + 0.1:
                self.temperature *= 0.95  # Decrease exploration
            
            # Clamp temperature
            self.temperature = np.clip(self.temperature, self.min_temperature, self.max_temperature)
    
    def compute_action(self, info: Dict[str, Any], obstacles: List[Tuple[float, float, float]], 
                      goal_pos: Optional[np.ndarray] = None) -> float:
        """
        Compute the optimal control action using MPPI.
        
        Args:
            info: Info dictionary from environment step
            obstacles: List of obstacles [(x, y, radius), ...]
            goal_pos: Goal position (if None, uses default from environment)
            
        Returns:
            Optimal angular velocity command (scalar)
        """
        # Extract current state
        current_state = np.array([
            info['agent_position'][0],
            info['agent_position'][1], 
            info['agent_orientation']
        ])
        
        # Set goal position
        if goal_pos is None:
            goal_pos = np.array([1.3, 0.0])  # Default goal from your setup
        
        # Check if already at goal
        if np.linalg.norm(current_state[:2] - goal_pos) < self.goal_tol:
            return 0.0
        
        # Sample control sequences
        control_sequences = self.sample_control_sequences(current_state, self.prev_controls)
        
        # Predict trajectories
        trajectories = self.predict_trajectories(current_state, control_sequences)
        
        # Compute costs
        costs = self.compute_trajectory_costs(trajectories, control_sequences, goal_pos, obstacles)
        
        # Compute weights
        weights = self.compute_weights(costs)
        
        # Store for next iteration
        self.prev_weights = weights.copy()
        
        # Compute weighted average control sequence
        weighted_controls = np.sum(weights[:, np.newaxis] * control_sequences, axis=0)
        
        # Store for warm starting next iteration
        self.prev_controls = weighted_controls.copy()
        
        # Update temperature adaptively
        success_rate = self._estimate_success_rate(trajectories, goal_pos)
        self.update_temperature(weights, success_rate)
        
        # Return first control action
        return float(weighted_controls[0])
    
    def _get_goal_direction(self, state: np.ndarray) -> np.ndarray:
        """Get direction to goal."""
        goal_pos = np.array([1.3, 0.0])  # Default goal
        return goal_pos - state[:2]
    
    def _compute_heuristic_control(self, state: np.ndarray, goal_direction: np.ndarray) -> float:
        """Compute heuristic control for warm starting."""
        goal_angle = np.arctan2(goal_direction[1], goal_direction[0])
        current_angle = state[2]
        
        # Angle error
        angle_error = np.arctan2(np.sin(goal_angle - current_angle), 
                               np.cos(goal_angle - current_angle))
        
        # Simple proportional control
        control = 2.0 * angle_error
        return np.clip(control, -self.max_omega, self.max_omega)
    
    def _estimate_success_rate(self, trajectories: np.ndarray, goal_pos: np.ndarray) -> float:
        """Estimate success rate from trajectory predictions."""
        final_positions = trajectories[:, -1, :2]
        final_distances = np.linalg.norm(final_positions - goal_pos, axis=1)
        successes = np.sum(final_distances <= self.goal_tol)
        return float(successes) / len(trajectories)
    
    def get_predicted_trajectory(self, info: Dict[str, Any], 
                               goal_pos: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Get the predicted trajectory from the last MPPI solution.
        
        Args:
            info: Info dictionary from environment step
            goal_pos: Goal position
            
        Returns:
            Predicted trajectory of shape (T+1, 3)
        """
        current_state = np.array([
            info['agent_position'][0],
            info['agent_position'][1], 
            info['agent_orientation']
        ])
        
        if self.prev_controls is not None:
            # Use previous solution
            control_seq = self.prev_controls.reshape(1, -1)
            trajectory = self.predict_trajectories(current_state, control_seq)
            return trajectory[0]  # Return first (and only) trajectory
        else:
            # Return current state only if no solution available
            return current_state.reshape(1, -1)
    
    def get_sample_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the sampling process.
        
        Returns:
            Dictionary with sampling statistics
        """
        stats = {
            'temperature': self.temperature,
            'num_samples': self.K,
            'horizon': self.T,
            'noise_variance': self.noise_variance
        }
        
        if self.prev_weights is not None:
            stats.update({
                'weight_entropy': -np.sum(self.prev_weights * np.log(self.prev_weights + 1e-10)),
                'max_weight': np.max(self.prev_weights),
                'min_weight': np.min(self.prev_weights),
                'weight_std': np.std(self.prev_weights)
            })
        
        if len(self.temperature_history) > 0:
            stats['avg_temperature'] = np.mean(self.temperature_history[-10:])
        
        return stats
    
    def reset(self):
        """Reset the controller state (e.g., for new episode)."""
        self.prev_controls = None
        self.prev_weights = None
        self.temperature_history = []
        self.performance_history = []


def test_mppi_controller():
    """Test the MPPI controller with the DubinsEnv."""
    import sys
    sys.path.append('/home/kensuke/WM_CBF/on_policy')
    from dubins_env_outline import DubinsEnv, create_circular_obstacles
    from config import get_debug_config
    import cv2
    
    # Create environment
    config = get_debug_config()
    obstacles = config.environment.get_obstacles_list()
    
    env = DubinsEnv(
        image_size=(128, 128),
        obstacles=obstacles,
        goal_position=config.environment.goal_position,
        render_mode="rgb_array"
    )
    
    # Create MPPI controller
    controller = MPPIController(
        prediction_horizon=12,
        num_samples=500,
        dt=0.05,
        speed=1.0,
        max_angular_velocity=3.3,
        temperature=1.0,
        lambda_param=1.0,
        goal_weight=10.0,
        obstacle_weight=100.0,
        control_weight=0.1,
        obstacle_safety_margin=0.25,
        adaptive_temperature=True
    )
    
    # Test episode
    obs, info = env.reset(seed=42, options={'initial_state': config.experiment.initial_position})
    observations = [obs]
    
    print("Testing MPPI controller...")
    print(f"Initial position: {info['agent_position']}, orientation: {info['agent_orientation']:.3f}")
    print(f"Goal position: {config.environment.goal_position}")
    print(f"Initial goal distance: {info['goal_distance']:.3f}")
    
    for step in range(100):
        # Use MPPI controller to compute action
        action = controller.compute_action(info, obstacles, np.array(config.environment.goal_position))
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        observations.append(obs)
        
        if step % 10 == 0 or step < 5:
            stats = controller.get_sample_statistics()
            print(f"Step {step+1:3d}: pos=({info['agent_position'][0]:5.2f}, {info['agent_position'][1]:5.2f}), "
                  f"θ={info['agent_orientation']:5.2f}, goal_dist={info['goal_distance']:.3f}, "
                  f"action={action:5.2f}, reward={reward:6.2f}, temp={stats['temperature']:.2f}")
        
        if terminated or truncated:
            if info['goal_reached']:
                print(f"🎉 Goal reached in {step+1} steps!")
            elif info['collision']:
                print(f"💥 Collision occurred at step {step+1}")
            else:
                print(f"🚫 Episode ended at step {step+1}")
            break
    
    # Print final statistics
    final_stats = controller.get_sample_statistics()
    print(f"\\nFinal MPPI Statistics:")
    print(f"Temperature: {final_stats['temperature']:.3f}")
    print(f"Sample entropy: {final_stats.get('weight_entropy', 0):.3f}")
    print(f"Max weight: {final_stats.get('max_weight', 0):.3f}")
    
    # Save video
    if len(observations) > 1:
        height, width, channels = observations[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter('mppi_test.mp4', fourcc, 10.0, (width, height))
        
        for obs in observations:
            frame_bgr = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)
        
        video_writer.release()
        print(f"Video saved as 'mppi_test.mp4' with {len(observations)} frames")
    
    env.close()


if __name__ == "__main__":
    print("Testing MPPI Controller")
    test_mppi_controller()
