"""
Model Predictive Control (MPC) Controller for DubinsEnv

This controller uses MPC to plan optimal trajectories for the Dubins car
while avoiding obstacles and reaching the goal. It solves an optimization
problem at each timestep to find the best control sequence.
"""

import numpy as np
import math
from typing import Dict, Any, Tuple, List, Optional
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


class MPCController:
    """
    Model Predictive Control controller for the Dubins car environment.
    
    The MPC formulation includes:
    - Dubins car dynamics constraints
    - Obstacle avoidance constraints
    - Goal reaching objective
    - Control input constraints
    """
    
    def __init__(self,
                 prediction_horizon: int = 10,
                 control_horizon: int = 5,
                 dt: float = 0.05,
                 speed: float = 1.0,
                 max_angular_velocity: float = 3.3,
                 goal_weight: float = 10.0,
                 obstacle_weight: float = 50.0,
                 control_weight: float = 0.1,
                 obstacle_safety_margin: float = 0.3,
                 goal_tolerance: float = 0.1):
        """
        Initialize the MPC controller.
        
        Args:
            prediction_horizon: Number of steps to predict ahead
            control_horizon: Number of control steps to optimize (≤ prediction_horizon)
            dt: Time step for predictions
            speed: Constant forward speed of Dubins car
            max_angular_velocity: Maximum angular velocity
            goal_weight: Weight for goal reaching cost
            obstacle_weight: Weight for obstacle avoidance cost
            control_weight: Weight for control effort regularization
            obstacle_safety_margin: Safety margin around obstacles
            goal_tolerance: Distance tolerance for goal reaching
        """
        self.N = prediction_horizon
        self.M = min(control_horizon, prediction_horizon)
        self.dt = dt
        self.speed = speed
        self.max_omega = max_angular_velocity
        
        # Cost weights
        self.w_goal = goal_weight
        self.w_obstacle = obstacle_weight
        self.w_control = control_weight
        
        # Constraints
        self.safety_margin = obstacle_safety_margin
        self.goal_tol = goal_tolerance
        
        # Optimization state
        self.prev_solution = None
        self.warm_start = True
        
    def predict_trajectory(self, initial_state: np.ndarray, control_sequence: np.ndarray) -> np.ndarray:
        """
        Predict the trajectory given initial state and control sequence.
        
        Args:
            initial_state: Initial state [x, y, theta]
            control_sequence: Control sequence [omega_0, omega_1, ..., omega_{N-1}]
            
        Returns:
            Predicted trajectory of shape (N+1, 3) including initial state
        """
        trajectory = np.zeros((self.N + 1, 3))
        trajectory[0] = initial_state
        
        for i in range(self.N):
            x, y, theta = trajectory[i]
            
            # Use control input (extend last control if sequence is shorter)
            if i < len(control_sequence):
                omega = control_sequence[i]
            else:
                omega = control_sequence[-1] if len(control_sequence) > 0 else 0.0
            
            # Dubins car dynamics
            x_next = x + self.speed * np.cos(theta) * self.dt
            y_next = y + self.speed * np.sin(theta) * self.dt
            theta_next = theta + omega * self.dt
            
            # Normalize angle
            theta_next = np.arctan2(np.sin(theta_next), np.cos(theta_next))
            
            trajectory[i + 1] = [x_next, y_next, theta_next]
        
        return trajectory
    
    def compute_cost(self, control_sequence: np.ndarray, 
                    initial_state: np.ndarray, 
                    goal_pos: np.ndarray, 
                    obstacles: List[Tuple[float, float, float]]) -> float:
        """
        Compute the total cost for a given control sequence.
        
        Args:
            control_sequence: Control inputs [omega_0, ..., omega_{M-1}]
            initial_state: Current state [x, y, theta]
            goal_pos: Goal position [x_goal, y_goal]
            obstacles: List of obstacles [(x, y, radius), ...]
            
        Returns:
            Total cost
        """
        # Extend control sequence for prediction horizon
        extended_controls = np.zeros(self.N)
        extended_controls[:len(control_sequence)] = control_sequence
        if len(control_sequence) < self.N:
            extended_controls[len(control_sequence):] = control_sequence[-1] if len(control_sequence) > 0 else 0.0
        
        # Predict trajectory
        trajectory = self.predict_trajectory(initial_state, extended_controls)
        
        total_cost = 0.0
        
        # Goal reaching cost
        for i in range(1, self.N + 1):
            pos = trajectory[i, :2]
            goal_distance = np.linalg.norm(pos - goal_pos)
            total_cost += self.w_goal * goal_distance**2
        
        # Terminal goal cost (higher weight)
        final_pos = trajectory[-1, :2]
        final_goal_distance = np.linalg.norm(final_pos - goal_pos)
        total_cost += self.w_goal * 5.0 * final_goal_distance**2
        
        # Obstacle avoidance cost
        for i in range(1, self.N + 1):
            pos = trajectory[i, :2]
            
            for obs_x, obs_y, obs_radius in obstacles:
                obs_pos = np.array([obs_x, obs_y])
                distance = np.linalg.norm(pos - obs_pos)
                safe_distance = obs_radius + self.safety_margin
                
                if distance < safe_distance:
                    # Exponential penalty for being too close
                    violation = safe_distance - distance
                    total_cost += self.w_obstacle * np.exp(violation * 5.0)
                else:
                    # Soft constraint to stay away from obstacles
                    margin_violation = max(0, safe_distance * 1.5 - distance)
                    total_cost += self.w_obstacle * 0.01 * margin_violation**2
        
        # Control effort regularization
        for i in range(len(control_sequence)):
            total_cost += self.w_control * control_sequence[i]**2
        
        # Control smoothness (penalize large changes)
        for i in range(1, len(control_sequence)):
            control_change = control_sequence[i] - control_sequence[i-1]
            total_cost += self.w_control * 0.5 * control_change**2
        
        return total_cost
    
    def compute_action(self, info: Dict[str, Any], obstacles: List[Tuple[float, float, float]], 
                      goal_pos: Optional[np.ndarray] = None) -> float:
        """
        Compute the optimal control action using MPC.
        
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
        
        # Initial guess for control sequence
        if self.prev_solution is not None and self.warm_start and len(self.prev_solution) >= self.M:
            # Warm start: shift previous solution and add zero at the end
            initial_guess = np.zeros(self.M)
            initial_guess[:-1] = self.prev_solution[1:self.M]
            initial_guess[-1] = self.prev_solution[-1] if len(self.prev_solution) > 0 else 0.0
        else:
            # Cold start: use simple heuristic
            goal_direction = goal_pos - current_state[:2]
            goal_angle = np.arctan2(goal_direction[1], goal_direction[0])
            angle_error = np.arctan2(np.sin(goal_angle - current_state[2]), 
                                   np.cos(goal_angle - current_state[2]))
            initial_omega = np.clip(2.0 * angle_error, -self.max_omega, self.max_omega)
            initial_guess = np.full(self.M, initial_omega * 0.5)
        
        # Define objective function
        def objective(u):
            return self.compute_cost(u, current_state, goal_pos, obstacles)
        
        # Control constraints
        bounds = [(-self.max_omega, self.max_omega) for _ in range(self.M)]
        
        # Solve optimization problem
        try:
            result = minimize(
                objective,
                initial_guess,
                method='L-BFGS-B',
                bounds=bounds,
                options={
                    'maxiter': 100,
                    'ftol': 1e-6,
                    'gtol': 1e-6
                }
            )
            
            if result.success:
                optimal_controls = result.x
            else:
                # Fallback to initial guess if optimization fails
                optimal_controls = initial_guess
                
        except Exception as e:
            print(f"MPC optimization failed: {e}")
            # Fallback to simple proportional control
            goal_direction = goal_pos - current_state[:2]
            goal_angle = np.arctan2(goal_direction[1], goal_direction[0])
            angle_error = np.arctan2(np.sin(goal_angle - current_state[2]), 
                                   np.cos(goal_angle - current_state[2]))
            optimal_controls = np.array([np.clip(2.0 * angle_error, -self.max_omega, self.max_omega)])
        
        # Store solution for warm starting
        self.prev_solution = optimal_controls.copy()
        
        # Return first control action
        return float(optimal_controls[0])
    
    def get_predicted_trajectory(self, info: Dict[str, Any], 
                               goal_pos: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Get the predicted trajectory from the last MPC solution.
        
        Args:
            info: Info dictionary from environment step
            goal_pos: Goal position
            
        Returns:
            Predicted trajectory of shape (N+1, 3)
        """
        current_state = np.array([
            info['agent_position'][0],
            info['agent_position'][1], 
            info['agent_orientation']
        ])
        
        if self.prev_solution is not None:
            return self.predict_trajectory(current_state, self.prev_solution)
        else:
            # Return current state only if no solution available
            return current_state.reshape(1, -1)
    
    def reset(self):
        """Reset the controller state (e.g., for new episode)."""
        self.prev_solution = None


def test_mpc_controller():
    """Test the MPC controller with the DubinsEnv."""
    import sys
    sys.path.append('/home/kensuke/WM_CBF/wm_cbf_dubin')
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
    
    # Create MPC controller
    controller = MPCController(
        prediction_horizon=15,
        control_horizon=8,
        dt=0.05,
        speed=1.0,
        max_angular_velocity=3.3,
        goal_weight=10.0,
        obstacle_weight=100.0,
        control_weight=0.1,
        obstacle_safety_margin=0.25
    )
    
    # Test episode
    obs, info = env.reset(seed=42, options={'initial_state': config.experiment.initial_position})
    observations = [obs]
    
    print("Testing MPC controller...")
    print(f"Initial position: {info['agent_position']}, orientation: {info['agent_orientation']:.3f}")
    print(f"Goal position: {config.environment.goal_position}")
    print(f"Initial goal distance: {info['goal_distance']:.3f}")
    
    for step in range(100):
        # Use MPC controller to compute action
        action = controller.compute_action(info, obstacles, np.array(config.environment.goal_position))
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        observations.append(obs)
        
        if step % 10 == 0 or step < 5:
            print(f"Step {step+1:3d}: pos=({info['agent_position'][0]:5.2f}, {info['agent_position'][1]:5.2f}), "
                  f"θ={info['agent_orientation']:5.2f}, goal_dist={info['goal_distance']:.3f}, "
                  f"action={action:5.2f}, reward={reward:6.2f}")
        
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
        video_writer = cv2.VideoWriter('mpc_test.mp4', fourcc, 10.0, (width, height))
        
        for obs in observations:
            frame_bgr = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)
        
        video_writer.release()
        print(f"Video saved as 'mpc_test.mp4' with {len(observations)} frames")
    
    env.close()


def compare_controllers():
    """Compare MPC controller with the simple controller."""
    import sys
    sys.path.append('/home/kensuke/WM_CBF/wm_cbf_dubin')
    from .simple_controller import SimpleController
    from dubins_env_outline import DubinsEnv
    from config import get_debug_config
    
    print("Comparing MPC vs Simple Controller")
    print("="*50)
    
    config = get_debug_config()
    obstacles = config.environment.get_obstacles_list()
    
    # Test both controllers
    controllers = {
        "MPC": MPCController(
            prediction_horizon=12,
            control_horizon=6,
            goal_weight=10.0,
            obstacle_weight=100.0,
            obstacle_safety_margin=0.25
        ),
        "Simple": SimpleController(
            goal_attraction_gain=2.0,
            obstacle_repulsion_gain=4.0,
            obstacle_influence_radius=0.8,
            lookahead_distance=0.4
        )
    }
    
    results = {}
    
    for name, controller in controllers.items():
        print(f"\nTesting {name} controller...")
        
        env = DubinsEnv(
            image_size=(128, 128),
            obstacles=obstacles,
            goal_position=config.environment.goal_position,
            render_mode=None
        )
        
        # Run multiple episodes
        episode_results = []
        for episode in range(3):
            obs, info = env.reset(seed=42+episode, options={'initial_state': config.experiment.initial_position})
            
            if hasattr(controller, 'reset'):
                controller.reset()
            
            steps = 0
            for step in range(150):
                if name == "MPC":
                    action = controller.compute_action(info, obstacles, np.array(config.environment.goal_position))
                else:
                    action = controller.compute_action(info, obstacles)
                
                obs, reward, terminated, truncated, info = env.step(action)
                steps += 1
                
                if terminated or truncated:
                    break
            
            episode_results.append({
                'success': info.get('goal_reached', False),
                'steps': steps,
                'final_distance': info['goal_distance']
            })
        
        env.close()
        
        # Compute statistics
        success_rate = sum(1 for r in episode_results if r['success']) / len(episode_results)
        avg_steps = np.mean([r['steps'] for r in episode_results])
        avg_distance = np.mean([r['final_distance'] for r in episode_results])
        
        results[name] = {
            'success_rate': success_rate,
            'avg_steps': avg_steps,
            'avg_distance': avg_distance
        }
        
        print(f"{name}: Success={success_rate:.1%}, Steps={avg_steps:.1f}, Final Dist={avg_distance:.3f}")
    
    print("\n" + "="*50)
    print("COMPARISON SUMMARY")
    print("="*50)
    for name, result in results.items():
        print(f"{name:10}: {result['success_rate']:5.1%} success, "
              f"{result['avg_steps']:5.1f} steps, "
              f"{result['avg_distance']:5.3f} final distance")


if __name__ == "__main__":
    print("Testing MPC Controller")
    test_mpc_controller()
    
    print("\n" + "="*80 + "\n")
    
    print("Comparing Controllers")
    compare_controllers()
