"""
Debug script demonstrating deterministic initial states.

This script shows how to use deterministic initial states for debugging
and reproducible experiments.
"""

import sys
import os
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs import get_debug_config, get_default_config
from scripts.run_experiment import run_experiment


def test_deterministic_starts():
    """Test that deterministic starts are actually deterministic."""
    print("Testing Deterministic Initial States")
    print("="*50)
    
    # Test 1: Deterministic flag
    print("\n1. Testing deterministic flag:")
    config = get_debug_config()
    config.experiment.num_test_episodes = 3
    config.experiment.save_video = False
    config.experiment.verbose = False
    
    results = run_experiment(config)
    
    # Check if all episodes started from the same position
    initial_positions = [r['trajectory'][0] for r in results['individual_results']]
    print(f"Initial positions across episodes:")
    for i, pos in enumerate(initial_positions):
        print(f"  Episode {i+1}: ({pos[0]:.3f}, {pos[1]:.3f})")
    
    # Check if they're all the same
    all_same = all(np.allclose(pos, initial_positions[0]) for pos in initial_positions)
    print(f"All episodes started from same position: {all_same}")
    
    # Test 2: Explicit initial state
    print("\n2. Testing explicit initial state:")
    config = get_default_config()
    config.experiment.initial_position = (-1.0, 0.5, np.pi/4)  # Custom start position
    config.experiment.num_test_episodes = 2
    config.experiment.save_video = False
    config.experiment.verbose = False
    
    results = run_experiment(config)
    initial_positions = [r['trajectory'][0] for r in results['individual_results']]
    print(f"Custom initial positions:")
    for i, pos in enumerate(initial_positions):
        print(f"  Episode {i+1}: ({pos[0]:.3f}, {pos[1]:.3f})")
    
    # Test 3: Random starts (for comparison)
    print("\n3. Testing random starts (for comparison):")
    config = get_default_config()
    config.experiment.num_test_episodes = 3
    config.experiment.save_video = False
    config.experiment.verbose = False
    config.experiment.env_seed = None  # Different seed each time
    
    results = run_experiment(config)
    initial_positions = [r['trajectory'][0] for r in results['individual_results']]
    print(f"Random initial positions:")
    for i, pos in enumerate(initial_positions):
        print(f"  Episode {i+1}: ({pos[0]:.3f}, {pos[1]:.3f})")
    
    all_same = all(np.allclose(pos, initial_positions[0]) for pos in initial_positions)
    print(f"All episodes started from same position: {all_same}")


def debug_single_episode():
    """Run a single episode with deterministic start for detailed debugging."""
    print("\n" + "="*50)
    print("SINGLE EPISODE DEBUG")
    print("="*50)
    
    config = get_debug_config()
    config.experiment.num_test_episodes = 1
    config.experiment.max_test_steps = 50
    config.experiment.log_interval = 5  # Log every 5 steps
    config.experiment.verbose = True
    config.experiment.save_video = True
    
    print("Running single episode with deterministic start...")
    print(f"Deterministic position: (-1.2, 0.0, 0.0)")
    print(f"Goal position: {config.environment.goal_position}")
    print(f"Obstacles: {len(config.environment.obstacles)}")
    
    results = run_experiment(config)
    
    trajectory = results['individual_results'][0]['trajectory']
    actions = results['individual_results'][0]['actions']
    
    print(f"\nTrajectory summary:")
    print(f"  Start: ({trajectory[0][0]:.3f}, {trajectory[0][1]:.3f})")
    print(f"  End: ({trajectory[-1][0]:.3f}, {trajectory[-1][1]:.3f})")
    print(f"  Steps: {len(trajectory)-1}")
    print(f"  Success: {results['individual_results'][0]['success']}")
    print(f"  Final distance to goal: {results['individual_results'][0]['final_distance']:.3f}")
    
    # Show first few actions
    print(f"\nFirst 10 actions:")
    for i, action in enumerate(actions[:10]):
        print(f"  Step {i+1}: {action:.3f}")


def compare_deterministic_vs_random():
    """Compare performance between deterministic and random starts."""
    print("\n" + "="*50)
    print("DETERMINISTIC vs RANDOM COMPARISON")
    print("="*50)
    
    # Deterministic starts
    print("Testing with deterministic starts...")
    config_det = get_debug_config()
    config_det.experiment.num_test_episodes = 5
    config_det.experiment.save_video = False
    config_det.experiment.verbose = False
    
    results_det = run_experiment(config_det)
    
    # Random starts
    print("Testing with random starts...")
    config_rand = get_default_config()
    config_rand.experiment.num_test_episodes = 5
    config_rand.experiment.save_video = False
    config_rand.experiment.verbose = False
    
    results_rand = run_experiment(config_rand)
    
    # Compare results
    print(f"\nComparison Results:")
    print(f"Deterministic starts:")
    print(f"  Success rate: {results_det['success_rate']:.1%}")
    print(f"  Average steps: {results_det['average_steps']:.1f}")
    print(f"  Average reward: {results_det['average_reward']:.1f}")
    print(f"  Std dev steps: {np.std([r['steps'] for r in results_det['individual_results']]):.1f}")
    
    print(f"\nRandom starts:")
    print(f"  Success rate: {results_rand['success_rate']:.1%}")
    print(f"  Average steps: {results_rand['average_steps']:.1f}")
    print(f"  Average reward: {results_rand['average_reward']:.1f}")
    print(f"  Std dev steps: {np.std([r['steps'] for r in results_rand['individual_results']]):.1f}")


if __name__ == "__main__":
    # Test deterministic functionality
    test_deterministic_starts()
    
    # Run detailed single episode
    debug_single_episode()
    
    # Compare deterministic vs random
    compare_deterministic_vs_random()
