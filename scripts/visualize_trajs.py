"""
Trajectory Visualization Script

Loads trajectories from HDF5 files and creates various visualizations:
- 2D trajectory plots
- Animated trajectories
- Performance statistics
- Action sequences
- State evolution over time
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
import seaborn as sns
import h5py
import os
import argparse
from typing import List, Dict, Any, Tuple, Optional
import cv2

from collect_trajs import load_trajectories_from_hdf5


class TrajectoryVisualizer:
    """
    Visualizes trajectories collected from the Dubins car environment.
    """
    
    def __init__(self, trajectories: List[Dict[str, Any]], metadata: Dict[str, Any]):
        """
        Initialize visualizer with trajectory data.
        
        Args:
            trajectories: List of trajectory dictionaries
            metadata: Metadata from HDF5 file
        """
        self.trajectories = trajectories
        self.metadata = metadata
        self.env_info = metadata['environment_info']
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Extract environment parameters
        self.world_bounds = self.env_info['world_bounds']  # (x_min, x_max, y_min, y_max)
        self.goal_pos = np.array(self.env_info['goal_position'])
        self.obstacles = self.env_info['obstacles']
        
    def plot_environment(self, ax, alpha=0.7):
        """
        Plot the environment (obstacles, goal, boundaries).
        
        Args:
            ax: Matplotlib axis
            alpha: Transparency for environment elements
        """
        x_min, x_max, y_min, y_max = self.world_bounds
        
        # Set axis limits with some padding
        padding = 0.1
        ax.set_xlim(x_min - padding, x_max + padding)
        ax.set_ylim(y_min - padding, y_max + padding)
        
        # Draw world boundaries
        boundary = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                   linewidth=2, edgecolor='black', facecolor='none')
        ax.add_patch(boundary)
        
        # Draw obstacles
        for obs_x, obs_y, obs_radius in self.obstacles:
            obstacle = patches.Circle((obs_x, obs_y), obs_radius, 
                                    facecolor='red', alpha=alpha, edgecolor='darkred')
            ax.add_patch(obstacle)
        
        # Draw goal
        goal = patches.Circle(self.goal_pos, 0.1, facecolor='green', alpha=alpha, 
                            edgecolor='darkgreen', linewidth=2)
        ax.add_patch(goal)
        
        # Add labels
        ax.text(self.goal_pos[0], self.goal_pos[1] + 0.15, 'Goal', 
                ha='center', va='bottom', fontweight='bold')
        
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
    
    def plot_single_trajectory(self, traj_idx: int, show_arrows: bool = True, 
                             show_actions: bool = False, figsize: Tuple[int, int] = (12, 8)):
        """
        Plot a single trajectory with detailed information.
        
        Args:
            traj_idx: Index of trajectory to plot
            show_arrows: Whether to show direction arrows
            show_actions: Whether to show action subplot
            figsize: Figure size
        """
        if traj_idx >= len(self.trajectories):
            raise ValueError(f"Trajectory index {traj_idx} out of range (0-{len(self.trajectories)-1})")
        
        traj = self.trajectories[traj_idx]
        states = traj['states']
        actions = traj['actions']
        
        # Create figure
        if show_actions:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(8, 8))
        
        # Plot environment
        self.plot_environment(ax1)
        
        # Plot trajectory
        positions = states[:, :2]
        orientations = states[:, 2]
        
        # Color trajectory by time
        colors = plt.cm.viridis(np.linspace(0, 1, len(positions)))
        
        for i in range(len(positions) - 1):
            ax1.plot(positions[i:i+2, 0], positions[i:i+2, 1], 
                    color=colors[i], linewidth=2, alpha=0.8)
        
        # Mark start and end
        ax1.plot(positions[0, 0], positions[0, 1], 'go', markersize=10, label='Start')
        ax1.plot(positions[-1, 0], positions[-1, 1], 'ro', markersize=10, label='End')
        
        # Show direction arrows
        if show_arrows:
            arrow_indices = np.linspace(0, len(positions)-1, min(10, len(positions)), dtype=int)
            for idx in arrow_indices:
                if idx < len(positions):
                    x, y = positions[idx]
                    theta = orientations[idx]
                    dx, dy = 0.1 * np.cos(theta), 0.1 * np.sin(theta)
                    ax1.arrow(x, y, dx, dy, head_width=0.05, head_length=0.05, 
                            fc='blue', ec='blue', alpha=0.7)
        
        # Add trajectory info
        success_text = "SUCCESS" if traj['success'] else "FAILED"
        collision_text = " (COLLISION)" if traj['collision'] else ""
        title = (f"Trajectory {traj_idx}: {success_text}{collision_text}\\n"
                f"Steps: {traj['steps']}, Reward: {traj['total_reward']:.1f}, "
                f"Final Distance: {traj['final_distance_to_goal']:.3f}")
        ax1.set_title(title)
        ax1.legend()
        
        # Plot actions if requested
        if show_actions:
            time_steps = np.arange(len(actions))
            ax2.plot(time_steps, actions, 'b-', linewidth=2, label='Angular Velocity')
            ax2.set_xlabel('Time Step')
            ax2.set_ylabel('Angular Velocity (rad/s)')
            ax2.set_title('Action Sequence')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            # Highlight collision/success points
            if traj['collision']:
                ax2.axvline(len(actions)-1, color='red', linestyle='--', alpha=0.7, label='Collision')
            elif traj['success']:
                ax2.axvline(len(actions)-1, color='green', linestyle='--', alpha=0.7, label='Success')
        
        plt.tight_layout()
        return fig
    
    def plot_multiple_trajectories(self, max_trajectories: int = 20, 
                                 success_only: bool = False, figsize: Tuple[int, int] = (10, 8)):
        """
        Plot multiple trajectories on the same figure.
        
        Args:
            max_trajectories: Maximum number of trajectories to plot
            success_only: Only plot successful trajectories
            figsize: Figure size
        """
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # Plot environment
        self.plot_environment(ax, alpha=0.5)
        
        # Filter trajectories
        trajectories_to_plot = self.trajectories[:max_trajectories]
        if success_only:
            trajectories_to_plot = [t for t in trajectories_to_plot if t['success']]
        
        # Plot trajectories
        colors = plt.cm.tab20(np.linspace(0, 1, len(trajectories_to_plot)))
        
        for i, traj in enumerate(trajectories_to_plot):
            states = traj['states']
            positions = states[:, :2]
            
            alpha = 0.8 if traj['success'] else 0.4
            linestyle = '-' if traj['success'] else '--'
            
            ax.plot(positions[:, 0], positions[:, 1], 
                   color=colors[i], alpha=alpha, linewidth=2, linestyle=linestyle)
            
            # Mark start
            ax.plot(positions[0, 0], positions[0, 1], 'o', color=colors[i], markersize=6)
        
        # Add legend
        success_count = sum(1 for t in trajectories_to_plot if t['success'])
        total_count = len(trajectories_to_plot)
        
        title = f"Multiple Trajectories (n={total_count})\\n"
        title += f"Success Rate: {success_count}/{total_count} ({success_count/total_count:.1%})"
        ax.set_title(title)
        
        plt.tight_layout()
        return fig
    
    def plot_performance_statistics(self, figsize: Tuple[int, int] = (15, 10)):
        """
        Plot various performance statistics.
        
        Args:
            figsize: Figure size
        """
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
        
        # Extract data
        successes = [t['success'] for t in self.trajectories]
        collisions = [t['collision'] for t in self.trajectories]
        steps = [t['steps'] for t in self.trajectories]
        rewards = [t['total_reward'] for t in self.trajectories]
        final_distances = [t['final_distance_to_goal'] for t in self.trajectories]
        
        # 1. Success rate pie chart
        success_count = sum(successes)
        collision_count = sum(collisions)
        other_count = len(self.trajectories) - success_count - collision_count
        
        labels = ['Success', 'Collision', 'Other']
        sizes = [success_count, collision_count, other_count]
        colors = ['green', 'red', 'orange']
        
        axes[0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        axes[0].set_title('Episode Outcomes')
        
        # 2. Steps histogram
        axes[1].hist(steps, bins=20, alpha=0.7, edgecolor='black')
        axes[1].set_xlabel('Steps')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Episode Length Distribution')
        axes[1].axvline(np.mean(steps), color='red', linestyle='--', label=f'Mean: {np.mean(steps):.1f}')
        axes[1].legend()
        
        # 3. Reward histogram
        axes[2].hist(rewards, bins=20, alpha=0.7, edgecolor='black')
        axes[2].set_xlabel('Total Reward')
        axes[2].set_ylabel('Frequency')
        axes[2].set_title('Reward Distribution')
        axes[2].axvline(np.mean(rewards), color='red', linestyle='--', label=f'Mean: {np.mean(rewards):.1f}')
        axes[2].legend()
        
        # 4. Final distance to goal
        axes[3].hist(final_distances, bins=20, alpha=0.7, edgecolor='black')
        axes[3].set_xlabel('Final Distance to Goal')
        axes[3].set_ylabel('Frequency')
        axes[3].set_title('Final Distance Distribution')
        axes[3].axvline(np.mean(final_distances), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(final_distances):.3f}')
        axes[3].legend()
        
        # 5. Success vs steps
        success_steps = [s for i, s in enumerate(steps) if successes[i]]
        failure_steps = [s for i, s in enumerate(steps) if not successes[i]]
        
        if success_steps:
            axes[4].hist(success_steps, bins=15, alpha=0.7, label='Success', color='green')
        if failure_steps:
            axes[4].hist(failure_steps, bins=15, alpha=0.7, label='Failure', color='red')
        
        axes[4].set_xlabel('Steps')
        axes[4].set_ylabel('Frequency')
        axes[4].set_title('Steps by Outcome')
        axes[4].legend()
        
        # 6. Reward vs steps scatter
        colors = ['green' if s else 'red' for s in successes]
        axes[5].scatter(steps, rewards, c=colors, alpha=0.6)
        axes[5].set_xlabel('Steps')
        axes[5].set_ylabel('Total Reward')
        axes[5].set_title('Reward vs Steps')
        
        # Add correlation
        correlation = np.corrcoef(steps, rewards)[0, 1]
        axes[5].text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                    transform=axes[5].transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
        
        plt.tight_layout()
        return fig
    
    def create_animated_trajectory(self, traj_idx: int, save_path: Optional[str] = None, 
                                 fps: int = 10, figsize: Tuple[int, int] = (10, 8)):
        """
        Create an animated visualization of a single trajectory.
        
        Args:
            traj_idx: Index of trajectory to animate
            save_path: Path to save animation (None for display only)
            fps: Frames per second
            figsize: Figure size
        """
        if traj_idx >= len(self.trajectories):
            raise ValueError(f"Trajectory index {traj_idx} out of range")
        
        traj = self.trajectories[traj_idx]
        states = traj['states']
        actions = traj['actions']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Setup environment plot
        self.plot_environment(ax1, alpha=0.3)
        
        # Initialize trajectory line and agent
        line, = ax1.plot([], [], 'b-', linewidth=2, alpha=0.7, label='Trajectory')
        agent_dot, = ax1.plot([], [], 'bo', markersize=8, label='Agent')
        agent_arrow = ax1.annotate('', xy=(0, 0), xytext=(0, 0),
                                  arrowprops=dict(arrowstyle='->', color='blue', lw=2))
        
        # Setup action plot
        ax2.set_xlim(0, len(actions))
        ax2.set_ylim(min(actions) * 1.1, max(actions) * 1.1)
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Angular Velocity (rad/s)')
        ax2.set_title('Actions')
        ax2.grid(True, alpha=0.3)
        
        action_line, = ax2.plot([], [], 'r-', linewidth=2)
        current_action, = ax2.plot([], [], 'ro', markersize=8)
        
        def animate(frame):
            if frame < len(states):
                # Update trajectory
                positions = states[:frame+1, :2]
                line.set_data(positions[:, 0], positions[:, 1])
                
                # Update agent position and orientation
                current_pos = states[frame, :2]
                current_theta = states[frame, 2]
                
                agent_dot.set_data([current_pos[0]], [current_pos[1]])
                
                # Update agent arrow
                arrow_length = 0.15
                arrow_end = current_pos + arrow_length * np.array([np.cos(current_theta), np.sin(current_theta)])
                agent_arrow.set_position(current_pos)
                agent_arrow.xy = arrow_end
                
                # Update action plot
                if frame < len(actions):
                    time_steps = np.arange(frame + 1)
                    action_line.set_data(time_steps, actions[:frame+1])
                    current_action.set_data([frame], [actions[frame]])
                
                # Update title with current info
                step_info = f"Step: {frame}, Action: {actions[frame]:.3f}" if frame < len(actions) else f"Step: {frame} (Final)"
                ax1.set_title(f"Trajectory {traj_idx} Animation\\n{step_info}")
            
            return line, agent_dot, agent_arrow, action_line, current_action
        
        # Create animation
        anim = FuncAnimation(fig, animate, frames=len(states), interval=1000//fps, 
                           blit=False, repeat=True)
        
        ax1.legend()
        plt.tight_layout()
        
        if save_path:
            print(f"Saving animation to {save_path}...")
            anim.save(save_path, writer=PillowWriter(fps=fps))
            print("Animation saved!")
        
        return fig, anim
    
    def plot_action_analysis(self, figsize: Tuple[int, int] = (15, 8)):
        """
        Analyze and plot action patterns across all trajectories.
        
        Args:
            figsize: Figure size
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        # Collect all actions
        all_actions = []
        successful_actions = []
        failed_actions = []
        
        for traj in self.trajectories:
            actions = traj['actions']
            all_actions.extend(actions)
            
            if traj['success']:
                successful_actions.extend(actions)
            else:
                failed_actions.extend(actions)
        
        # 1. Overall action distribution
        axes[0].hist(all_actions, bins=50, alpha=0.7, edgecolor='black')
        axes[0].set_xlabel('Angular Velocity (rad/s)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Action Distribution (All Trajectories)')
        axes[0].axvline(np.mean(all_actions), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(all_actions):.3f}')
        axes[0].legend()
        
        # 2. Success vs failure actions
        if successful_actions:
            axes[1].hist(successful_actions, bins=30, alpha=0.7, label='Success', color='green')
        if failed_actions:
            axes[1].hist(failed_actions, bins=30, alpha=0.7, label='Failure', color='red')
        
        axes[1].set_xlabel('Angular Velocity (rad/s)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Action Distribution by Outcome')
        axes[1].legend()
        
        # 3. Action variance over time
        max_length = max(len(traj['actions']) for traj in self.trajectories)
        action_matrix = np.full((len(self.trajectories), max_length), np.nan)
        
        for i, traj in enumerate(self.trajectories):
            actions = traj['actions']
            action_matrix[i, :len(actions)] = actions
        
        # Calculate mean and std over trajectories at each timestep
        mean_actions = np.nanmean(action_matrix, axis=0)
        std_actions = np.nanstd(action_matrix, axis=0)
        timesteps = np.arange(len(mean_actions))
        
        # Only plot up to where we have reasonable data
        valid_steps = np.sum(~np.isnan(action_matrix), axis=0) > len(self.trajectories) * 0.1
        valid_timesteps = timesteps[valid_steps]
        
        axes[2].plot(valid_timesteps, mean_actions[valid_steps], 'b-', linewidth=2, label='Mean')
        axes[2].fill_between(valid_timesteps, 
                           mean_actions[valid_steps] - std_actions[valid_steps],
                           mean_actions[valid_steps] + std_actions[valid_steps],
                           alpha=0.3, label='±1 Std')
        axes[2].set_xlabel('Time Step')
        axes[2].set_ylabel('Angular Velocity (rad/s)')
        axes[2].set_title('Action Evolution Over Time')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # 4. Action magnitude vs success
        action_magnitudes = [np.mean(np.abs(traj['actions'])) for traj in self.trajectories]
        successes = [traj['success'] for traj in self.trajectories]
        
        success_magnitudes = [mag for i, mag in enumerate(action_magnitudes) if successes[i]]
        failure_magnitudes = [mag for i, mag in enumerate(action_magnitudes) if not successes[i]]
        
        if success_magnitudes:
            axes[3].hist(success_magnitudes, bins=20, alpha=0.7, label='Success', color='green')
        if failure_magnitudes:
            axes[3].hist(failure_magnitudes, bins=20, alpha=0.7, label='Failure', color='red')
        
        axes[3].set_xlabel('Mean |Angular Velocity|')
        axes[3].set_ylabel('Frequency')
        axes[3].set_title('Action Magnitude by Outcome')
        axes[3].legend()
        
        plt.tight_layout()
        return fig


def visualize_from_file(filepath: str, output_dir: str = "visualizations"):
    """
    Load trajectories from HDF5 file and create visualizations.
    
    Args:
        filepath: Path to HDF5 file
        output_dir: Directory to save visualizations
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print(f"Loading trajectories from {filepath}...")
    trajectories, metadata = load_trajectories_from_hdf5(filepath)
    print(f"Loaded {len(trajectories)} trajectories")
    
    # Create visualizer
    visualizer = TrajectoryVisualizer(trajectories, metadata)
    
    # Generate visualizations
    print("Creating visualizations...")
    
    # 1. Performance statistics
    #fig1 = visualizer.plot_performance_statistics()
    #fig1.savefig(os.path.join(output_dir, "performance_stats.png"), dpi=300, bbox_inches='tight')
    #plt.close(fig1)
    
    # 2. Multiple trajectories overview
    fig2 = visualizer.plot_multiple_trajectories(max_trajectories=1000)
    fig2.savefig(os.path.join(output_dir, "trajectory_overview.png"), dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    # 3. Action analysis
    #fig3 = visualizer.plot_action_analysis()
    #fig3.savefig(os.path.join(output_dir, "action_analysis.png"), dpi=300, bbox_inches='tight')
    #plt.close(fig3)
    
    '''# 4. Individual trajectory examples
    successful_trajectories = [i for i, t in enumerate(trajectories) if t['success']]
    failed_trajectories = [i for i, t in enumerate(trajectories) if not t['success']]
    
    # Plot a few successful trajectories
    for i, traj_idx in enumerate(successful_trajectories[:3]):
        fig4 = visualizer.plot_single_trajectory(traj_idx, show_actions=True)
        fig4.savefig(os.path.join(output_dir, f"successful_trajectory_{i}.png"), 
                    dpi=300, bbox_inches='tight')
        plt.close(fig4)
    
    # Plot a few failed trajectories
    for i, traj_idx in enumerate(failed_trajectories[:3]):
        fig5 = visualizer.plot_single_trajectory(traj_idx, show_actions=True)
        fig5.savefig(os.path.join(output_dir, f"failed_trajectory_{i}.png"), 
                    dpi=300, bbox_inches='tight')
        plt.close(fig5)
    
    # 5. Create animation for best trajectory
    if successful_trajectories:
        best_traj_idx = successful_trajectories[0]  # First successful one
        animation_path = os.path.join(output_dir, "best_trajectory_animation.gif")
        fig6, anim = visualizer.create_animated_trajectory(best_traj_idx, animation_path)
        plt.close(fig6)
    '''
    print(f"Visualizations saved to {output_dir}/")
    
    # Print summary
    stats = metadata.get('statistics', {})
    print(f"\\nDataset Summary:")
    print(f"- Total trajectories: {len(trajectories)}")
    print(f"- Success rate: {stats.get('success_rate', 0):.1%}")
    print(f"- Average steps: {stats.get('average_steps', 0):.1f}")
    print(f"- Average reward: {stats.get('average_reward', 0):.2f}")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Visualize trajectories from HDF5 files')
    parser.add_argument('--filepath', type=str, default='/data/dubins/dubins_gap.h5', help='Path to HDF5 trajectory file')
    parser.add_argument('--output_dir', type=str, default='visualizations', 
                       help='Output directory for visualizations')
    parser.add_argument('--show_interactive', action='store_true', 
                       help='Show interactive plots (in addition to saving)')
    
    args = parser.parse_args()
    
    # Create visualizations
    visualize_from_file(args.filepath, args.output_dir)
    
    if args.show_interactive:
        print("Loading data for interactive visualization...")
        trajectories, metadata = load_trajectories_from_hdf5(args.filepath)
        visualizer = TrajectoryVisualizer(trajectories, metadata)
        
        # Show some interactive plots
        fig1 = visualizer.plot_performance_statistics()
        fig2 = visualizer.plot_multiple_trajectories()
        
        plt.show()


if __name__ == "__main__":
    main()
