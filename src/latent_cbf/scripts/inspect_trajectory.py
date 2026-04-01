"""
Simple script to inspect trajectory data
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import imageio
from PIL import Image
import io
def inspect_trajectory_file(filepath):
    """Inspect the contents of a trajectory HDF5 file."""
    
    print(f"Opening: {filepath}")
    print("=" * 50)
    
    with h5py.File(filepath, 'r') as f:
        print("Top-level keys:", list(f.keys()))
        print()
        
        # Check metadata
        if 'metadata' in f:
            print("METADATA:")
            metadata = f['metadata']
            print(f"  Controller: {metadata.attrs.get('controller_type', 'Unknown')}")
            print(f"  Environment: {metadata.attrs.get('environment_name', 'Unknown')}")
            print()
        
        # Check statistics
        if 'statistics' in f:
            print("STATISTICS:")
            stats = f['statistics']
            for key in stats.keys():
                print(f"  {key}: {stats[key][()]}")
            print()
        
        # Check trajectories
        if 'trajectories' in f:
            trajs = f['trajectories']
            traj_keys = [k for k in trajs.keys() if k.startswith('trajectory_')]
            print(f"TRAJECTORIES: {len(traj_keys)} found")
            print(f"  Keys: {traj_keys}")
            print()
            
            # Inspect first trajectory
            for traj_key in traj_keys:
                traj = trajs[traj_key]
                print(f"TRAJECTORY {traj_key}:")
                print(f"  Keys: {list(traj.keys())}")
                
                # Check each dataset
                for key in traj.keys():
                    if hasattr(traj[key], 'shape'):
                        print(f"  {key}: shape={traj[key].shape}, dtype={traj[key].dtype}")
                        
                
                
                # make into 1d array
                failures = traj['failures'][:].copy()
                N = failures.shape[0]
                imgs = traj['observations'][:][1:N+1].copy()
                margin_gps = traj['margin_gp'][:N].copy()
                margin_nogps = traj['margin_nogp'][:N].copy()
                images = []
                # Create individual frames
                for i in range(N):
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                    
                    
                    ax1.imshow(imgs[i])
                    ax1.set_title(f'Frame {i}')
                    ax1.axis('off')
                    
                    # Right plot: Margins and failures
                    ax2.plot(np.tanh(margin_gps[:i+1]), 'g-', label='Margin GP', linewidth=2)
                    ax2.plot(np.tanh(margin_nogps[:i+1]/10), 'orange', label='Margin NoGP', linewidth=2)
                    ax2.plot(failures[:i+1], 'r-', label='Failures', linewidth=2)
                    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                    ax2.axvline(x=8, color='black', linestyle='--', alpha=0.5)
                    ax2.axvline(x=16, color='black', linestyle='--', alpha=0.5)
                    ax2.axvline(x=24, color='black', linestyle='--', alpha=0.5)
                    ax2.axvline(x=32, color='black', linestyle='--', alpha=0.5)
                    ax2.legend()
                    ax2.set_title('WM Predictions vs Ground Truth')
                    ax2.set_xlabel('Time Step')
                    ax2.set_ylabel('Value')
                    ax2.grid(True, alpha=0.3)
                    
                    # Set consistent y-axis limits
                    all_values = np.concatenate([margin_gps[:i+1], margin_nogps[:i+1], failures[:i+1]])
                    if len(all_values) > 0:
                        y_min, y_max = np.min(all_values), np.max(all_values)
                        y_range = y_max - y_min
                        if y_range > 0:
                            ax2.set_ylim(-1.1, 1.1)
                        ax2.set_xlim(0, N)
                    
                    plt.tight_layout()
    
                    # Save to in-memory buffer instead of file
                    buffer = io.BytesIO()
                    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
                    buffer.seek(0)
                    
                    # Read from buffer and convert to numpy array
                    pil_image = Image.open(buffer)
                    pil_image = pil_image.resize((1112, 490), Image.Resampling.LANCZOS)
                    img_array = np.array(pil_image)
                    images.append(img_array)
                    
                    plt.close()
                    buffer.close()
                
                # Create GIF from in-memory images
                if images:
                    imageio.mimsave(f'visualizations/margin_gps_{traj_key}.gif', images, fps=10)
                    print(f"GIF saved as visualizations/margin_gps_{traj_key}.gif")
                else:
                    print("No images created")


        


if __name__ == "__main__":
    # Change this path to your trajectory file
    filepath = "/data/dubins/trajs/wm_test.h5"
    inspect_trajectory_file(filepath)