#!/usr/bin/env python3
"""
Batch script to collect diffusion trajectories for multiple checkpoints.

This script runs collect_trajs.py for multiple checkpoint values (500, 1000, 1500, ..., 10000)
to generate training data for different diffusion policy checkpoints.
"""

import subprocess
import sys
import os
from typing import List

def run_collect_trajs(checkpoint: int) -> bool:
    """
    Run collect_trajs.py for a specific checkpoint.
    
    Args:
        checkpoint: Checkpoint number to use
        
    Returns:
        True if successful, False otherwise
    """
    cmd = [
        "python", "scripts/collect_trajs.py",
        "--controller", "diffusion",
        "--config", "diffusion", 
        "--n_trajectories", "100",
        "--filename", f"diffusion_trajectories_{checkpoint}",
        "--output_dir", "/data/dubins/diffusion_trajs",
        "--save_images",
        "--verbose",
        "--checkpoint", str(checkpoint)
    ]
    
    print(f"\n{'='*60}")
    print(f"Running checkpoint {checkpoint}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ Checkpoint {checkpoint} completed successfully!")
        if result.stdout:
            print("STDOUT:", result.stdout[-500:])  # Last 500 chars
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Checkpoint {checkpoint} failed with return code {e.returncode}")
        if e.stdout:
            print("STDOUT:", e.stdout[-500:])
        if e.stderr:
            print("STDERR:", e.stderr[-500:])
        return False
    except Exception as e:
        print(f"❌ Checkpoint {checkpoint} failed with exception: {e}")
        return False

def main():
    """Main function to run batch collection."""
    # Check if we're in the right directory
    if not os.path.exists("scripts/collect_trajs.py"):
        print("❌ Error: scripts/collect_trajs.py not found!")
        print("Please run this script from the latent_cbf directory.")
        sys.exit(1)
    
    # Define checkpoint values: 500, 1000, 1500, ..., 10000
    checkpoints = list(range(500, 19501, 500))  # 500, 1000, 1500, ..., 10000
    
    print(f"🚀 Starting batch collection for {len(checkpoints)} checkpoints:")
    print(f"Checkpoints: {checkpoints}")
    
    # Track results
    successful = []
    failed = []
    
    # Run collection for each checkpoint
    for checkpoint in checkpoints:
        success = run_collect_trajs(checkpoint)
        if success:
            successful.append(checkpoint)
        else:
            failed.append(checkpoint)
    
    # Print summary
    print(f"\n{'='*60}")
    print("BATCH COLLECTION SUMMARY")
    print(f"{'='*60}")
    print(f"Total checkpoints: {len(checkpoints)}")
    print(f"✅ Successful: {len(successful)} - {successful}")
    print(f"❌ Failed: {len(failed)} - {failed}")
    
    if failed:
        print(f"\n⚠️  Some checkpoints failed. You may want to retry them individually.")
        print("Failed checkpoint commands:")
        for checkpoint in failed:
            print(f"python scripts/collect_trajs.py --controller diffusion --config diffusion --n_trajectories 100 --filename diffusion_trajectories_{checkpoint} --output_dir /data/dubins --save_images --verbose --checkpoint {checkpoint}")
    else:
        print(f"\n🎉 All checkpoints completed successfully!")
    
    # Exit with error code if any failed
    sys.exit(1 if failed else 0)

if __name__ == "__main__":
    main()
