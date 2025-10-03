"""
Simple Video Creator for Trajectory Visualizations

Creates MP4 videos from image observations stored in HDF5 trajectory files.
Each trajectory becomes a separate video file.
"""

import sys
import os
import numpy as np
import cv2
import h5py
import argparse
from typing import List, Optional
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.collect_trajs import load_trajectories_from_hdf5


def create_video_from_images(images: np.ndarray, output_path: str, fps: int = 10, 
                           text_overlay: Optional[str] = None) -> bool:
    """
    Create a video from a sequence of images.
    
    Args:
        images: Array of images with shape (T, H, W, 3)
        output_path: Path to save the video
        fps: Frames per second
        text_overlay: Optional text to overlay on video
        
    Returns:
        True if successful, False otherwise
    """
    if len(images) == 0:
        print(f"No images provided for {output_path}")
        return False
    
    # Get video dimensions
    height, width = images[0].shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not video_writer.isOpened():
        print(f"Error: Could not open video writer for {output_path}")
        return False
    
    try:
        for i, img in enumerate(images):
            # Convert RGB to BGR for OpenCV
            frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            # Add text overlay if provided
            if text_overlay:
                # Add semi-transparent background for text
                overlay = frame.copy()
                cv2.rectangle(overlay, (5, 5), (300, 60), (0, 0, 0), -1)
                frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
                
                # Add text
                cv2.putText(frame, text_overlay, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"Step: {i}/{len(images)-1}", (10, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            video_writer.write(frame)
        
        video_writer.release()
        return True
        
    except Exception as e:
        print(f"Error writing video {output_path}: {e}")
        video_writer.release()
        return False


def create_trajectory_videos(hdf5_path: str, output_dir: str = "videos", 
                           fps: int = 10, max_videos: Optional[int] = None,
                           success_only: bool = False, failed_only: bool = False) -> List[str]:
    """
    Create videos for all trajectories in an HDF5 file.
    
    Args:
        hdf5_path: Path to HDF5 trajectory file
        output_dir: Directory to save videos
        fps: Frames per second for videos
        max_videos: Maximum number of videos to create (None for all)
        success_only: Only create videos for successful trajectories
        failed_only: Only create videos for failed trajectories
        
    Returns:
        List of created video file paths
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load trajectories
    print(f"Loading trajectories from {hdf5_path}...")
    trajectories, metadata = load_trajectories_from_hdf5(hdf5_path)
    print(f"Loaded {len(trajectories)} trajectories")
    
    # Filter trajectories if requested
    if success_only:
        trajectories = [t for t in trajectories if t['success']]
        print(f"Filtered to {len(trajectories)} successful trajectories")
    elif failed_only:
        trajectories = [t for t in trajectories if not t['success']]
        print(f"Filtered to {len(trajectories)} failed trajectories")
    
    # Limit number of videos if requested
    if max_videos is not None and len(trajectories) > max_videos:
        trajectories = trajectories[:max_videos]
        print(f"Limited to first {max_videos} trajectories")
    
    if len(trajectories) == 0:
        print("No trajectories to process!")
        return []
    
    # Check if trajectories have image observations
    sample_traj = trajectories[0]
    if sample_traj['observations'] is None:
        print("Error: Trajectories do not contain image observations!")
        print("Make sure to collect trajectories with --save_images flag")
        return []
    
    print(f"Creating {len(trajectories)} videos...")
    
    created_videos = []
    
    # Create videos with progress bar
    for i, traj in enumerate(tqdm(trajectories, desc="Creating videos")):
        # Get trajectory info
        episode_idx = traj['episode_idx']
        success = traj['success']
        collision = traj['collision']
        steps = traj['steps']
        total_reward = traj['total_reward']
        final_distance = traj['final_distance_to_goal']
        
        # Create filename
        status = "success" if success else ("collision" if collision else "failed")
        filename = f"traj_{episode_idx:04d}_{status}_steps{steps}_reward{total_reward:.1f}.mp4"
        video_path = os.path.join(output_dir, filename)
        
        # Create text overlay
        overlay_text = f"Traj {episode_idx}: {status.upper()}"
        
        # Get images
        images = traj['observations']
        
        # Create video
        if create_video_from_images(images, video_path, fps, overlay_text):
            created_videos.append(video_path)
        else:
            print(f"Failed to create video for trajectory {episode_idx}")
    
    print(f"\\nSuccessfully created {len(created_videos)} videos in {output_dir}/")
    
    # Print summary
    if created_videos:
        print("\\nCreated videos:")
        for video_path in created_videos[:10]:  # Show first 10
            filename = os.path.basename(video_path)
            print(f"  - {filename}")
        
        if len(created_videos) > 10:
            print(f"  ... and {len(created_videos) - 10} more")
    
    return created_videos


def create_summary_video(hdf5_path: str, output_path: str = "summary_video.mp4", 
                        fps: int = 10, max_trajectories: int = 10) -> bool:
    """
    Create a single summary video showing multiple trajectories side by side.
    
    Args:
        hdf5_path: Path to HDF5 trajectory file
        output_path: Path to save summary video
        fps: Frames per second
        max_trajectories: Maximum number of trajectories to include
        
    Returns:
        True if successful
    """
    # Load trajectories
    trajectories, metadata = load_trajectories_from_hdf5(hdf5_path)
    
    # Filter and limit trajectories
    trajectories = trajectories[:max_trajectories]
    
    # Check for image observations
    if not trajectories or trajectories[0]['observations'] is None:
        print("Error: No image observations available")
        return False
    
    print(f"Creating summary video with {len(trajectories)} trajectories...")
    
    # Determine grid layout
    n_trajs = len(trajectories)
    cols = min(4, n_trajs)  # Maximum 4 columns
    rows = (n_trajs + cols - 1) // cols
    
    # Get image dimensions
    sample_img = trajectories[0]['observations'][0]
    img_h, img_w = sample_img.shape[:2]
    
    # Calculate output dimensions
    output_w = img_w * cols
    output_h = img_h * rows
    
    # Find maximum length
    max_length = max(len(traj['observations']) for traj in trajectories)
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (output_w, output_h))
    
    if not video_writer.isOpened():
        print(f"Error: Could not create video writer for {output_path}")
        return False
    
    try:
        for frame_idx in tqdm(range(max_length), desc="Creating summary video"):
            # Create combined frame
            combined_frame = np.zeros((output_h, output_w, 3), dtype=np.uint8)
            
            for traj_idx, traj in enumerate(trajectories):
                # Calculate position in grid
                row = traj_idx // cols
                col = traj_idx % cols
                
                # Get image for this frame (or last image if trajectory ended)
                if frame_idx < len(traj['observations']):
                    img = traj['observations'][frame_idx]
                else:
                    img = traj['observations'][-1]  # Use last frame
                
                # Convert to BGR
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                
                # Add trajectory info
                status = "SUCCESS" if traj['success'] else "FAILED"
                color = (0, 255, 0) if traj['success'] else (0, 0, 255)
                
                cv2.putText(img_bgr, f"Traj {traj['episode_idx']}", (5, 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                cv2.putText(img_bgr, status, (5, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
                # Place in combined frame
                y_start = row * img_h
                y_end = y_start + img_h
                x_start = col * img_w
                x_end = x_start + img_w
                
                combined_frame[y_start:y_end, x_start:x_end] = img_bgr
            
            video_writer.write(combined_frame)
        
        video_writer.release()
        print(f"Summary video saved to {output_path}")
        return True
        
    except Exception as e:
        print(f"Error creating summary video: {e}")
        video_writer.release()
        return False


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Create videos from trajectory image observations')
    parser.add_argument('hdf5_path', type=str, help='Path to HDF5 trajectory file')
    parser.add_argument('--output_dir', type=str, default='videos', 
                       help='Output directory for videos')
    parser.add_argument('--fps', type=int, default=10, help='Frames per second')
    parser.add_argument('--max_videos', type=int, default=None, 
                       help='Maximum number of videos to create')
    parser.add_argument('--success_only', action='store_true', 
                       help='Only create videos for successful trajectories')
    parser.add_argument('--failed_only', action='store_true', 
                       help='Only create videos for failed trajectories')
    parser.add_argument('--summary', action='store_true', 
                       help='Also create a summary video with multiple trajectories')
    parser.add_argument('--summary_only', action='store_true', 
                       help='Only create summary video, not individual videos')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.hdf5_path):
        print(f"Error: HDF5 file not found: {args.hdf5_path}")
        return
    
    if args.summary_only:
        # Only create summary video
        summary_path = os.path.join(args.output_dir, "summary_video.mp4")
        os.makedirs(args.output_dir, exist_ok=True)
        create_summary_video(args.hdf5_path, summary_path, args.fps)
    else:
        # Create individual videos
        created_videos = create_trajectory_videos(
            args.hdf5_path, 
            args.output_dir, 
            args.fps, 
            args.max_videos,
            args.success_only,
            args.failed_only
        )
        
        # Also create summary video if requested
        if args.summary and created_videos:
            summary_path = os.path.join(args.output_dir, "summary_video.mp4")
            create_summary_video(args.hdf5_path, summary_path, args.fps)


def find_and_process_h5_files(data_dir: str = "/data", output_base_dir: str = "videos"):
    """
    Find all HDF5 files in a directory and create videos for each.
    
    Args:
        data_dir: Directory to search for HDF5 files
        output_base_dir: Base directory for video outputs
    """
    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} not found")
        return
    
    # Find HDF5 files
    h5_files = [f for f in os.listdir(data_dir) if f.endswith('.h5')]
    
    if not h5_files:
        print(f"No HDF5 files found in {data_dir}")
        return
    
    print(f"Found {len(h5_files)} HDF5 files:")
    for f in h5_files:
        print(f"  - {f}")
    
    # Process each file
    for h5_file in h5_files:
        h5_path = os.path.join(data_dir, h5_file)
        
        # Create output directory based on filename
        file_base = os.path.splitext(h5_file)[0]
        output_dir = os.path.join(output_base_dir, file_base)
        
        print(f"\\nProcessing {h5_file}...")
        try:
            created_videos = create_trajectory_videos(h5_path, output_dir, fps=10, max_videos=20)
            
            if created_videos:
                # Also create summary video
                summary_path = os.path.join(output_dir, "summary_video.mp4")
                create_summary_video(h5_path, summary_path, fps=10, max_trajectories=8)
                
        except Exception as e:
            print(f"Error processing {h5_file}: {e}")


if __name__ == "__main__":
    if len(os.sys.argv) == 1:
        # No command line arguments - look for HDF5 files automatically
        print("No arguments provided. Looking for HDF5 files in /data directory...")
        find_and_process_h5_files()
    else:
        # Use command line arguments
        main()
