#!/usr/bin/env python3
"""
Script to combine all trajectory HDF5 files into a single buffer.h5 file.
Combines ``<DATA_ROOT>/trajs/*.h5`` into ``<DATA_ROOT>/buffers/dreamer_buffer.h5``
where ``<DATA_ROOT>`` is ``DATA_ROOT`` in ``configs.paths``.
"""

import sys
import os
import glob
from pathlib import Path

import h5py
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.paths import DREAMER_BUFFER, TRAJS_DIR


def get_all_trajectory_files():
    """Get all trajectory files to combine."""
    traj_files = glob.glob(str(TRAJS_DIR / "*.h5"))
        
    # Sort files for consistent ordering
    traj_files_sorted = sorted(traj_files)
    all_files_sorted = traj_files_sorted
    
    print(f"Found {len(traj_files)}  trajectory files")
    print(f"Total files to combine: {len(all_files_sorted)}")
    
    return all_files_sorted

def copy_group_structure(source_group, target_group):
    """Recursively copy HDF5 group structure."""
    for key in source_group.keys():
        if isinstance(source_group[key], h5py.Group):
            # Create subgroup and copy recursively
            if key not in target_group:
                target_group.create_group(key)
            copy_group_structure(source_group[key], target_group[key])
        elif isinstance(source_group[key], h5py.Dataset):
            # Copy dataset
            source_ds = source_group[key]
            target_group.create_dataset(
                key, 
                data=source_ds[...], 
                dtype=source_ds.dtype,
                compression=source_ds.compression if source_ds.compression else None
            )

def combine_trajectory_files():
    """Combine all trajectory files into a single buffer.h5 file."""
    
    # Get all files to combine
    files_to_combine = get_all_trajectory_files()
    
    if not files_to_combine:
        print("No files found to combine!")
        return
    
    output_file = str(DREAMER_BUFFER)
    
    # Remove existing output file if it exists
    if os.path.exists(output_file):
        print(f"Removing existing {output_file}")
        os.remove(output_file)
    
    print(f"Creating combined file: {output_file}")
    
    with h5py.File(output_file, 'w') as output_f:
        
        # Initialize counters
        total_trajectories = 0
        trajectory_id_counter = 0
        
        # First pass: collect metadata and count trajectories
        print("First pass: Collecting metadata and counting trajectories...")
        combined_metadata = {}
        combined_statistics = {}
        
        for file_path in files_to_combine:
            print(f"Processing: {os.path.basename(file_path)}")
            
            with h5py.File(file_path, 'r') as input_f:
                # Count trajectories in this file
                if 'trajectories' in input_f:
                    num_trajs = len([k for k in input_f['trajectories'].keys() if k.startswith('trajectory_')])
                    total_trajectories += num_trajs
                    print(f"  - Found {num_trajs} trajectories")
                
                # Collect metadata (use first file's metadata as base)
                if 'metadata' in input_f and not combined_metadata:
                    copy_group_structure(input_f['metadata'], output_f.create_group('metadata'))
                    combined_metadata = dict(input_f['metadata'].attrs)
                
                # Collect statistics (accumulate if needed)
                if 'statistics' in input_f and not combined_statistics:
                    copy_group_structure(input_f['statistics'], output_f.create_group('statistics'))
                    combined_statistics = dict(input_f['statistics'].attrs)
        
        print(f"Total trajectories to copy: {total_trajectories}")
        
        # Create trajectories group
        trajectories_group = output_f.create_group('trajectories')
        
        # Second pass: copy all trajectories with renumbered IDs
        print("Second pass: Copying trajectories with renumbered IDs...")
        
        for file_idx, file_path in enumerate(files_to_combine):
            print(f"Processing: {os.path.basename(file_path)}")
            
            with h5py.File(file_path, 'r') as input_f:
                if 'trajectories' not in input_f:
                    continue
                
                # Get trajectory keys and sort them
                traj_keys = [k for k in input_f['trajectories'].keys() if k.startswith('trajectory_')]
                traj_keys.sort()
                
                for traj_key in traj_keys:
                    # Create new trajectory ID
                    new_traj_key = f"trajectory_{trajectory_id_counter:06d}"
                    
                    # Copy trajectory group
                    copy_group_structure(
                        input_f['trajectories'][traj_key], 
                        trajectories_group.create_group(new_traj_key)
                    )
                    
                    trajectory_id_counter += 1
                    
                    if trajectory_id_counter % 100 == 0:
                        print(f"  - Copied {trajectory_id_counter} trajectories...")
        
        print(f"Successfully combined {trajectory_id_counter} trajectories into {output_file}")
        
        # Add some metadata about the combination
        output_f.attrs['total_trajectories'] = trajectory_id_counter
        output_f.attrs['source_files'] = [os.path.basename(f) for f in files_to_combine]
        output_f.attrs['combined_date'] = str(np.datetime64('now'))
        
        print(f"Added metadata attributes:")
        print(f"  - total_trajectories: {output_f.attrs['total_trajectories']}")
        print(f"  - source_files: {len(output_f.attrs['source_files'])} files")
        print(f"  - combined_date: {output_f.attrs['combined_date']}")

def verify_combined_file():
    """Verify the structure and contents of the combined file."""
    output_file = str(DREAMER_BUFFER)
    
    if not os.path.exists(output_file):
        print(f"Output file {output_file} does not exist!")
        return
    
    print(f"\nVerifying combined file: {output_file}")
    
    with h5py.File(output_file, 'r') as f:
        print(f"File size: {os.path.getsize(output_file) / (1024**3):.2f} GB")
        
        # Check top-level structure
        print(f"Top-level groups: {list(f.keys())}")
        
        # Check trajectory count
        if 'trajectories' in f:
            traj_count = len([k for k in f['trajectories'].keys() if k.startswith('trajectory_')])
            print(f"Number of trajectories: {traj_count}")
            
            # Check a few sample trajectories
            traj_keys = [k for k in f['trajectories'].keys() if k.startswith('trajectory_')]
            traj_keys.sort()
            
            print(f"First trajectory: {traj_keys[0] if traj_keys else 'None'}")
            print(f"Last trajectory: {traj_keys[-1] if traj_keys else 'None'}")
            
            # Sample trajectory structure
            if traj_keys:
                sample_traj = f['trajectories'][traj_keys[0]]
                print(f"Sample trajectory structure: {list(sample_traj.keys())}")
                for key in sample_traj.keys():
                    if isinstance(sample_traj[key], h5py.Dataset):
                        print(f"  - {key}: shape={sample_traj[key].shape}, dtype={sample_traj[key].dtype}")
        
        # Check attributes
        print(f"File attributes: {dict(f.attrs)}")

if __name__ == "__main__":
    print("Starting trajectory combination process...")
    
    try:
        combine_trajectory_files()
        verify_combined_file()
        print("\n Successfully combined all trajectory files!")
        
    except Exception as e:
        print(f"\n Error during combination: {e}")
        import traceback
        traceback.print_exc()
