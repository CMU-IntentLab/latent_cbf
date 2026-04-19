import argparse
import os
import sys
import cv2
import pickle
import numpy as np
from tqdm import tqdm
from typing import Any, List, Union

from pathlib import Path
from robobuf.buffers import ObsWrapper, Transition, ReplayBuffer
import json

import torch
import numpy as np
import functools
import h5py

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.paths import BUFFERS_DIR, TRAJS_DIR

def extract_traj_from_hdf5(f):
    """
    Extract trajectory data from Dubins HDF5 format
    """
    trajectories = f['trajectories']
    traj_list = []
    
    # Process each trajectory in the file
    for traj_key in sorted(trajectories.keys()):
        traj = trajectories[traj_key]
        
        # Extract data arrays
        states = traj['states'][:]  # Shape: (T+1, 3) - [x, y, theta]
        actions = traj['actions'][:]  # Shape: (T,) - angular velocities
        rewards = traj['rewards'][:]  # Shape: (T,) - rewards
        observations = traj['observations'][:]  # Shape: (T+1, 128, 128, 3) - RGB images
        
        # Convert to list format expected by robobuf
        traj_data = []
        for i in range(len(actions)):
            # State: [x, y, theta] - 3D position and orientation
            state = states[i]
            
            # Action: angular velocity (scalar)
            action = actions[i]
            
            # Reward: scalar reward
            reward = rewards[i]
            
            # Observation: RGB image + state
            obs = {
                "cam_0": observations[i],  # RGB image (128, 128, 3)
                "state": state[2],  # [theta]
            }
            
            traj_data.append([obs, action, reward])
        
        traj_list.append(traj_data)
    
    return traj_list

def convert_trajectories(
    hdf5_files: List[Path], output_path: Path, image_postprocess: Any = None
):
    """
    Combine individual Dubins HDF5 trajectories into a single buffer
    """
    print(f"Working with {len(hdf5_files)} files and saving to {output_path}")

    out_buffer = None
    all_acs = []
    all_obs = []

    for file in tqdm(hdf5_files, total=len(hdf5_files)):
        with h5py.File(file, "r") as f:
            # Extract trajectories from Dubins HDF5 format
            traj_data = extract_traj_from_hdf5(f)
            traj_buffer = ReplayBuffer.load_traj_list(traj_data)
            if out_buffer:
                out_buffer.append_traj_list(traj_buffer.to_traj_list())
            else:
                out_buffer = traj_buffer
    
    new_traj_list = []
    for old_traj in tqdm(
        out_buffer.to_traj_list(),
        desc="Processing stats",
        total=len(out_buffer._traj_starts),
    ):
        for i, (old_obs, action, reward) in enumerate(old_traj):
            # Collect all states for normalization
            all_obs.append(old_obs['state'])
            # Collect all actions for normalization
            all_acs.append(action)

    # Compute normalization statistics
    all_acs_arr = np.array(all_acs)

    
    max_ac = np.max(all_acs_arr.copy(), axis=0)
    min_ac = np.min(all_acs_arr.copy(), axis=0)
    # Ensure max_ac and min_ac are numpy arrays of shape [1,]
    max_ac = np.array([max_ac]) if np.ndim(max_ac) == 0 else np.array(max_ac).reshape(1,)
    min_ac = np.array([min_ac]) if np.ndim(min_ac) == 0 else np.array(min_ac).reshape(1,)
    
    
    all_obs_arr = np.array(all_obs)
    max_ob = np.max(all_obs_arr.copy(), axis=0)
    min_ob = np.min(all_obs_arr.copy(), axis=0)
    # Ensure max_ac and min_ac are numpy arrays of shape [1,]
    max_ob = np.array([max_ob]) if np.ndim(max_ob) == 0 else np.array(max_ob).reshape(1,)
    min_ob = np.array([min_ob]) if np.ndim(min_ob) == 0 else np.array(min_ob).reshape(1,)

    print(f"Action stats - Min: {min_ac}, Max: {max_ac}")
    print(f"State stats - Min: {min_ob}, Max: {max_ob}")

    # Normalize and create new trajectory list
    for old_traj in tqdm(
        out_buffer.to_traj_list(),
        desc="Postprocessing",
        total=len(out_buffer._traj_starts),
    ):
        new_traj = ReplayBuffer()
        isFirst = True
        for i, (new_obs, action, reward) in enumerate(old_traj):
            
            # Normalize state: [x, y, theta] -> [-1, 1]
            orig_obs = new_obs['state'].copy()
            new_obs['state'] = (new_obs['state'] - min_ob) / (max_ob - min_ob)
            new_obs['state'] = new_obs['state'] * 2 - 1
            

            #import ipdb; ipdb.set_trace()
            # Normalize action: angular velocity -> [-1, 1]
            action_normalized = (action - min_ac) / (max_ac - min_ac)
            action_normalized = action_normalized * 2 - 1
            
            new_traj.add(
                Transition(ObsWrapper(new_obs), action_normalized, reward),
                is_first=(isFirst),
            )
            isFirst = False
        new_traj_list += new_traj.to_traj_list()
    
    out_buffer = ReplayBuffer.load_traj_list(new_traj_list)
    
    traj_list = out_buffer.to_traj_list()
    num_traj = len(traj_list[0])

    # Prepare normalization dictionaries
    #import ipdb; ipdb.set_trace()
    ob_dict = {'maximum': max_ob.tolist(), 'minimum': min_ob.tolist()}
    ac_dict = {'maximum': max_ac.tolist(), 'minimum': min_ac.tolist()}
    
    print(f"Sample normalized state: {traj_list[0][-1][0]['state']}")
    print(f"Sample normalized action: {traj_list[0][-1][1]}")
    
    # Validate normalization
    for i in range(num_traj):
        assert np.max(traj_list[0][i][0]['state']) <= 1        
        assert np.min(traj_list[0][i][0]['state']) >= -1 
        assert np.min(traj_list[0][i][1]) >= -1
        assert np.max(traj_list[0][i][1]) <= 1

    # Save normalization parameters
    for k in ac_dict:
        ac_dict[k] = list(ac_dict[k])
        ob_dict[k] = list(ob_dict[k])

    
    ac_output_path = output_path.parent / "ac_norm.json"
    ob_output_path = output_path.parent / "ob_norm.json"
    
    with open(ac_output_path, 'w') as f:
        json.dump(ac_dict, f)
    with open(ob_output_path, 'w') as f:
        json.dump(ob_dict, f)

    # Save trajectory buffer
    print(output_path)
    with open(output_path, "wb") as f:
        pickle.dump(traj_list, f)
    print(
        f"Saved {len(out_buffer._traj_starts)} trajectories with {len(out_buffer)} transitions total to {output_path}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_dir",
        type=str,
        default=str(TRAJS_DIR),
        help="Root directory containing Dubins HDF5 files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(BUFFERS_DIR),
        help="Output directory for buffer files",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="dubins_gap.h5",
        help="Specific HDF5 file to process (optional, processes all .h5 files if not specified)",
    )

    args = parser.parse_args()

    traj_dir = Path(args.root_dir)
    
    # Find HDF5 files to process
    if args.input_file:
        # Process specific file
        hdf5_files = [traj_dir / args.input_file]
        print('reading file: ', hdf5_files[0])
        if not hdf5_files[0].exists():
            print(f"Error: File {hdf5_files[0]} not found!")
            exit(1)
    else:
        # Process all HDF5 files in directory
        hdf5_files = list(Path(traj_dir).glob("*.h5"))
        if not hdf5_files:
            print(f"No .h5 files found in {traj_dir}")
            exit(1)
    
    print(f"Found {len(hdf5_files)} HDF5 files to process")
    
    output_path = Path(args.output_dir) / "buffer.pkl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    convert_trajectories(hdf5_files, output_path)
