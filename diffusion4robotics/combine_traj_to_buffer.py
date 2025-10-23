import argparse
import cv2
import pickle
import numpy as np
from tqdm import tqdm
from typing import Any, List, Union

from pathlib import Path
from robobuf.buffers import ObsWrapper, Transition, ReplayBuffer
import json

#import pytorch3d.transforms as pt
import torch
import numpy as np
import functools
import h5py
from scipy.spatial.transform import Rotation as R

def eef_pose_to_state(T, gripper):
    # Extract translation
    x, y, z = T[:3, 3]
    
    # Extract rotation matrix and convert to quaternion
    rotation_matrix = T[:3, :3]
    quat = R.from_matrix(rotation_matrix).as_quat()  # Returns [qx, qy, qz, qw]
    
    # Concatenate into final state
    eef_state = np.concatenate(([x, y, z], quat, [gripper]))
    return eef_state

def extract_traj_from_hdf5(f):
    data = f['data']
    # For example, a dataset containing an array of dict-like steps
    traj_list = []
    for i in range(len(data["actions"])):
        eef_state = eef_pose_to_state(data['ee_states'][i].reshape(4,4).T, data["gripper_states"][i])
        state = {"cam_0": data["camera_0"][i],
            "cam_1": data["camera_1"][i],
            "state": eef_state,
            "joint_state": data["joint_states"][i]}
        traj_list.append([state, data["actions"][i], 0])
    return [traj_list]

def convert_trajectories(
    hdf5_files: List[Path], output_path: Path, image_postprocess: Any = None
):
    """
    Combine individual pickle trajectories into a single buffer
    """
    print(f"Working with {len(hdf5_files)} files and saving to {output_path}")

    out_buffer = None
    all_acs = []
    all_obs = []

    for file in tqdm(hdf5_files, total=len(hdf5_files)):
        with h5py.File(file, "r") as f:
            # Assuming each file contains a trajectory dataset
            # You need to adjust this based on your HDF5 structure
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
            all_obs.append(old_obs['state'])
            # delta pos, axis angle delta, gripper
            all_acs.append(action)

    all_acs_arr = np.array(all_acs)
    max_ac = np.max(all_acs_arr.copy(), axis=0)
    min_ac = np.min(all_acs_arr.copy(), axis=0)
    all_obs_arr = np.array(all_obs)
    max_ob = np.max(all_obs_arr.copy(), axis=0)
    min_ob = np.min(all_obs_arr.copy(), axis=0)

    for old_traj in tqdm(
        out_buffer.to_traj_list(),
        desc="Postprocessing",
        total=len(out_buffer._traj_starts),
    ):
        new_traj = ReplayBuffer()
        isFirst = True
        for i, (new_obs, action, reward) in enumerate(old_traj):
            
            orig_obs = new_obs['state'] 
            new_obs['state'] = (new_obs['state'] - min_ob)/(max_ob-min_ob)
            new_obs['state'] = new_obs['state']*2 - 1
            new_obs['state'][3:7] = orig_obs[3:7]
            action = (action - min_ac)/(max_ac-min_ac)
            action = action*2 - 1
            
            new_traj.add(
                Transition(ObsWrapper(new_obs), action, reward),
                is_first=(isFirst),
            )
            isFirst = False
        new_traj_list += new_traj.to_traj_list()
    out_buffer = ReplayBuffer.load_traj_list(new_traj_list)
    
    traj_list = out_buffer.to_traj_list()
    num_traj = len(traj_list[0])

    ob_dict={'maximum':list(max_ob), 'minimum':list(min_ob)}
    ac_dict={'maximum':list(max_ac), 'minimum':list(min_ac)}
    print(traj_list[0][-1][0]['state'])
    for i in range(num_traj):
        assert np.max(traj_list[0][i][0]['state']) <= 1        
        assert np.min(traj_list[0][i][0]['state']) >= -1 
        assert np.min(traj_list[0][i][1]) >= -1
        assert np.max(traj_list[0][i][1]) <= 1


    for k in ac_dict:
        ac_dict[k] = list(ac_dict[k])
        ob_dict[k] = list(ob_dict[k])
    ac_output_path = output_path.parent / "ac_norm.json"
    ob_output_path = output_path.parent / "ob_norm.json"
    with open(ac_output_path, 'w') as f:
        json.dump(ac_dict, f)
    with open(ob_output_path, 'w') as f:
        json.dump(ob_dict, f)


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
        default="/data",
        help="Root directory for separate trajectories",
    )

    args = parser.parse_args()

    demo_type = "real_data"
    image_postprocess = None

    traj_dir = Path(args.root_dir)
    hdf5_files = []    
    hdf5_files.extend(Path(traj_dir).glob("*.hdf5"))
    output_path = Path(
        args.root_dir,
        "buffers",
        "buffer.pkl",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    convert_trajectories(hdf5_files, output_path)

