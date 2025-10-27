

# initial data collection for diffusion policy
python scripts/collect_trajs.py --n_trajectories 200 --save_images --filename "dubins_gap" && python scripts/visualize_trajs.py 

# prepping data for dp
python scripts/combine_traj_to_buffer_dubins.py --input_file dubins_gap.h5 --root_dir /data/dubins/trajs
python scripts/train_diffusion.py task.train_buffer.cam_indexes=[0]


# getting diffusion policy rollout data
python scripts/collect_trajs.py --controller diffusion --config diffusion --n_trajectories 20 --filename diffusion_trajectories --output_dir /data/dubins --save_images --verbose
# optional visualization
python scripts/visualize_trajs.py --filepath /data/dubins/diffusion_trajectories.h5 



# training wm
python scripts/combine_trajectories.py
python scripts/dreamer_offline.py

# training HJ
python scripts/wm_ddpg.py


# testing wm checkpoint
python scripts/collect_trajs.py --controller diffusion --use_wm_prediction --wm_checkpoint /data/dubins/test/dreamer/rssm_ckpt.pt --wm_history_length 8 --config diffusion_wm --filename wm_test_cbf --n_trajectories 5 --save_images

# visualize rollouts
python scripts/eval_dreamer.py