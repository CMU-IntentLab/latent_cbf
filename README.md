
cd src/latent_cbf

# initial data collection for diffusion policy
python scripts/collect_trajs.py --n_trajectories 200 --save_images --filename "dubins_gap" && python scripts/visualize_trajs.py 

# prepping data for dp
python scripts/combine_traj_to_buffer_dubins.py --input_file dubins_gap.h5 --root_dir /data/dubins/trajs

# train dp. this takes about 30 minutes on a 4090
python scripts/train_diffusion.py --buffer-path /data/dubins/buffers/buffer.pkl 


# getting diffusion policy rollout data.
# Note: This will start the policy at OOD initial conditions to generate exploratory actions. This will take about 0.5-1 hr on a 4090. 
scripts/collect_batch_diffusion.sh

# example optional visualization
python scripts/visualize_trajs.py --filepath /data/dubins/trajs/diffusion_trajectories_{ITER}.h5 


# training wm
python scripts/combine_trajectories.py

# This will take about 2 hours. Do not stress if the logged videos do not show the vehicle in the reconstructions, it usually takes about 6-8k iterations (20 min on a 4090) before the models learns to visualize the vehicle position.
python scripts/dreamer_offline.py

# Test the margin functions
python scripts/wm_trajectory_stats.py  --traj-h5 /data/dubins/trajs/diffusion_trajectories_19500.h5

# training HJ
python scripts/wm_ddpg.py
python scripts/wm_ddpg.py --no_gp False


# testing wm checkpoint
python scripts/collect_trajs.py --controller diffusion --use_wm_prediction --wm_checkpoint /data/dubins/dreamer/rssm_ckpt.pt --wm_history_length 8 --config diffusion_wm --filename wm_test_cbf --n_trajectories 5 --save_images --filter_mode cbf # can use lr or none

# visualize rollouts
python scripts/eval_dreamer.py --filter_mode cbf # can use lr or none