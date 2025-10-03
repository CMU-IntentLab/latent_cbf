


python scripts/collect_trajs.py --n_trajectories 200 --save_images --filename "dubins_gap" && python visualize_trajs.py 

python scripts/visualize_trajs.py


conda activate data4robotics
python combine_traj_to_buffer_dubins.py --input_file dubins_gap.h5 --root_dir /data/dubins

nice -n 19 python finetune.py task.train_buffer.cam_indexes=[0]



python scripts/collect_trajs.py --controller diffusion --config diffusion --n_trajectories 20 --filename diffusion_trajectories --output_dir /data/dubins --save_images --verbose

python scripts/visualize_trajs.py --filepath /data/dubins/diffusion_trajectories.h5 