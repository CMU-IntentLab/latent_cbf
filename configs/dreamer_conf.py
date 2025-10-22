from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
@dataclass
class DreamerConfig:    
    seed: int = 0
    deterministic_run: bool = False
    logdir: str = '/data/dubins/test/dreamer'
    steps: int = 40_000
    eval_every: int = 1_000
    eval_episode_num: int = 10
    log_every: int = 1_000
    reset_every: int =  0
    device: str = 'cuda:0'
    compile: bool = True
    precision: int =  32
    debug: bool =  False
    video_pred_log: bool =  True
    action_repeat: int = 1

    dataset_path: str = '/data/dubins/buffers/dreamer_buffer.h5'
    num_train_trajs: float = 0.8
    
    # World Model Prediction Settings
    wm_checkpoint_path: str = '/data/dubins/test/dreamer/rssm_ckpt.pt'
    use_wm_prediction: bool = True
    wm_history_length: int = 8  # Number of timesteps of history to use for prediction

    dyn_hidden: int = 512
    dyn_deter: int = 512
    dyn_stoch: int = 32
    dyn_discrete: int = 0 
    dyn_rec_depth: int = 1
    dyn_mean_act: str = 'none'
    dyn_std_act: str = 'sigmoid2'
    dyn_min_std: float = 0.1
    units: int = 512
    act: str ='SiLU'
    norm: bool = True
    dyn_scale: float = 0.5
    rep_scale: float = 0.1
    kl_free: float = 1.0
    weight_decay: float = 0.0
    unimix_ratio: float = 0.01
    initial: str = 'learned'
    
    batch_size: int = 32
    batch_length: int = 16
    train_ratio: int = 64
    model_lr: float = 1e-4
    opt_eps: float = 1e-8
    grad_clip: int = 1000
    dataset_size: int = 1_000_000
    opt: str = 'adam'


    encoder: Dict[str, Any] = field(default_factory=lambda:{'mlp_keys': 'obs_state', 'cnn_keys': 'image', 'act': 'SiLU', 'norm': True, 'cnn_depth': 32, 'kernel_size': 4, 'minres': 4, 'mlp_layers': 5, 'mlp_units': 1024, 'symlog_inputs': True})
    decoder: Dict[str, Any] = field(default_factory=lambda:{'mlp_keys': 'obs_state', 'cnn_keys': 'image', 'act': 'SiLU', 'norm': True, 'cnn_depth': 32, 'kernel_size': 4, 'minres': 4, 'mlp_layers': 5, 'mlp_units': 1024, 'cnn_sigmoid': False, 'image_dist': 'mse', 'vector_dist': 'symlog_mse', 'outscale': 1.0})
    actor: Dict[str, Any] = field(default_factory=lambda:{'layers': 2, 'dist': 'normal', 'entropy': 3e-4, 'unimix_ratio': 0.01, 'std': 'learned', 'min_std': 0.1, 'max_std': 1.0, 'temp': 0.1, 'lr': 3e-5, 'eps': 1e-5, 'grad_clip': 100.0, 'outscale': 1.0})
    critic: Dict[str, Any] = field(default_factory=lambda:{'layers': 2, 'dist': 'symlog_disc', 'slow_target': True, 'slow_target_update': 1, 'slow_target_fraction': 0.02, 'lr': 3e-5, 'eps': 1e-5, 'grad_clip': 100.0, 'outscale': 0.0})
    cont_head:  Dict[str, Any] = field(default_factory=lambda:{'layers': 2, 'loss_scale': 1.0, 'outscale': 1.0})
    margin_head:  Dict[str, Any] = field(default_factory=lambda:{'layers': 2, 'loss_scale': 1.0})
    grad_heads: List[str] = field(default_factory=lambda: ['decoder', 'cont'])


    discount: float = 0.997
    discount_lambda: float = 0.95
    imag_horizon: int = 15
    imag_gradient: str = 'dynamics'
    imag_gradient_mix: float =  0.0
    eval_state_mean: bool = False
    
    gamma_lx: float = 0.75
    debug: bool = False
    gradient_thresh: float = 0.1
    zs_weight: float = 0.1
    relu_weight: float = 1.0
    gp_weight: float = 10.0


    reward_threshold: Optional[float] = None
    buffer_size: int = 40000
    actor_lr: float = 1e-4
    critic_lr: float = 1e-3
    gamma_pyhj: float = 0.9999 # type=float, default=0.95)
    tau: float = 0.005 # type=float, default=0.005)
    exploration_noise: float = 0.1 # type=float, default=0.1)
    epoch: int = 1 # type=int, default=10)
    total_episodes: int = 20 # type=int, default=160)
    step_per_epoch: int = 40000 # type=int, default=40000)
    step_per_collect: int = 8 # type=int, default=8)
    update_per_step: float = 0.125 # type=float, default=0.125)
    batch_size_pyhj: int = 512 # type=int, default=512)
    control_net: List[int] = field(default_factory=lambda: [ 512, 512, 512]) # type=int, nargs="*", default=None) # for control policy
    critic_net: List[int] = field(default_factory=lambda: [512, 512, 512])  # type=int, nargs="*", default=None) # for critic net
    training_num: int = 1 # type=int, default=8)
    test_num: int = 1 # type=int, default=100)
    render: float = 0. # type=float, default=0.)
    rew_norm: bool = False # action="store_true", default=False)
    n_step: int = 1 # type=int, default=1)
    continue_training_logdir: Optional[str] = None # type=str, default=None)
    continue_training_epoch: Optional[int] = None # type=int, default=None)
    actor_gradient_steps: int = 1 # type=int, default=1)
    is_game_baseline: bool = False # type=bool, default=False) # it will be set automatically
    target_update_freq: int = 400 # type=int, default=400)
    auto_alpha: float = 1
    alpha_lr: float = 3e-4
    alpha: float = 0.2
    weight_decay_pyhj: float = 0.001
    actor_activation: str = "ReLU" #type=str, default="ReLU")
    critic_activation: str = "ReLU"
    warm_start_path: Optional[str] = None # type=str, default=None)
    kwargs: Dict[str, Any] = field(default_factory=lambda: {}) # type=str, default="")

    gamma_lx: float = 0.75
    offline_data_path: str = '/home/kensuke/ManiSkill/examples/baselines/ppo/runs/BlockTopple-v0__ppo_rgb__1__1753308792/test_videos/trajectory.rgb.pd_ee_delta_pose.physx_cuda.h5'
    pretrain: int = 500
    hybrid_steps: int = 1_000_000
    hybrid: bool = True


    no_gp: bool = False
    rssm_ckpt_path: str = "/data/dubins/test/dreamer/rssm_ckpt.pt"
    filter_directory_gp: str = '/data/dubins/test/dreamer/PyHJ/gp/epoch_id_15/policy.pth'
    filter_directory_nogp: str = '/data/dubins/test/dreamer/PyHJ/nogp/epoch_id_15/policy.pth'
    num_runs: int = 1
    cbf_gamma: float = 0.9
    lr_thresh: float = 0.3
    filter_mode: str = 'cbf' # 'cbf' or 'lr' or 'none'

    task: str = 'dubins-wm'
    '''# Dreamer config
    parallel: bool = True
    eval_every: int = 10_000
    eval_episode_num: int = 10
    log_every: int = 10_000
    reset_every: int =  0
    device: str = 'cuda:0'
    compile: bool = True
    precision: int =  32
    debug: bool =  False
    video_pred_log: bool =  True
    action_repeat: int = 1
    steps: int = 10_000_000

    #time_limit: int = 1e3
    offline_traindir: str = ''
    offline_evaldir: str = ''

    dyn_hidden: int = 512
    dyn_deter: int = 512
    dyn_stoch: int = 32
    dyn_discrete: int = 32
    dyn_rec_depth: int = 1
    dyn_mean_act: str = 'none'
    dyn_std_act: str = 'sigmoid2'
    dyn_min_std: float = 0.1
    units: int = 512
    act: str ='SiLU'
    norm: bool = True
    dyn_scale: float = 0.5
    rep_scale: float = 0.1
    kl_free: float = 1.0
    weight_decay: float = 0.0
    unimix_ratio: float = 0.01
    initial: str = 'learned'



    # Exploration
    expl_behavior: str = 'greedy'
    expl_until: int = 1000
    expl_extr_scale: float = 0.0
    expl_intr_scale: float = 1.0
    disag_target: str = 'stoch'
    disag_log: bool =True
    disag_models: int = 10
    disag_offset: int = 1
    disag_layers: int = 4
    disag_units: int = 400
    disag_action_cond: bool = False


    batch_size: int = 32
    batch_length: int = 16
    train_ratio: int = 64
    model_lr: float = 1e-4
    opt_eps: float = 1e-8
    grad_clip: int = 1000
    dataset_size: int = 1_000_000
    opt: str = 'adam'


    time_limit: int = 100
    grayscale: bool = False
    prefill: int = 2500
    reward_EMA: bool = True

    # Behavior.
    discount: float = 0.997
    discount_lambda: float = 0.95
    imag_horizon: int = 15
    imag_gradient: str = 'dynamics'
    imag_gradient_mix: float =  0.0
    eval_state_mean: bool = False


    encoder: Dict[str, Any] = field(default_factory=lambda:{'mlp_keys': 'state', 'cnn_keys': '.*\_cam$', 'act': 'SiLU', 'norm': True, 'cnn_depth': 32, 'kernel_size': 4, 'minres': 4, 'mlp_layers': 5, 'mlp_units': 1024, 'symlog_inputs': True})
    decoder: Dict[str, Any] = field(default_factory=lambda:{'mlp_keys': 'state', 'cnn_keys': '.*\_cam$', 'act': 'SiLU', 'norm': True, 'cnn_depth': 32, 'kernel_size': 4, 'minres': 4, 'mlp_layers': 5, 'mlp_units': 1024, 'cnn_sigmoid': False, 'image_dist': 'mse', 'vector_dist': 'symlog_mse', 'outscale': 1.0})
    actor: Dict[str, Any] = field(default_factory=lambda:{'layers': 2, 'dist': 'normal', 'entropy': 3e-4, 'unimix_ratio': 0.01, 'std': 'learned', 'min_std': 0.1, 'max_std': 1.0, 'temp': 0.1, 'lr': 3e-5, 'eps': 1e-5, 'grad_clip': 100.0, 'outscale': 1.0})
    critic: Dict[str, Any] = field(default_factory=lambda:{'layers': 2, 'dist': 'symlog_disc', 'slow_target': True, 'slow_target_update': 1, 'slow_target_fraction': 0.02, 'lr': 3e-5, 'eps': 1e-5, 'grad_clip': 100.0, 'outscale': 0.0})
    reward_head:  Dict[str, Any] = field(default_factory=lambda:{'layers': 2, 'dist': 'symlog_disc', 'loss_scale': 1.0, 'outscale': 0.0})
    cont_head:  Dict[str, Any] = field(default_factory=lambda:{'layers': 2, 'loss_scale': 1.0, 'outscale': 1.0})
    margin_head:  Dict[str, Any] = field(default_factory=lambda:{'layers': 2, 'loss_scale': 1.0})
    grad_heads: List[str] = field(default_factory=lambda: ['decoder', 'reward', 'cont'])



    reward_threshold: Optional[float] = None
    buffer_size: int = 40000
    actor_lr: float = 1e-4
    critic_lr: float = 1e-3
    gamma_pyhj: float = 0.9999 # type=float, default=0.95)
    tau: float = 0.005 # type=float, default=0.005)
    exploration_noise: float = 0.1 # type=float, default=0.1)
    epoch: int = 1 # type=int, default=10)
    total_episodes: int = 20 # type=int, default=160)
    step_per_epoch: int = 40000 # type=int, default=40000)
    step_per_collect: int = 8 # type=int, default=8)
    update_per_step: float = 0.125 # type=float, default=0.125)
    batch_size_pyhj: int = 512 # type=int, default=512)
    control_net: List[int] = field(default_factory=lambda: [ 512, 512, 512]) # type=int, nargs="*", default=None) # for control policy
    critic_net: List[int] = field(default_factory=lambda: [512, 512, 512])  # type=int, nargs="*", default=None) # for critic net
    training_num: int = 1 # type=int, default=8)
    test_num: int = 1 # type=int, default=100)
    render: float = 0. # type=float, default=0.)
    rew_norm: bool = False # action="store_true", default=False)
    n_step: int = 1 # type=int, default=1)
    continue_training_logdir: Optional[str] = None # type=str, default=None)
    continue_training_epoch: Optional[int] = None # type=int, default=None)
    actor_gradient_steps: int = 1 # type=int, default=1)
    is_game_baseline: bool = False # type=bool, default=False) # it will be set automatically
    target_update_freq: int = 400 # type=int, default=400)
    auto_alpha: float = 1
    alpha_lr: float = 3e-4
    alpha: float = 0.2
    weight_decay_pyhj: float = 0.001
    actor_activation: str = "ReLU" #type=str, default="ReLU")
    critic_activation: str = "ReLU"
    warm_start_path: Optional[str] = None # type=str, default=None)
    kwargs: Dict[str, Any] = field(default_factory=lambda: {}) # type=str, default="")

    gamma_lx: float = 0.75
    offline_data_path: str = '/home/kensuke/ManiSkill/examples/baselines/ppo/runs/BlockTopple-v0__ppo_rgb__1__1753308792/test_videos/trajectory.rgb.pd_ee_delta_pose.physx_cuda.h5'
    pretrain: int = 500
    hybrid_steps: int = 1_000_000
    hybrid: bool = True


    use_gp: bool = False
    wm_directory: str = "/home/kensuke/WM_CBF/ManiSkill/examples/baselines/dreamerv3-torch/runs/wm_edit/wm_lz.pt"
    filter_directory_nogp: str = '/home/kensuke/WM_CBF/ManiSkill/examples/baselines/dreamerv3-torch/LCRL/no_gp/noise_0.1_actor_lr_0.0001_critic_lr_0.001_batch_512_step_per_epoch_40000_kwargs_{}_seed_1/epoch_id_20/policy.pth'
    filter_directory_gp: str = '/home/kensuke/WM_CBF/ManiSkill/examples/baselines/dreamerv3-torch/LCRL/gp/noise_0.1_actor_lr_0.0001_critic_lr_0.001_batch_512_step_per_epoch_40000_kwargs_{}_seed_1/epoch_id_20/policy.pth'
    filter_thresh: float = 0.2
    num_runs: int = 1
    cbf_gamma: float = 0.7
    filter_mode: str = 'cbf' # 'cbf' or 'least_restrictive' or 'lr'''