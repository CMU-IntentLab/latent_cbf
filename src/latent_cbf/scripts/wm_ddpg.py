import argparse
import os
import sys
import pprint

import gymnasium #as gym
import gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
pyhj_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '/PyHJ'))
sys.path.append(pyhj_dir)
print(sys.path)
from dreamerv3_torch import models
from dreamerv3_torch import tools
import ruamel.yaml as yaml
import wandb
from PyHJ.data import Collector, VectorReplayBuffer, BehaviorCollector
from PyHJ.env import DummyVectorEnv
from PyHJ.exploration import GaussianNoise
from PyHJ.trainer import offpolicy_trainer
from PyHJ.utils import TensorboardLogger, WandbLogger
from PyHJ.utils.net.common import Net
from PyHJ.utils.net.continuous import Actor, Critic
import PyHJ.reach_rl_gym_envs as reach_rl_gym_envs

from termcolor import cprint
from datetime import datetime
import pathlib
from pathlib import Path
import collections
from PIL import Image
import io
from PyHJ.data import Batch
import matplotlib.pyplot as plt
from configs import DreamerConfig, Config, get_diffusion_config
from dreamer_offline import make_dataset
from controllers.factory import create_controller_from_config





def main(exp_config, config):
    env = gymnasium.make(config.task, params = [config])
    config.num_actions = env.action_space.n if hasattr(env.action_space, "n") else env.action_space.shape[0]
    wm = models.WorldModel(env.observation_space_full, env.action_space, 0, config)

    ckpt_path = config.rssm_ckpt_path
    checkpoint = torch.load(ckpt_path)
    state_dict = {k[14:]:v for k,v in checkpoint['agent_state_dict'].items() if '_wm' in k}
    wm.load_state_dict(state_dict)
    wm.eval()

    dp = create_controller_from_config(exp_config)

    config.batch_size = 1
    config.batch_length = 3
    expert_eps = collections.OrderedDict()
    expert_val_eps = collections.OrderedDict()
    tools.fill_offline_dataset(config, expert_eps, expert_val_eps)
    offline_dataset = make_dataset(expert_eps, config)



    env.set_wm(wm, offline_dataset, config, dp)


    # check if the environment has control and disturbance actions:
    assert hasattr(env, 'action_space') #and hasattr(env, 'action2_space'), "The environment does not have control and disturbance actions!"
    config.state_shape = env.observation_space.shape or env.observation_space.n
    config.action_shape = env.action_space.shape or env.action_space.n
    config.max_action = env.action_space.high[0]



    train_envs = DummyVectorEnv(
        [lambda: gymnasium.make(config.task, params = [wm, offline_dataset, config, dp]) for _ in range(config.training_num)]
    )
    test_envs = DummyVectorEnv(
        [lambda: gymnasium.make(config.task, params = [wm, offline_dataset, config, dp]) for _ in range(config.test_num)]
    )


    # seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    train_envs.seed(config.seed)
    test_envs.seed(config.seed)
    # model

    actor_activation = torch.nn.ReLU
    critic_activation = torch.nn.ReLU

    if config.critic_net is not None:
        critic_net = Net(
            config.state_shape,
            config.action_shape,
            hidden_sizes=config.critic_net,
            activation=critic_activation,
            concat=True,
            device=config.device
        )
    else:
        # report error:
        raise ValueError("Please provide critic_net!")

    critic = Critic(critic_net, device=config.device).to(config.device)
    critic_optim = torch.optim.AdamW(critic.parameters(), lr=config.critic_lr, weight_decay=config.weight_decay_pyhj)

    log_path = None

    from PyHJ.policy import avoid_DDPGPolicy_annealing as DDPGPolicy

    print("DDPG under the Avoid annealed Bellman equation with no Disturbance has been loaded!")

    actor_net = Net(config.state_shape, hidden_sizes=config.control_net, activation=actor_activation, device=config.device)
    actor = Actor(
        actor_net, config.action_shape, max_action=config.max_action, device=config.device
    ).to(config.device)
    actor_optim = torch.optim.AdamW(actor.parameters(), lr=config.actor_lr)


    policy = DDPGPolicy(
    critic,
    critic_optim,
    tau=config.tau,
    gamma=config.gamma_pyhj,
    exploration_noise=GaussianNoise(sigma=config.exploration_noise),
    reward_normalization=config.rew_norm,
    estimation_step=config.n_step,
    action_space=env.action_space,
    actor=actor,
    actor_optim=actor_optim,
    actor_gradient_steps=config.actor_gradient_steps,
    )

    if config.no_gp:
        log_path = os.path.join(config.logdir, 'PyHJ', 'dubins-wm', 'nogp')
    else:
        log_path = os.path.join(config.logdir+'/PyHJ/gp')


    # collector
    train_collector = BehaviorCollector(
        policy,
        train_envs,
        VectorReplayBuffer(config.buffer_size, len(train_envs)),
        exploration_noise=True
    )
    test_collector = Collector(policy, test_envs)

    if config.warm_start_path is not None:
        policy.load_state_dict(torch.load(config.warm_start_path))
        config.kwargs = config.kwargs + "warmstarted"

    epoch = 0



    if config.continue_training_epoch is not None:
        epoch = config.continue_training_epoch
        policy.load_state_dict(torch.load(
            os.path.join(
                log_path+"/epoch_id_{}".format(epoch),
                "policy.pth"
            )
        ))


    if config.continue_training_logdir is not None:
        policy.load_state_dict(torch.load(config.continue_training_logdir))
        epoch = config.continue_training_epoch


    def save_best_fn(policy, epoch=epoch):
        torch.save(
            policy.state_dict(), 
            os.path.join(
                log_path+"/epoch_id_{}".format(epoch),
                "policy.pth"
            )
        )


    def stop_fn(mean_rewards):
        return False


    if not os.path.exists(log_path+"/epoch_id_{}".format(epoch)):
        print("Just created the log directory!")
        # print("log_path: ", log_path+"/epoch_id_{}".format(epoch))
        os.makedirs(log_path+"/epoch_id_{}".format(epoch))

        
    logger = None
    warmup = 1

    for iter in range(warmup+config.total_episodes):
        if iter  < warmup:
            policy._gamma = 0 # for warmup the value fn
            policy.warmup = True
        else:
            policy._gamma = config.gamma_pyhj
            policy.warmup = False

        if config.continue_training_epoch is not None:
            print("epoch: {}, remaining epochs: {}".format(epoch//config.epoch, config.total_episodes - iter))
        else:
            print("epoch: {}, remaining epochs: {}".format(iter, config.total_episodes - iter))
        epoch = epoch + config.epoch
        print("log_path: ", log_path+"/epoch_id_{}".format(epoch))
        if config.total_episodes > 1:
            writer = SummaryWriter(log_path+"/epoch_id_{}".format(epoch)) #filename_suffix="_"+timestr+"_epoch_id_{}".format(epoch))
        else:
            if not os.path.exists(log_path+"/total_epochs_{}".format(epoch)):
                print("Just created the log directory!")
                print("log_path: ", log_path+"/total_epochs_{}".format(epoch))
                os.makedirs(log_path+"/total_epochs_{}".format(epoch))
            writer = SummaryWriter(log_path+"/total_epochs_{}".format(epoch)) #filename_suffix="_"+timestr+"_epoch_id_{}".format(epoch))
        if logger is None:
            logger = WandbLogger()
            logger.load(writer)
        logger = TensorboardLogger(writer)
        
        # import pdb; pdb.set_trace()
        result = offpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        config.epoch,
        config.step_per_epoch,
        config.step_per_collect,
        config.test_num,
        config.batch_size_pyhj,
        update_per_step=config.update_per_step,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        logger=logger
        )
        
        save_best_fn(policy, epoch=epoch)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_gp", action="store_true", default=False)
    args = parser.parse_args()
    exp_config = get_diffusion_config()
    config = DreamerConfig()
    if args.no_gp:
        config.no_gp = True
    else:
        config.no_gp = False
    config.size = exp_config.environment.image_size
    config.turnRate = exp_config.max_angular_velocity
    main(exp_config, config)