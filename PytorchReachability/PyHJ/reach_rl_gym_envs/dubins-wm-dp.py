from os import path
from typing import Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import io
from PIL import Image
import matplotlib.patches as patches
import torch
import math
class Dubins_WM_DP_Env(gym.Env):
    # TODO: 1. baseline over approximation; 2. our critic loss drop faster 
    def __init__(self, params):
        
        if len(params) == 1:
            config = params[0]
        else:
            wm = params[0]
            past_data = params[1]
            config = params[2]
            dp = params[3]
            self.set_wm(wm, past_data, config, dp)

        self.render_mode = None
        self.device = 'cuda:0'
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(544,), dtype=np.float32)
        image_size = config.size[0] #128
        img_obs_space = gym.spaces.Box(
                low=0, high=255, shape=(image_size, image_size, 3), dtype=np.uint8
            )
        obs_space = gym.spaces.Box(
                low=-1., high=1., shape=(2,), dtype=np.float32
            )
        bool_space = gym.spaces.Box(
                low=0., high=1., shape=(1,)
            )
        self.observation_space_full = gym.spaces.Dict({
            'image': img_obs_space,
            'obs_state': obs_space,
            'is_first': bool_space,
            'is_last': bool_space,
            'is_terminal': bool_space,
        })
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32) # joint action space
        self.image_size=config.size[0]
        self.turnRate = config.turnRate
        self.no_gp = config.no_gp

    def set_wm(self, wm, past_data, config, dp):
        self.device = config.device
        self.encoder = wm.encoder.to(self.device)
        self.wm = wm.to(self.device)
        self.data = past_data
        self.dp = dp
        if config.dyn_discrete:
            self.feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            self.feat_size = config.dyn_stoch + config.dyn_deter
    
    def step(self, action):
        ac = self.action_buffer.pop(0)
        if ac is not None:
            action = np.array([ac])/self.turnRate

        assert action <= 1, f"raw {ac}, new {action}"
        assert action >= -1, f"raw {ac}, new {action}"

        init = {k: v[:, -1] for k, v in self.latent.items()}
        ac_torch = torch.tensor([[action]], dtype=torch.float32).to(self.device)*self.turnRate
        self.latent = self.wm.dynamics.imagine_with_action(ac_torch, init)
        rew, cont = self.safety_margin(self.latent) # rew is negative if unsafe
        
        self.feat = self.wm.dynamics.get_feat(self.latent).detach().cpu().numpy()

        if len(self.action_buffer) == 0:
            truncated = True
        else:
            truncated = False
        terminated = False


        ac_unnorm = action
        info = {"is_first":False, "is_terminal":terminated, 'action': ac_unnorm}
        return np.copy(self.feat), rew, terminated, truncated, info
    
    def reset(self, initial_state=None,seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
       
        
        init_traj = next(self.data)
        
        if np.random.rand() < 0.5:
            dp_img = init_traj['image'][0,-1]
            info = {'agent_orientation': init_traj['state'][0,-1]}
            ac = self.dp.compute_action(info, dp_img)
            self.action_buffer = list(np.concatenate([[ac]] + self.dp.action_buffer))
        else:
            self.action_buffer = [None]*8

        data = self.wm.preprocess(init_traj)
        embed = self.encoder(data)
        self.latent, _ = self.wm.dynamics.observe(
            embed, data["action"], data["is_first"]
        )

        for k, v in self.latent.items(): 
            self.latent[k] = v[:, [-1]]
        self.feat = self.wm.dynamics.get_feat(self.latent).detach().cpu().numpy() 
        return np.copy(self.feat), {"is_first": True, "is_terminal": False}
      

    def safety_margin(self, state):
        g_xList = []
        
        feat = self.wm.dynamics.get_feat(state).detach()
        cont = self.wm.heads["cont"](feat)

        if self.no_gp:
            with torch.no_grad():
                    outputs = torch.tanh(self.wm.heads["margin_nogp"](feat))
                    g_xList.append(outputs.detach().cpu().numpy())
        else:
            with torch.no_grad():
                outputs = torch.tanh(self.wm.heads["margin_gp"](feat))
                g_xList.append(outputs.detach().cpu().numpy())
        safety_margin = np.array(g_xList).squeeze()

        return safety_margin, cont.mean.squeeze().detach().cpu().numpy()
    