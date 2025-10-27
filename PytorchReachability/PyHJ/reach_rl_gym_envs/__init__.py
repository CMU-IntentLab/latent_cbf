"""Registers the internal gym envs then loads the env plugins for module using the entry point."""
from typing import Any

from gymnasium.envs.registration import (
    load_plugin_envs,
    make,
    pprint_registry,
    register,
    registry,
    spec,
)




register(
    id="dubins-wm",
    entry_point="PyHJ.reach_rl_gym_envs.dubins-wm-dp:Dubins_WM_DP_Env",
    max_episode_steps=16,
    reward_threshold=1e8,
)

