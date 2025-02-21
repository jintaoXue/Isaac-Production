# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
# from collections.abc import Sequence

# from isaaclab_assets.robots.cartpole import CARTPOLE_CFG

# import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
# from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
# from isaaclab.utils.math import sample_uniform

from .....isaaclab_assets.isaaclab_assets.robots import production_assets

@configclass
class HRTaskAllocEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 5.0
    action_space = 10
    #The real state/observation_space is complicated, settiing 2 is only for initializing gym Env
    observation_space = 2
    state_space = 2

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # machine, human, robot
    # obj_part10_cfg: ArticulationCfg = production_assets.obj_part10_cfg

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=4.0, replicate_physics=True)

