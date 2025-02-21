# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Human-robot task allocation for production environment.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-TaskAllocation-Direct-v0",
    entry_point=f"{__name__}.human_robot_task_allocation_env_base:HRTaskAllocEnvBase",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.human_robot_task_allocation_env_cfg:HRTaskAllocEnvCfg",
        "ppo": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        # "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CartpolePPORunnerCfg",
        # "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        # "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

