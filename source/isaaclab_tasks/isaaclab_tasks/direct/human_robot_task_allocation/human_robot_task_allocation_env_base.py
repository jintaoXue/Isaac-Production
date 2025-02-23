# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

from isaaclab_assets.robots.cartpole import CARTPOLE_CFG

import isaaclab.sim as sim_utils
# from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform

from .human_robot_task_allocation_env_cfg import HRTaskAllocEnvCfg

# from isaacsim.core.utils.nucleus import get_assets_root_path
from isaacsim.core.utils.prims import delete_prim, get_prim_at_path, set_prim_visibility
import isaacsim.core.utils.stage as stage_utils
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.prims import RigidPrim, Articulation
from isaacsim.core.api.world import World

from .human_robot_task_manager import Materials, TaskManager
class HRTaskAllocEnvBase(DirectRLEnv):
    cfg: HRTaskAllocEnvCfg

    def __init__(self, cfg: HRTaskAllocEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        a = 1
        # self._cart_dof_idx, _ = self.cartpole.find_joints(self.cfg.cart_dof_name)
        # self._pole_dof_idx, _ = self.cartpole.find_joints(self.cfg.pole_dof_name)
        
        # self.joint_pos = self.cartpole.data.joint_pos
        # self.joint_vel = self.cartpole.data.joint_vel

    def _setup_scene(self):

        assert self.scene.num_envs == 1, "Temporary only support num_envs == 1"
        assert self.cfg._valid_train_cfg()
        self.cuda_device = torch.device(self.cfg.cuda_device_str)
        for i in range(self.scene.num_envs):
            sub_env_path = f"/World/envs/env_{i}"
            # the usd file already has a ground plane
            add_reference_to_stage(usd_path = self.cfg.asset_path, prim_path = sub_env_path + "/obj")
            # raw_ground_path = sub_env_path + "/obj" + "/GroundPlane"
            # ground_prim = self._stage.GetPrimAtPath(raw_ground_path)
            # set_prim_visibility(prim=ground_prim, visible=False)
            # if get_prim_at_path(raw_ground_path):
            #     delete_prim(raw_ground_path)
        # spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        
        # for debug, visualize only prims 
        # stage_utils.print_stage_prim_paths()

        cube_list, hoop_list, bending_tube_list, upper_tube_list, product_list = [],[],[],[],[]
        for i in range(self.cfg.n_max_product):
            cube, hoop, bending_tube, upper_tube, product = self.set_up_material(num=i)
            cube_list.append(cube)
            hoop_list.append(hoop)
            bending_tube_list.append(bending_tube)
            upper_tube_list.append(upper_tube)
            product_list.append(product)
        #materials states
        self.materials : Materials = Materials(cube_list=cube_list, hoop_list=hoop_list, bending_tube_list=bending_tube_list, upper_tube_list=upper_tube_list, product_list = product_list)
        '''for humans workers (characters), robots (agv+boxs) and task manager'''
        character_list =self.set_up_human(num=self.cfg.n_max_human)
        agv_list, box_list = self.set_up_robot(num=self.cfg.n_max_robot)
        self.task_manager : TaskManager = TaskManager(character_list, agv_list, box_list, self.cuda_device, self._train_cfg['params']['config'])
        
        self.initialize_pre_def_routes(from_file = True)
        self.reset_machine_state()
        '''max_env_length_settings'''
        self.max_env_length_settings = [[1250, 1250, 1250], [1000, 1000, 1000], [1000, 1000, 1000]]

        '''test settings'''
        if self._test and self._test_all_settings:
            self.test_all_idx = -1
            self.test_settings_list = []
            for w in range(self._train_cfg['params']['config']["max_num_worker"]):
                for r in range(self._train_cfg['params']['config']["max_num_robot"]):
                    for i in range(self._train_cfg['params']['config']['test_times']):  
                        self.test_settings_list.append((w+1,r+1))
        '''gantt chart'''
        self.actions_list = []
        self.time_frames = []
        self.gantt_charc = []
        self.gantt_agv = []
        
        # # clone and replicate
        # self.scene.clone_environments(copy_from_source=False)
        
        # add lights
        # light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        # light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = self.action_scale * actions.clone()

    def reset_machine_state(self):
        # conveyor
        #0 free 1 working
        self.convey_state = 0
        #cutting machine
        #to do 
        self.cutting_state_dic = {0:"free", 1:"work", 2:"reseting"}
        self.cutting_machine_state = 0
        self.c_machine_oper_time = 0
        self.c_machine_oper_len = 10
        #gripper
        speed = 0.6
        self.operator_gripper = torch.tensor([speed]*10, device='cuda:0')
        self.gripper_inner_task_dic = {0: "reset", 1:"pick_cut", 2:"place_cut_to_inner_station", 3:"place_cut_to_outer_station", 
                                    4:"pick_product_from_inner", 5:"pick_product_from_outer", 6:"place_product_from_inner", 7:"place_product_from_outer"}
        self.gripper_inner_task = 0
        self.gripper_inner_state_dic = {0: "free_empty", 1:"picking", 2:"placing"}
        self.gripper_inner_state = 0

        self.gripper_outer_task_dic = {0: "reset", 1:"pick_upper_tube_for_inner_station", 2:"pick_upper_tube_for_outer_station", 3:"place_upper_tube_to_inner_station", 4:"place_upper_tube_to_outer_station"}
        self.gripper_outer_task = 0
        self.gripper_outer_state_dic = {0: "free_empty", 1:"picking", 2:"placing"}
        self.gripper_outer_state = 0

        #welder 
        # self.max_speed_welder = 0.1
        self.welder_inner_oper_time = 0
        self.welder_outer_oper_time = 0
        self.welding_once_time = 20
        self.operator_welder = torch.tensor([0.4], device='cuda:0')
        self.welder_task_dic = {0: "reset", 1:"weld_left", 2:"weld_right", 3:"weld_middle",}
        self.welder_state_dic = {0: "free_empty", 1: "moving_left", 2:"welding_left", 3:"welded_left", 4:"moving_right",
                                 5:"welding_right", 6:"rotate_and_welding", 7:"welded_right", 8:"welding_middle" , 9:"welded_upper"}
        self.welder_inner_task = 0
        self.welder_inner_state = 0
        self.welder_outer_task = 0
        self.welder_outer_state = 0
        
        #station
        # self.welder_inner_oper_time = 10
        self.operator_station = torch.tensor([0.3, 0.3, 0.3, 0.3], device='cuda:0')
        self.station_task_left_dic = {0: "reset", 1:"weld"}
        self.station_state_left_dic = {0: "reset_empty", 1:"loading", 2:"rotating", 3:"waiting", 4:"welding", 5:"welded", 6:"finished", -1:"resetting"}
        self.station_task_inner_left = 0
        self.station_task_outer_left = 0
        self.station_state_inner_left = -1
        self.station_state_outer_left = -1

        self.station_middle_task_dic = {0: "reset", 1:"weld_left", 2:"weld_middle", 3:"weld_right"}
        self.station_state_middle_dic = {-1:"resetting", 0: "reset_empty", 1:"placing", 2:"placed", 3:"moving_left", 4:"welding_left", 
                                         5:"welded_left", 6:"welding_right", 7:"welded_right", 8:"welding_upper", 9:"welded_upper"}
        self.station_state_inner_middle = 0
        self.station_state_outer_middle = 0
        self.station_task_inner_middle = 0
        self.station_task_outer_middle = 0
        
        self.station_right_task_dic = {0: "reset", 1:"weld"}
        self.station_state_right_dic = {0: "reset_empty", 1:"placing", 2:"placed", 3:"moving", 4:"welding_right", -1:"resetting"}
        self.station_state_inner_right = 0
        self.station_state_outer_right = 0
        self.station_task_outer_right = 0
        self.station_task_inner_right = 0
        
        self.process_groups_dict = {}
        self.proc_groups_inner_list = []
        self.proc_groups_outer_list = []
        '''side table state'''
        self.depot_state_dic = {0: "empty", 1:"placing", 2: "placed"}
        # self.table_capacity = 4
        self.depot_hoop_set = set()
        self.depot_bending_tube_set = set()
        self.state_depot_hoop = 0
        self.state_depot_bending_tube = 0
        self.depot_product_set = set()
        '''progress step'''
        self.pre_progress_step = 0
        self.available_task_dic = {'none': -1}

        return


    def _set_up_machine(self):
        self.obj_belt_0 = RigidPrim(
            prim_paths_expr="/World/envs/.*/obj/ConveyorBelt_A09_0_0/Belt",
            name="ConveyorBelt_A09_0_0/Belt",
            track_contact_forces=True,
        )
        self.obj_0_1 = RigidPrim(
            prim_paths_expr="/World/envs/.*/obj/obj_0_1",
            name="obj_0_1",
            track_contact_forces=True,
        )
        self.obj_belt_1 = RigidPrim(
            prim_paths_expr="/World/envs/.*/obj/ConveyorBelt_A09_0_2/Belt",
            name="ConveyorBelt_A09_0_2/Belt",
            track_contact_forces=True,
        )
        self.obj_part_10 = Articulation(
            prim_paths_expr="/World/envs/.*/obj/part10", name="obj_part_10", reset_xform_properties=False
        )
        self.obj_part_7 = Articulation(
            prim_paths_expr="/World/envs/.*/obj/part7", name="obj_part_7", reset_xform_properties=False
        )
        self.obj_part_7_manipulator = RigidPrim(
            prim_paths_expr="/World/envs/.*/obj/part7/manipulator2/robotiq_arg2f_base_link", name="obj_part_7_manipulator", reset_xform_properties=False
        )
        self.obj_part_9_manipulator = RigidPrim(
            prim_paths_expr="/World/envs/.*/obj/part9/manipulator2/robotiq_arg2f_base_link", name="obj_part_9_manipulator", reset_xform_properties=False
        )
        self.obj_11_station_0 =  Articulation(
            prim_paths_expr="/World/envs/.*/obj/part11/node/Station0", name="obj_11_station_0", reset_xform_properties=False
        )
        self.obj_11_station_0_revolution = RigidPrim(
            prim_paths_expr="/World/envs/.*/obj/part11/node/Station0/revolution", name="Station0/revolution", reset_xform_properties=False
        )
        self.obj_11_station_1_revolution = RigidPrim(
            prim_paths_expr="/World/envs/.*/obj/part11/node/Station1/revolution", name="Station1/revolution", reset_xform_properties=False
        )
        self.obj_11_station_0_middle = RigidPrim(
            prim_paths_expr="/World/envs/.*/obj/part11/node/Station0/middle_left", name="Station0/middle_left", reset_xform_properties=False
        )
        self.obj_11_station_1_middle = RigidPrim(
            prim_paths_expr="/World/envs/.*/obj/part11/node/Station1/middle_left", name="Station1/middle_left", reset_xform_properties=False
        )
        self.obj_11_station_0_right = RigidPrim(
            prim_paths_expr="/World/envs/.*/obj/part11/node/Station0/right", name="Station0/right", reset_xform_properties=False
        )
        self.obj_11_station_1_right = RigidPrim(
            prim_paths_expr="/World/envs/.*/obj/part11/node/Station1/right", name="Station1/right", reset_xform_properties=False
        )
        self.obj_11_station_1 =  Articulation(
            prim_paths_expr="/World/envs/.*/obj/part11/node/Station1", name="obj_11_station_1", reset_xform_properties=False
        )
        self.obj_11_welding_0 =  Articulation(
            prim_paths_expr="/World/envs/.*/obj/part11/node/Welding0", name="obj_11_welding_0", reset_xform_properties=False
        )
        self.obj_11_welding_1 =  Articulation(
            prim_paths_expr="/World/envs/.*/obj/part11/node/Welding1", name="obj_11_welding_1", reset_xform_properties=False
        )
        self.obj_2_loader_0 =  Articulation(
            prim_paths_expr="/World/envs/.*/obj/part2/Loaders/Loader0", name="obj_2_loader_0", reset_xform_properties=False
        )
        self.obj_2_loader_1 =  Articulation(
            prim_paths_expr="/World/envs/.*/obj/part2/Loaders/Loader1", name="obj_2_loader_1", reset_xform_properties=False
        )

    def set_up_material(self, num):
        if num > 0:
            _str = ("{}".format(num)).zfill(2)
        else:
            _str = "0"
        materials_cube = RigidPrim(
            prim_paths_expr="/World/envs/.*/obj/Materials/cubes/cube_"+_str,
            name="cube_"+_str,
            track_contact_forces=True,
        )
        materials_hoop = RigidPrim(
            prim_paths_expr="/World/envs/.*/obj/Materials/hoops/hoop_"+_str,
            name="hoop_"+_str,
            track_contact_forces=True,
        )
        materials_bending_tube = RigidPrim(
            prim_paths_expr="/World/envs/.*/obj/Materials/bending_tubes/bending_tube_"+_str,
            name="bending_tube_"+_str,
            track_contact_forces=True,
        )
        materials_upper_tube = RigidPrim(
            prim_paths_expr="/World/envs/.*/obj/Materials/upper_tubes/upper_tube_"+_str,
            name="upper_tube_"+_str,
            track_contact_forces=True,
        )
        product = RigidPrim(
            prim_paths_expr="/World/envs/.*/obj/Materials/products/product_"+_str,
            name="product_"+_str,
            track_contact_forces=True,
        )
        return materials_cube, materials_hoop, materials_bending_tube, materials_upper_tube, product

    def set_up_human(self, num):

        character_list = []
        for i in range(1, num+1):

            _str = ("{}".format(i)).zfill(2)
            character = RigidPrim(
                prim_paths_expr="/World/envs/.*/obj/Characters/male_adult_construction_"+_str,
                name="character_{}".format(i+1),
                track_contact_forces=True,
            )
            character_list.append(character)

        return character_list 
    
    def set_up_robot(self, num):

        box_list = []
        robot_list = []
        for i in range(1, num+1):
            box = RigidPrim(
                prim_paths_expr="/World/envs/.*/obj/AGVs/box_0{}".format(i),
                name="box_{}".format(i),
                track_contact_forces=True,
            )
            box_list.append(box)

            agv = Articulation(
                prim_paths_expr="/World/envs/.*/obj/AGVs/agv_0{}".format(i),
                name="agv_{}".format(i),
                reset_xform_properties=False,
            )
            robot_list.append(agv)
        return robot_list, box_list 


