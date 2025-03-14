


import numpy as np
import torch
import math
from ...utils import quaternion




######### for human fatigue #####

def random_zero_index(data):

    if data.count(0)>=1:
        indexs = np.argwhere(np.array(data) == 0)
        _idx = np.random.randint(low=0, high = len(indexs)) 
        return indexs[_idx][0]
    else:
        return -1

class Fatigue(object):

    def __init__(self, human_idx, human_type) -> None:
        #combine all the subtask and state
        self.phy_free_state_dic = {"free", "waiting_box"}
        self.psy_free_state_dic = {"free", "waiting_box"}
        self.phy_fatigue_ce_dic = {"approaching": 0.0001, "put_hoop_into_box": 0.1, "put_bending_tube_into_box": 0.2, 
                        'put_hoop_on_table': 0.1, 'put_bending_tube_on_table': 0.2, 'hoop_loading_inner': 0.05, "hoop_loading_outer": 0.05, 'bending_tube_loading_inner': 0.1, 
                        'bending_tube_loading_outer': 0.1, "cutting_cube": 0.01, "placing_product": 0.3}
        self.psy_fatigue_ce_dic = {"approaching": 0.0001, "put_hoop_into_box": 0.1, "put_bending_tube_into_box": 0.2, 
                        'put_hoop_on_table': 0.1, 'put_bending_tube_on_table': 0.2, 'hoop_loading_inner': 0.05, "hoop_loading_outer": 0.05, 'bending_tube_loading_inner': 0.1, 
                        'bending_tube_loading_outer': 0.1, "cutting_cube": 0.01, "placing_product": 0.3}
        self.phy_recovery_ce_dic = {"human_type_0": 0.1}
        self.psy_recovery_ce_dic = {"human_type_0": 0.1}
        scale = 0.1
        self.phy_fatigue_ce_dic = self.scale_coefficient(scale, self.phy_fatigue_ce_dic)
        self.psy_fatigue_ce_dic = self.scale_coefficient(scale, self.psy_fatigue_ce_dic)
        self.phy_recovery_ce_dic = self.scale_coefficient(scale, self.phy_recovery_ce_dic)
        self.psy_recovery_ce_dic = self.scale_coefficient(scale, self.psy_recovery_ce_dic)
        
        self.ONE_STEP_TIME = 1.0

        # self.device = cuda_device
        self.idx = human_idx
        
        self.phy_recovery_coefficient = self.phy_recovery_ce_dic[human_type]
        self.psy_recovery_coefficient = self.psy_recovery_ce_dic[human_type]
        self.phy_fatigue = None
        self.psy_fatigue = None
        self.pre_state_type = None
        self.state_history = None
        self.time_history = None 

        return
    
    def scale_coefficient(self, scale, dic : dict):
        return {key: (v * scale)  for (key, v) in dic.items()}
    
    def reset(self):
        self.phy_fatigue = 0
        self.psy_fatigue = 0
        self.pre_state_type = 'free'
        self.state_history = []
        self.time_history = [] 
        return
    
    def step(self, state_type, subtask):
        
        self.phy_fatigue = self.step_helper_phy(self.phy_fatigue, state_type, subtask)
        self.psy_fatigue = self.step_helper_psy(self.psy_fatigue, state_type, subtask)
        return
    
    def predict(self, subtask_list, time_steps_list):
        
        return

    def step_helper_phy(self, F_0, state_type, subtask):
        # forgetting-fatigue-recovery exponential model
        # paper name: Incorporating Human Fatigue and Recovery Into the Learning–Forgetting Process
        if state_type in self.phy_free_state_dic:
            F_0 = F_0*math.exp(-self.phy_recovery_coefficient*self.ONE_STEP_TIME)
        else:
            if state_type == "approaching":
                _lambda = -self.phy_fatigue_ce_dic[state_type]
            else:
                assert subtask in self.phy_fatigue_ce_dic.keys()
                _lambda = -self.phy_fatigue_ce_dic[subtask]
            F_0 = F_0 + (1-F_0)*(1-math.exp(_lambda*self.ONE_STEP_TIME))
        return F_0
    
    def step_helper_psy(self, F_0, state_type, subtask):
        # forgetting-fatigue-recovery exponential model
        # paper name: Incorporating Human Fatigue and Recovery Into the Learning–Forgetting Process
        if state_type in self.psy_free_state_dic:
            F_0 = F_0*math.exp(-self.psy_recovery_coefficient*self.ONE_STEP_TIME)
        else:
            if state_type == "approaching":
                _lambda = -self.psy_fatigue_ce_dic[state_type]
            else:
                assert subtask in self.psy_fatigue_ce_dic.keys()
                _lambda = -self.psy_fatigue_ce_dic[subtask]
            F_0 = F_0 + (1-F_0)*(1-math.exp(_lambda*self.ONE_STEP_TIME))
        return F_0
    


class Characters(object):

    def __init__(self, character_list) -> None:
        self.character_list = character_list
        self.state_character_dic = {0:"free", 1:"approaching", 2:"waiting_box", 3:"putting_in_box", 4:"putting_on_table", 5:"loading", 6:"cutting_machine"}
        self.task_range = {'hoop_preparing', 'bending_tube_preparing', 'hoop_loading_inner', 'bending_tube_loading_inner', 'hoop_loading_outer', 'bending_tube_loading_outer', "cutting_cube", 
                           'placing_product'}
        self.sub_task_character_dic = {0:"free", 1:"put_hoop_into_box", 2:"put_bending_tube_into_box", 3:"put_hoop_on_table", 4:"put_bending_tube_on_table", 
                                    5:'hoop_loading_inner', 6:'bending_tube_loading_inner', 7:'hoop_loading_outer', 8: 'bending_tube_loading_outer', 9: 'cutting_cube', 10:'placing_product'}
        
        self.low2high_level_task_dic = {"put_hoop_into_box":"hoop_preparing", "put_bending_tube_into_box":'bending_tube_preparing', "put_hoop_on_table":'hoop_preparing', 
                        "put_bending_tube_on_table":'bending_tube_preparing', 'hoop_loading_inner':'hoop_loading_inner', 'bending_tube_loading_inner':'bending_tube_loading_inner', 
                        'hoop_loading_outer':'hoop_loading_outer', 'bending_tube_loading_outer': 'bending_tube_loading_outer', 'cutting_cube': 'cutting_cube', 'placing_product':'placing_product'}
        
        self.poses_dic = {"put_hoop_into_box": [1.28376, 6.48821, np.deg2rad(0)] , "put_bending_tube_into_box": [1.28376, 13.12021, np.deg2rad(0)], 
                        "put_hoop_on_table": [-12.26318, 4.72131, np.deg2rad(0)], "put_bending_tube_on_table":[-32, 8.0, np.deg2rad(-90)],
                        'hoop_loading_inner':[-16.26241, 6.0, np.deg2rad(180)],'bending_tube_loading_inner':[-29.06123, 6.3725, np.deg2rad(0)],
                        'hoop_loading_outer':[-16.26241, 6.0, np.deg2rad(180)], 'bending_tube_loading_outer': [-29.06123, 6.3725, np.deg2rad(0)],
                        'cutting_cube':[-29.83212, -1.54882, np.deg2rad(0)], 'placing_product':[-40.47391, 12.91755, np.deg2rad(0)],
                        'initial_pose_0':[-11.5768, 6.48821, 0.0], 'initial_pose_1':[-30.516169, 7.5748153, 0.0]}
        
        self.poses_dic2num = {"put_hoop_into_box": 0 , "put_bending_tube_into_box": 1, 
                "put_hoop_on_table": 2, "put_bending_tube_on_table":3,
                'hoop_loading_inner':4,'bending_tube_loading_inner':5,
                'hoop_loading_outer':6, 'bending_tube_loading_outer': 7,
                'cutting_cube':8, 'placing_product':9,
                'initial_pose_0':10, 'initial_pose_1':11}
        
        self.routes_dic = None

        self.picking_pose_hoop = [1.28376, 6.48821, np.deg2rad(0)] 
        self.picking_pose_bending_tube = [1.28376, 13.12021, np.deg2rad(0)] 
        self.picking_pose_table_hoop = [-12.26318, 4.72131, np.deg2rad(0)]
        self.picking_pose_table_bending_tube = [-32, 8.0, np.deg2rad(-90)]

        self.loading_pose_hoop = [-16.26241, 6.0, np.deg2rad(180)]
        self.loading_pose_bending_tube = [-29.06123, 6.3725, np.deg2rad(0)]

        self.cutting_cube_pose = [-29.83212, -1.54882, np.deg2rad(0)]

        self.placing_product_pose = [-40.47391, 12.91755, np.deg2rad(0)]
        self.PUTTING_TIME = 5
        self.LOADING_TIME = 5
        
        self.fatigue_list : list[Fatigue] = []
        for i in range(0,len(self.character_list)):
            self.fatigue_list.append(Fatigue(i, 'human_type_0'))
        return
    
    def reset(self, acti_num_charc = None, random = None):
        if acti_num_charc is None:
            acti_num_charc = np.random.randint(1, 4)
        self.acti_num_charc = acti_num_charc
        self.states = [0]*acti_num_charc
        self.tasks = [0]*acti_num_charc
        self.list = self.character_list[:acti_num_charc]
        self.x_paths = [[] for i in range(acti_num_charc)]
        self.y_paths = [[] for i in range(acti_num_charc)]
        self.yaws = [[] for i in range(acti_num_charc)]
        self.path_idxs = [0 for i in range(acti_num_charc)]
        if random is None:
            random = np.random.choice(len(self.poses_dic), acti_num_charc, replace=False)
        pose_list = list(self.poses_dic.values())
        pose_str_list = list(self.poses_dic.keys())
        initial_pose_str = []
        for i in range(0, acti_num_charc):
            position = pose_list[random[i]][:2]+[0.0415]
            initial_pose_str.append(pose_str_list[random[i]])
            self.list[i].set_world_poses(torch.tensor([position]), torch.tensor([[1., 0., 0., 0.]]))
            self.list[i].set_velocities(torch.zeros((1,6)))
            self.reset_idx(i)
            self.reset_path(i)
        self.loading_operation_time_steps = [0 for i in range(acti_num_charc)]

        for i in range(0, acti_num_charc):
            fatigue : Fatigue = self.fatigue_list[i]
            fatigue.reset()

        return initial_pose_str


    def reset_idx(self, idx):
        if idx < 0 :
            return
        self.states[idx] = 0
        self.tasks[idx] = 0

    def assign_task(self, high_level_task, random = False):
        #todo 
        if high_level_task not in self.task_range:
            return -2
        if random:
            idx = random_zero_index(self.tasks)
        else: 
            idx = self.find_available_charac()

        if idx == -1:
            return idx
        if high_level_task == 'hoop_preparing':
            # idx = self.find_available_charac()
            self.tasks[idx] = 1 
        elif high_level_task == 'bending_tube_preparing':
            # idx = self.find_available_charac()
            self.tasks[idx] = 2
        elif high_level_task == 'hoop_loading_inner':
            self.tasks[idx] = 5
        elif high_level_task == 'bending_tube_loading_inner':
            self.tasks[idx] = 6
        elif high_level_task == 'hoop_loading_outer':
            self.tasks[idx] = 7
        elif high_level_task == 'bending_tube_loading_outer':
            self.tasks[idx] = 8
        elif high_level_task == 'cutting_cube':
            if random:
                self.tasks[idx] = 9
            #TODO warning
            else:
                for _idx in range(0, len(self.list)):
                    xyz, _ = self.list[_idx].get_world_poses()
                    if self.tasks[_idx] == 0 and xyz[0][0] < -22:
                        self.tasks[_idx] = 9
                        return _idx
                    else:
                        self.tasks[idx] = 9
                        return idx
            # if self.tasks[1] == 0: #only assign worker 1 to do the cutting cube task 
            #     self.tasks[1] = 9
            #     idx = 1
            # else:
            #     return -1
        elif high_level_task == 'placing_product':
            self.tasks[idx] = 10
        return idx
    
    def find_available_charac(self, idx=0):
        try:
            return self.tasks.index(idx)
        except: 
            return -1

    def step_next_pose(self, charac_idx = 0):
        reaching_flag = False
        #skip the initial pose
        # if len(self.x_paths[agv_idx]) == 0:
        #     position = [current_pose[0], current_pose[1], 0]
        #     euler_angles = [0,0, current_pose[2]]
        #     return position, quaternion.eulerAnglesToQuaternion(euler_angles), True

        self.path_idxs[charac_idx] += 1
        path_idx = self.path_idxs[charac_idx]
        # if agv_idx == 0:
        #     a = 1
        if path_idx == (len(self.x_paths[charac_idx]) - 1):
            reaching_flag = True
            position = [self.x_paths[charac_idx][-1], self.y_paths[charac_idx][-1], 0]
            euler_angles = [0,0, self.yaws[charac_idx][-1]]
        else:
            position = [self.x_paths[charac_idx][path_idx], self.y_paths[charac_idx][path_idx], 0]
            euler_angles = [0,0, self.yaws[charac_idx][path_idx]]

        orientation = quaternion.eulerAnglesToQuaternion(euler_angles)
        return position, orientation, reaching_flag
    
    def step_fatigue(self, idx, state, task):
        state_type = self.state_character_dic[state]
        task_type = self.sub_task_character_dic[task]
        fatigue : Fatigue = self.fatigue_list[idx]
        fatigue.step(state_type, task_type)

    def get_fatigue(self, idx):
        fatigue : Fatigue = self.fatigue_list[idx]
        return fatigue.phy_fatigue, fatigue.psy_fatigue
    
    def reset_path(self, charac_idx):
        self.x_paths[charac_idx] = []
        self.y_paths[charac_idx] = []
        self.yaws[charac_idx] = []
        self.path_idxs[charac_idx] = 0

    def low2high_level_task_mapping(self, task):
        task = self.sub_task_character_dic[task]
        if task in self.low2high_level_task_dic.keys():
            return self.low2high_level_task_dic[task]
        else: return -1