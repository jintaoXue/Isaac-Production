

import random
import numpy as np
import torch
import math
from ...utils import quaternion
from .ekf_filter import EkfFatigue, EKfRecover
from .pf_filter import ParticleFilter, RecParticleFilter
from .eg_hrta_env_cfg import HRTaskAllocEnvCfg, high_level_task_dic, high_level_task_rev_dic, BoxCapacity
import random


######### for human fatigue #####

def random_zero_index(data : np.ndarray):
    zero_data = data == 0
    if np.count_nonzero(zero_data)>=1:
        indexs = np.argwhere(zero_data)
        _idx = np.random.randint(low=0, high = len(indexs)) 
        return indexs[_idx][0]
    else:
        return -1

class Fatigue(object):

    def __init__(self, human_idx, human_types) -> None:
        #task_human_subtasks_dic
        #"approaching" subtask in ommitted as it is high dynamic and hard to caculate
        self.cfg = HRTaskAllocEnvCfg()
        self.hyper_param_time = self.cfg.hyper_param_time
        self.task_human_subtasks_dic =  {'none': ['free'], 'hoop_preparing': ['put_hoop_into_box', 'put_hoop_on_table']*BoxCapacity.hoop, 
            'bending_tube_preparing': ['put_bending_tube_into_box','put_bending_tube_on_table']*BoxCapacity.bending_tube, 
            'hoop_loading_inner': ['hoop_loading_inner'], 'bending_tube_loading_inner': ['bending_tube_loading_inner'], 
            'hoop_loading_outer': ['hoop_loading_outer'], 'bending_tube_loading_outer':['bending_tube_loading_outer'], 
            'cutting_cube':['cutting_cube'], 'collect_product':['free'], 'placing_product':['placing_product']*BoxCapacity.product}
        
        self.phy_free_state_dic = {"free", "waiting_box", "approaching"}
        self.psy_free_state_dic = {"free", "waiting_box", "approaching"}
        #coefficient dic: combine all the subtask and state
        self.raw_phy_fatigue_ce_dic = {"free": None, "waiting_box": None, "approaching": None, "put_hoop_into_box": 0.04, "put_bending_tube_into_box": 0.06, 
                        'put_hoop_on_table': 0.04, 'put_bending_tube_on_table': 0.06, 'hoop_loading_inner': 0.12, "hoop_loading_outer": 0.12, 'bending_tube_loading_inner': 0.15, 
                        'bending_tube_loading_outer': 0.15, "cutting_cube": 0.01, "placing_product": 0.15}
        self.raw_psy_fatigue_ce_dic = {"free": None, "waiting_box": None, "approaching": None, "put_hoop_into_box": 0.1, "put_bending_tube_into_box": 0.15, 
                        'put_hoop_on_table': 0.1, 'put_bending_tube_on_table': 0.15, 'hoop_loading_inner': 0.05, "hoop_loading_outer": 0.05, 'bending_tube_loading_inner': 0.1, 
                        'bending_tube_loading_outer': 0.1, "cutting_cube": 0.01, "placing_product": 0.3}
        self.raw_phy_recovery_ce_dic = {"free": 0.05, "waiting_box": 0.05, "approaching": 0.02}
        self.raw_psy_recovery_ce_dic = {"free": 0.05, "waiting_box": 0.05, "approaching": 0.02}
        self.human_types = human_types
        self.human_type_coe_dic = {"strong": 0.8, "normal": 1.0, "weak": 1.2}

        self.ONE_STEP_TIME = 0.1
        self.ftg_thresh_phy = self.cfg.ftg_thresh_phy
        self.ftg_thresh_psy = self.cfg.ftg_thresh_psy
        self.ftg_task_mask = None
        
        # self.device = cuda_device
        self.idx = human_idx
        # self.phy_recovery_coefficient = self.phy_recovery_ce_dic[human_type]
        # self.psy_recovery_coefficient = self.psy_recovery_ce_dic[human_type]
        
        self.phy_fatigue = None
        self.psy_fatigue = None
        self.time_step = None
        self.state_subtask_history = None
        self.state_task_history = None
        self.phy_history = None # value, state, time
        self.psy_history = None
        # self.time_history = None 

        return
    
    def have_overwork(self):
        return self.phy_fatigue>self.ftg_thresh_phy or self.psy_fatigue>self.ftg_thresh_psy

    def scale_coefficient(self, scale, dic : dict):
        return {key: (v * scale if v is not None else None)  for (key, v) in dic.items()}
    
    def add_coefficient_randomness(self, scale, dic : dict):
        _dict = {}
        for (key, v) in dic.items():
            
            _dict[key] = (v + v*np.random.uniform(-scale, scale)) if v is not None else None

        return _dict

    def get_phy_fatigue_coe(self):
        if self.cfg.use_partial_filter:
            return list(self.pfs_phy_fat_ce_dic.values())[3:]
        return list(self.phy_fatigue_ce_dic.values())[3:]
    
    def reset(self):
        # if self.time_step is not None and self.time_step > 100:
        #     self.plot_curve()
        #     if self.cfg.use_partial_filter:
        #         for k, v in self.phy_fatigue_ce_dic.items():
        #             if v is not None:
        #                 filter : ParticleFilter = self.pfs_phy_fat[k]
        #                 filter.plot_results(filter.times, filter.F_estimates, filter.lambda_estimates, 'fatigue_' + k)
        #         R_filter : RecParticleFilter = self.pfs_phy_rec['free']
        #         R_filter.plot_results(R_filter.times, R_filter.F_estimates, R_filter.lambda_estimates, name='recover')
        self.time_step = 0
        self.phy_fatigue = 0
        self.psy_fatigue = 0
        self.pre_state_type = 'free'
        self.phy_history = [(0, self.time_step)] # value, time_step
        self.psy_history = [(0, self.time_step)]
        self.state_subtask_history = [('free', 'free', self.phy_fatigue, self.psy_fatigue, self.time_step)] #state, subtask, time_step 
        self.state_task_history = [('free', 'free', self.phy_fatigue, self.psy_fatigue, self.time_step)] #state, subtask, time_step 
        
        scale_phy = 3
        scale_psy = 0.5
        scale_phy_recover= 0.3
        self.human_type = random.choice(self.human_types)
        self.phy_fatigue_ce_dic = self.scale_coefficient(scale_phy*self.human_type_coe_dic[self.human_type], self.raw_phy_fatigue_ce_dic)
        self.psy_fatigue_ce_dic = self.scale_coefficient(scale_psy, self.raw_psy_fatigue_ce_dic)
        self.phy_recovery_ce_dic = self.scale_coefficient(scale_phy_recover, self.raw_phy_recovery_ce_dic)
        self.psy_recovery_ce_dic = self.scale_coefficient(scale_psy, self.raw_psy_recovery_ce_dic)

        random_percent = 0.4
        self.pfs_phy_fat_ce_dic = self.add_coefficient_randomness(random_percent, self.phy_fatigue_ce_dic)
        self.pfs_phy_rec_ce_dic = self.add_coefficient_randomness(random_percent, self.phy_recovery_ce_dic)
        self.pfs_phy_fat = {}
        self.pfs_phy_rec = {}

        for (key, v) in self.phy_fatigue_ce_dic.items():
            if v is not None:
                # self.pfs_phy_fat[key] = EkfFatigue(dt=1, num_steps=100, true_lambda=v, F0=0, Q=np.diag([0.01, 0.0001]), R=np.array([[0.1]]), x0=np.array([0., 0.1]), P0=np.diag([1.0, 1.0]))
                self.pfs_phy_fat[key] = ParticleFilter(dt=0.1, num_steps=100, true_lambda=v, F0=0, num_particles=500, sigma_w=0.01, sigma_v=0.001, lamda_init = v, upper_bound=v*(1+random_percent), lower_bound=v*(1-random_percent))
                # self.pfs_phy_fat[key] = ParticleFilter(dt=0.1, num_steps=100, true_lambda=v, F0=0, num_particles=500, sigma_w=0.01, sigma_v=0.001, lamda_init = v, upper_bound=v*(1+random_percent), lower_bound=v*(1+random_percent))
        
        for (key, v) in self.phy_recovery_ce_dic.items():
            if v is not None:
                # self.pfs_phy_rec[key] = EKfRecover(dt=0.1, num_steps=100, true_mu=v, R0=0, Q=np.diag([0.01, 0.0001]), R=np.array([[0.1]]), x0=np.array([0., 0.1]), P0=np.diag([1.0, 1.0])) 
                self.pfs_phy_rec[key] = RecParticleFilter(dt=0.1, num_steps=100, true_lambda=v, F0=0, num_particles=500, sigma_w=1e-2, sigma_v=1e-3, lamda_init = v, upper_bound=v*(1+random_percent), lower_bound=v*(1-random_percent)) 

        self.task_phy_prediction_dic = {task: 0.  for (key, task) in high_level_task_dic.items()} 
        self.task_psy_prediction_dic = {task: 0.  for (key, task) in high_level_task_dic.items()} 
        self.task_phy_prediction_dic = self.update_predict_dic()
        self.task_filter_phy_prediction_dic = self.update_filter_predict_dic()
        self.update_ftg_mask()
        self.ftg_task_mask = torch.ones(len(high_level_task_dic))

        return

    def step(self, state_type, subtask, task, ftg_prediction = None):
        if self.cfg.use_partial_filter == True:
            self.step_pfs(self.phy_fatigue, state_type, subtask, self.ONE_STEP_TIME)
            recover_coe_accuracy = self.get_filter_recover_coe_accuracy()
        self.phy_fatigue = self.step_helper_delta_phy_fatigue(self.phy_fatigue, state_type, subtask, self.ONE_STEP_TIME,  self.phy_fatigue_ce_dic, self.phy_recovery_ce_dic, self.phy_free_state_dic)
        # self.psy_fatigue = self.step_helper_psy(self.psy_fatigue, state_type, subtask, self.ONE_STEP_TIME)
        self.time_step += 1
        self.phy_history.append((self.phy_fatigue, self.time_step))
        self.psy_history.append((self.psy_fatigue, self.time_step))
        pre_state_type, pre_subtask, _ , _, _= self.state_subtask_history[-1]
        if pre_state_type != state_type or pre_subtask != subtask:
            self.state_subtask_history.append((state_type, subtask, self.phy_fatigue, self.psy_fatigue, self.time_step)) #state, subtask, time_step
        _, pre_task, _ , _, _= self.state_task_history[-1]
        if pre_task != task:
            self.state_task_history.append((state_type, task, self.phy_fatigue, self.psy_fatigue, self.time_step)) #state, subtask, time_step
        
        self.task_phy_prediction_dic = self.update_predict_dic()
        self.task_filter_phy_prediction_dic = self.update_filter_predict_dic()
        if self.cfg.use_partial_filter == True:
            self.update_ftg_mask(self.task_filter_phy_prediction_dic)
        else:
            self.update_ftg_mask(ftg_prediction)

        return

    def step_pfs(self, F, state_type, subtask, step_time):
        
        if state_type in self.phy_free_state_dic:
            _filter : ParticleFilter = self.pfs_phy_rec[state_type]
            # _filter : EKfRecover = self.pfs_phy_rec[state_type]
            # if self.time_step != _filter.prev_time_step + 1:
            #     _filter.reinit(self.time_step, F, self.pfs_phy_rec_ce_dic[state_type])
            # else:
                # _filter.step(F, F, self.time_step)
            F = F*math.exp(-self.phy_recovery_ce_dic[state_type]*step_time)
            _filter.step(F, F, self.time_step)
            self.pfs_phy_rec_ce_dic[state_type] = _filter.lambda_estimates[-1]
        else:
            assert subtask in self.phy_fatigue_ce_dic.keys()
            _filter : ParticleFilter = self.pfs_phy_fat[subtask]
            _lambda = -self.phy_fatigue_ce_dic[subtask]
            F = F + (1-F)*(1-math.exp(_lambda*step_time))
            _filter.step(F, F, self.time_step)
            self.pfs_phy_fat_ce_dic[subtask] = _filter.lambda_estimates[-1]

        return
     
    def get_filter_recover_coe_accuracy(self):
        true_coe = np.array(list(self.phy_recovery_ce_dic.values()))
        filter_prediction = np.array(list(self.pfs_phy_rec_ce_dic.values()))
        return np.sqrt(np.square((true_coe - filter_prediction)/true_coe).mean())
    
    def get_filter_fatigue_coe_accuracy(self):
        none_type_num = 3
        true_coe = np.array(list(self.phy_fatigue_ce_dic.values())[none_type_num:])
        filter_prediction = np.array(list(self.pfs_phy_fat_ce_dic.values())[none_type_num:])
        return np.sqrt(np.square((true_coe - filter_prediction)/true_coe).mean())
    
    def update_ftg_mask(self, prediction : dict = None):

        if prediction is not None:
            #adopt rule based mask
            _fatigue = np.array(list(prediction.values())) + self.phy_fatigue
            _mask = np.where(_fatigue < self.ftg_thresh_phy, 1, 0)
            self.ftg_task_mask = torch.from_numpy(_mask) 
            self.ftg_task_mask[0] = 1
        else:
            pass
        
        return

    def update_predict_dic(self):
        # step_time_scale = (1+self.hyper_param_time*math.log(1+self.phy_fatigue))
        # self.task_phy_prediction_dic = {task: self.phy_fatigue  for (key, task) in high_level_task_dic.items()} 
        # self.task_psy_prediction_dic = {task: self.psy_fatigue  for (key, task) in high_level_task_dic.items()} 
        # for key, v in self.task_phy_prediction_dic.items():
        #     subtask_seq = self.task_human_subtasks_dic[key]
        #     for subtask in subtask_seq:
        #         time = self.ONE_STEP_TIME
        #         if 'put' in subtask or subtask == 'placing_product':
        #             time = self.cfg.human_putting_time * self.ONE_STEP_TIME * step_time_scale
        #         elif 'loading' in subtask:
        #             time = self.cfg.human_loading_time * self.ONE_STEP_TIME * step_time_scale
        #         elif subtask == 'cutting_cube':
        #             time = self.cfg.cutting_machine_oper_len * self.ONE_STEP_TIME * step_time_scale
        #         self.task_phy_prediction_dic[key] = self.step_helper_phy(self.task_phy_prediction_dic[key], subtask, subtask, time)
        #         self.task_psy_prediction_dic[key] = self.step_helper_psy(self.task_psy_prediction_dic[key], subtask, subtask, time)
        #     self.task_phy_prediction_dic[key] -= self.phy_fatigue
        #     self.task_psy_prediction_dic[key] -= self.psy_fatigue
        phy_predict = self.update_predict_helper(self.phy_fatigue_ce_dic, self.phy_recovery_ce_dic, self.phy_free_state_dic)
        return phy_predict
    
    def update_filter_predict_dic(self):

        filter_phy_predict = self.update_predict_helper(self.pfs_phy_fat_ce_dic, self.pfs_phy_rec_ce_dic, self.phy_free_state_dic)

        return filter_phy_predict

    def update_predict_helper(self, phy_ce_dic, phy_recover_ce_dic, phy_free_state_dic):
        step_time_scale = (1+self.hyper_param_time*math.log(1+self.phy_fatigue))
        phy_predict_dic = {task: self.phy_fatigue  for (key, task) in high_level_task_dic.items()}
        # psy_predict_dic = {task: self.psy_fatigue  for (key, task) in high_level_task_dic.items()} 
        for key, v in phy_predict_dic.items():
            subtask_seq = self.task_human_subtasks_dic[key]
            for subtask in subtask_seq:
                time = self.ONE_STEP_TIME
                if 'put' in subtask or subtask == 'placing_product':
                    time = self.cfg.human_putting_time * self.ONE_STEP_TIME * step_time_scale
                elif 'loading' in subtask:
                    time = self.cfg.human_loading_time * self.ONE_STEP_TIME * step_time_scale
                elif subtask == 'cutting_cube':
                    time = self.cfg.cutting_machine_oper_len * self.ONE_STEP_TIME * step_time_scale
                phy_predict_dic[key] = self.step_helper_delta_phy_fatigue(phy_predict_dic[key], subtask, subtask, time, phy_ce_dic, phy_recover_ce_dic, phy_free_state_dic)
                # psy_predict_dic[key] = self.step_helper_delta_psy_fatigue(psy_predict_dic[key], subtask, subtask, time, psy_ce_dic, psy_recover_ce_dic, phy_free_state_dic, psy_free_state_dic)
            phy_predict_dic[key] -= self.phy_fatigue
            # psy_predict_dic[key] -= self.psy_fatigue
        return phy_predict_dic


    def step_helper_delta_phy_fatigue(self, F_0, state_type, subtask, step_time, fatigue_coe_dic, recover_coe_dic, free_state_dic):
        # forgetting-fatigue-recovery exponential model
        # paper name: Incorporating Human Fatigue and Recovery Into the Learning–Forgetting Process
        if state_type in free_state_dic:
            F_0 = F_0*math.exp(-recover_coe_dic[state_type]*step_time)
        else:
            assert subtask in fatigue_coe_dic.keys()
            _lambda = -fatigue_coe_dic[subtask]
            F_0 = F_0 + (1-F_0)*(1-math.exp(_lambda*step_time))
        return F_0
    
    # def step_helper_phy(self, F_0, state_type, subtask, step_time, fatigue_coe, revcover_coe):
    #     # forgetting-fatigue-recovery exponential model
    #     # paper name: Incorporating Human Fatigue and Recovery Into the Learning–Forgetting Process
    #     if state_type in self.phy_free_state_dic:
    #         F_0 = F_0*math.exp(-self.phy_recovery_ce_dic[state_type]*step_time)
    #     else:
    #         assert subtask in self.phy_fatigue_ce_dic.keys()
    #         _lambda = -self.phy_fatigue_ce_dic[subtask]
    #         F_0 = F_0 + (1-F_0)*(1-math.exp(_lambda*step_time))
    #     return F_0
    
    # def step_helper_psy(self, F_0, state_type, subtask, step_time):
    #     # forgetting-fatigue-recovery exponential model
    #     # paper name: Incorporating Human Fatigue and Recovery Into the Learning–Forgetting Process
    #     if state_type in self.psy_free_state_dic:
    #         F_0 = F_0*math.exp(-self.psy_recovery_ce_dic[state_type]*step_time)
    #     else:
    #         assert subtask in self.psy_fatigue_ce_dic.keys()
    #         _lambda = -self.psy_fatigue_ce_dic[subtask]
    #         F_0 = F_0 + (1-F_0)*(1-math.exp(_lambda*step_time))
    #     return F_0
    
    def plot_curve(self):
        import matplotlib.pyplot as plt
        plt.style.use('seaborn-v0_8-white')
        # plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams['pdf.fonttype'] = 42
        # plt.tick_params(axis='both', labelsize=50)
        params = {'legend.fontsize': 15,
            'legend.handlelength': 2}
        plt.rcParams.update(params)
        # plt.rcParams['xtick.labelsize'] = 14  # x轴刻度标签字体大小
        # plt.rcParams['ytick.labelsize'] = 14  # y轴刻度标签字体大小
        fig = plt.figure(figsize=(20,10), dpi=100)
        # gs = gridspec(1,4, )
        # gs = fig.add_gridspec(1,4) 
        ax = plt.subplot(111)
        # ax_2 = plt.subplot(212)
        ax.set_title('Fatigue curve, Human type:' + self.human_type, fontsize=20)
        ax.set_xlabel('time step', fontsize=15)
        ax.tick_params(axis='both', which='both', labelsize=15)
        line_labels = ['Physiological fatigue', 'Psychological fatigue']
        data = [self.phy_history, self.psy_history]
        color_dict = {'Physiological fatigue': 'crimson', 'Psychological fatigue': 'orange', 'EDQN2': 'forestgreen', 'EBQ-G': 'dodgerblue', 'EBQ-N': 'palevioletred', 'EBQ-GN':'blueviolet', "NoSp": 'silver'}
        for _data, line_label in zip(data, line_labels):
            _data = np.array(_data)
            x,y = _data[:, 1], _data[:, 0]
            ax.plot(x, y, '-', color=color_dict[line_label], label=line_label, ms=5, linewidth=2, marker='.', linestyle='dashed')
        
        vlines = np.array(self.state_task_history)[:, -1]
        ax.vlines(vlines.astype(np.int32), 0, 1, linestyles='dashed', colors='silver')
        ax.legend()
        
        plt.tight_layout()
        plt.show()
        # path = os.path.dirname(__file__)
        # fig.savefig('{}.pdf'.format(path + '/' + 'polyline'), bbox_inches='tight')
    


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
        _cfg = HRTaskAllocEnvCfg()
        self.cfg = _cfg
        self.PUTTING_TIME = _cfg.human_putting_time
        self.LOADING_TIME = _cfg.human_loading_time
        self.CUTTING_MACHINE_TIME = _cfg.cutting_machine_oper_len
        self.RANDOM_TIME = _cfg.human_time_random
        self.hyper_param_time = _cfg.hyper_param_time

        self.n_max_human = _cfg.n_max_human
        self.fatigue_list : list[Fatigue] = []
        self.human_types = ["strong", "normal", "weak"]
        for i in range(0,len(self.character_list)):
            self.fatigue_list.append(Fatigue(i, self.human_types))
        self.fatigue_task_masks = None
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
        for i in range(0, len(self.character_list)):
            if i < acti_num_charc:
                position = pose_list[random[i]][:2]+[0.0415]
                initial_pose_str.append(pose_str_list[random[i]])
                self.character_list[i].set_world_poses(torch.tensor([position]), torch.tensor([[1., 0., 0., 0.]]))
                self.character_list[i].set_velocities(torch.zeros((1,6)))
                self.reset_idx(i)
                self.reset_path(i)
            else:
                position = [0, 0, -100]
                self.character_list[i].set_world_poses(torch.tensor([position]), torch.tensor([[1., 0., 0., 0.]]))
                self.character_list[i].set_velocities(torch.zeros((1,6)))
        
        self.loading_operation_time_steps = [0. for i in range(acti_num_charc)]
        
        #1 is avaiable, 0 means worker is over fatigue threshold
        self.fatigue_task_masks = torch.zeros((self.n_max_human, len(high_level_task_dic)), dtype=torch.int32)
        for i in range(0, acti_num_charc):
            fatigue : Fatigue = self.fatigue_list[i]
            fatigue.reset()
            self.fatigue_task_masks[i] = fatigue.ftg_task_mask
        self.cost_mask_from_net = None
        return initial_pose_str

    # def get_fatigue_task_masks(self):
    #     fatigue_task_masks = torch.zeros((self.n_max_human, len(high_level_task_dic)), device=self.cfg.cuda_device_str)
    #     for i in range(0, self.acti_num_charc):
    #         fatigue_task_masks[i] = self.fatigue_list[i].ftg_task_mask
    #     return fatigue_task_masks

    def reset_idx(self, idx):
        if idx < 0 :
            return
        self.states[idx] = 0
        self.tasks[idx] = 0

    def assign_task(self, high_level_task, random = False):
        #todo 
        if high_level_task not in self.task_range:
            return -2

        # _fatigue_mask = self.fatigue_task_masks[:self.acti_num_charc, _fatigue_mask_idx]
        #task == 0 means the human doing no task, is free
        # np_task = np.array(self.tasks)
        # np_task = np.where(_fatigue_mask, np_task, -1)
        worker_tasks = self.tasks
        _fatigue_mask_idx = high_level_task_rev_dic[high_level_task] + 1
        if self.cfg.use_partial_filter:
            _fatigue_mask = self.fatigue_task_masks[:self.acti_num_charc, _fatigue_mask_idx].tolist()
            worker_tasks = [self.tasks[i] if _fatigue_mask[i] else -1 for i in range(len(_fatigue_mask))]
        elif self.cost_mask_from_net is not None:
            assert len(self.cost_mask_from_net) == 1, "error cost mask shape from cost function"
            _fatigue_mask = self.cost_mask_from_net[0, _fatigue_mask_idx, :]
            _fatigue_mask[self.acti_num_charc:] = 0
            worker_tasks = [ self.tasks[i] if _fatigue_mask[i].item() else -1 for i in range(len(_fatigue_mask))]
        
        # elif self.cfg.use_partial_filter:
            
        if random:
            idx = random_zero_index(worker_tasks)
        else:
            idx = self.find_available_charac(worker_tasks)
            
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
                    if worker_tasks[_idx] == 0 and xyz[0][0] < -22:
                        self.tasks[_idx] = 9
                        return _idx
                self.tasks[idx] = 9
        elif high_level_task == 'placing_product':
            self.tasks[idx] = 10
        
        return idx
    
    def find_available_charac(self, mask : list, idx=0):
        try:
            return mask.index(idx)
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
    
    def step_processing(self, idx):
        fatigue : Fatigue = self.fatigue_list[idx]
        step_time = 1/(1+self.hyper_param_time*math.log(1+fatigue.phy_fatigue)) 
        return step_time

    def step_fatigue(self, idx, state, subtask, task, ftg_prediction = None):
        
        state_type = self.state_character_dic[state]
        subtask = self.sub_task_character_dic[subtask]
        fatigue : Fatigue = self.fatigue_list[idx]
        fatigue.step(state_type, subtask, task, ftg_prediction)
        self.fatigue_task_masks[idx] = fatigue.ftg_task_mask

    def get_fatigue(self, idx):
        fatigue : Fatigue = self.fatigue_list[idx]
        return fatigue.phy_fatigue, fatigue.psy_fatigue
    
    def have_overwork(self):
        for i in range(self.acti_num_charc):
            if self.fatigue_list[i].have_overwork():
                return True
        return False
    
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