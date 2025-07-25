Isaac Production

---

# Isaac Production
Isaac Production is a training platform for human-robot task allocation in manufacting. Built on [NVIDIA Isaac Lab](https://isaac-sim.github.io/IsaacLab/main/index.html) and [NVIDIA Isaac Sim](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)

## after deploying isaac-lab and create conda env
in isaac-production folder, link the isaac-sim repository:
ln -s ${HOME}/isaacsim _isaac_sim
pip install heapdict


# env part description
Part1 上料输送轨道+激光除锈工位+下料输送轨道
Part2 法兰料架x2，左+右
Part3 固定物体
Part4 龙门架静态
Part5 运料框
Part6 固定物体 气罐
Part7 龙门架，下料抓手
Part8 龙门架中间放料区
Part9 上料抓手 
Part10 激光切料区
Part11 激光焊接区


# todo
gantt chart
argparse

self.max_episode_length 是否可以修改? (no)

wandb 

修复自动补全

self.scheduler lr_scheduler schedule_type (请参考~/Repos/miniconda3/envs/isaac-lab/lib/python3.10/site-packages/rl_games/common/a2c_common.py)

env reset fialed: 解决办法，在gym register 部分order_enforce=False

test setting的num robot character 不一定对

## human fatigue 建模

## 加一个movement time 把map_route 函数加一下

## 加一个predict task max fatigue

## 修正一下task的持续时间 以及机器控制

## 如何修改代码
1.rule-based只用fatigue model的predcit 函数给到predict的结果，的到worker的task mask分别用于high-level decison 和low-level decision
2.基于cost function的，这个cost function 会收集所有同质/异质worker的生产状态，或者基于/结合粒子滤波，cost function的输出呢还是high-level task space，不过会有多个worker
3.在做决策的时候用综合的mask，mask掉不安全的输出

### 网络设计

### 下一步要改的细节
    1.如果疲劳程度超过1, 工人不管在做什么subtask任务，都会变成休息（会使问题，难度更大
    2.cost function预测的具体是什么，是完成某一个task的fatigue增加量
    3.cost function的loss函数
        可以分为连续性监督和离散型监督，加上action，一共三个loss函数
    4.cost function的数据收集工作（可以先用仿真器训练一个）
        task_clearing 是收集结束状态， assign_task 是开始状态
    5. 如果是异质的工人，怎么进行训练
 6. code evaluation epoch的加入
 7. fatigue的值验证
 8. fatigue 的曲线调整，以及训练的reward设计，env length调整
 9. worker 要改成异质的，首先就是要把疲劳参数和step函数做一个修改，然后是网络参数的输入要改，异质worker的初始化方式也要修改，奖励函数和环境的长度也要做修改


# 问题，supervise traning 存在过拟合

fix bug
神经网络修正
对于None action的选择

把预测值改成预测delta

# 5.4
wandb上数据的关键变化点可以记录在table里面

action的返回值里面再包括额外的信息
obs的extra也应该要包含返回的额外信息

# debug一下为什么单个worker的decision这么差
self.task_manager.characters.acti_num_charc
1
self.task_mask[1:]
tensor([1., 1., 0., 0., 0., 0., 1., 0., 0.])
q
tensor([[ -6.2242,  -6.2513,  -6.4300, -20.0000, -20.0000, -20.0000, -20.0000,
          -6.5274, -20.0000, -20.0000]], device='cuda:0')
action_mask
tensor([[1., 1., 1., 0., 0., 0., 0., 1., 0., 0.]], device='cuda:0')

# 调小 box capacity
hoop 为4
但是bending tube为2试试
num product为2

问题：机器人数量增加，效率反而下降了


# 6.16
要实现这个PPO-lag discrete的话就要修改储存的memory，修改网络结构，计算adv surrogate要有区别
要么就用PPO-penatly

还要把low-level改成nearest path的方式


# 6.19 各算法对比
PPO
https://arxiv.org/abs/1707.06347

ppolag_dis有两个版本：  
   1. 没有cost
   2. 直接penalty in reward，
   3. 用cost critic, 用lagrangian结合
   4. cost_mask (by cost critic, by task predicticve neural, by filter)

ppolag_filter通过predictive的方式，加上cost mask作为硬约束

EBQ同样也是，考虑加mask和不加mask的区别
    1. 没有cost
    2. cost penalty in reward
    3. cost critic penalty in critic
    4. cost_mask (by cost critic, by task predicticve neural, by filter)


setting3:
    fatigue coe是已知的，输入给网络
    fatigue coe是未知的，用rl filter


## 6.21 决定还是用rl filter的方式
对比算法实现

能确保算法训练符合故事

EBQ 先去掉cost 约束试试
    if done_flag[0]:

### 7.2 记得保存训练模型
fatigue 的参数值 完成时间要好好调整一下

## 7.10
%图片2 的红字对齐要改，ppo图片要改，序号不对，
comparison algorithm怎么设计？分为加不加safe set 还是说直接对比性能就好了


## 7.16
RL sutton 的书133页 + DDQN的 原文
https://arxiv.org/pdf/1509.06461


        {
            "name": "test: rl_filter filter headless wandb",
            "type": "python",
            "request": "launch",
            "args" : ["--task", "Isaac-TaskAllocation-Direct-v1", "--algo", "rl_filter", "--headless", "--wandb_activate", "True", "--test", "True", "--load_dir", 
            "/rl_filter_2025-07-20_12-17-12/nn", "--load_name", "/HRTA_direct_ep_82400.pth", "--wandb_project", "test_HRTA_fatigue", "--test_times", "10"],
            "program": "${workspaceFolder}/train.py",
            "console": "integratedTerminal",
            "justMyCode": true,
        },




#### 算法记录
对比算法：
1. D3QN 
nomask_4090_rl_filter_2025-07-25_15-02-16

2. PF-CD3Q
rl_filter_2025-07-20_12-17-12 4070

3. PF-CD3Q with penalty

4. PPO-lag
no_mask_ppolag_filter_dis_2025-07-23_22-24-04   4070

5. PF-PPO-lag
ppolag_filter_dis_2025-07-21_23-34-32  4070

6. PPO-dis with penalty

7. DQN with penalty

8. PF-DQN

对比网络：
1. PF-CD3Q 
2. 网络不加Fatigue的信息

对比filter的精确度
EKF
KF
PF

对比setting:
测试参数实时变化

程序的截止时间要改一下


PPO
https://arxiv.org/abs/1707.06347

ppolag_dis有两个版本：  
   1. 没有cost
   2. 直接penalty in reward，
   3. 用cost critic, 用lagrangian结合
   4. cost_mask (by cost critic, by task predicticve neural, by filter)

ppolag_filter通过predictive的方式，加上cost mask作为硬约束

EBQ同样也是，考虑加mask和不加mask的区别
    1. 没有cost
    2. cost penalty in reward
    3. cost critic penalty in critic
    4. cost_mask (by cost critic, by task predicticve neural, by filter)


setting3:
    fatigue coe是已知的，输入给网络
    fatigue coe是未知的，用rl filter


