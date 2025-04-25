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

num worker:1, num agv&box:1, env_length:2539, max_env_len:5000, task_finished:True
num worker:1, num agv&box:1, env_length:2932, max_env_len:5000, task_finished:True
num worker:1, num agv&box:1, env_length:2967, max_env_len:5000, task_finished:True
num worker:1, num agv&box:1, env_length:2735, max_env_len:5000, task_finished:True
num worker:1, num agv&box:1, env_length:2583, max_env_len:5000, task_finished:True
num worker:1, num agv&box:1, env_length:2944, max_env_len:5000, task_finished:True
num worker:1, num agv&box:1, env_length:2581, max_env_len:5000, task_finished:True
num worker:1, num agv&box:1, env_length:2763, max_env_len:5000, task_finished:True
num worker:1, num agv&box:1, env_length:2903, max_env_len:5000, task_finished:True
num worker:1, num agv&box:1, env_length:2720, max_env_len:5000, task_finished:True
num worker:1, num agv&box:1, env_length:2715, max_env_len:5000, task_finished:True
num worker:1, num agv&box:1, env_length:2577, max_env_len:5000, task_finished:True
num worker:1, num agv&box:1, env_length:2600, max_env_len:5000, task_finished:True
num worker:1, num agv&box:1, env_length:2922, max_env_len:5000, task_finished:True
num worker:1, num agv&box:1, env_length:2780, max_env_len:5000, task_finished:True

num worker:1, num agv&box:2, env_length:2537, max_env_len:5000, task_finished:True
num worker:1, num agv&box:2, env_length:2729, max_env_len:5000, task_finished:True
num worker:1, num agv&box:2, env_length:2593, max_env_len:5000, task_finished:True
num worker:1, num agv&box:2, env_length:2569, max_env_len:5000, task_finished:True
num worker:1, num agv&box:2, env_length:2583, max_env_len:5000, task_finished:True
num worker:1, num agv&box:2, env_length:2576, max_env_len:5000, task_finished:True
num worker:1, num agv&box:2, env_length:2579, max_env_len:5000, task_finished:True
num worker:1, num agv&box:2, env_length:2568, max_env_len:5000, task_finished:True
num worker:1, num agv&box:2, env_length:2701, max_env_len:5000, task_finished:True
num worker:1, num agv&box:2, env_length:2590, max_env_len:5000, task_finished:True
num worker:1, num agv&box:2, env_length:2682, max_env_len:5000, task_finished:True
num worker:1, num agv&box:2, env_length:2714, max_env_len:5000, task_finished:True
num worker:1, num agv&box:2, env_length:2550, max_env_len:5000, task_finished:True

num worker:1, num agv&box:3, env_length:2607, max_env_len:5000, task_finished:True
num worker:1, num agv&box:3, env_length:2537, max_env_len:5000, task_finished:True
num worker:1, num agv&box:3, env_length:2729, max_env_len:5000, task_finished:True
num worker:1, num agv&box:3, env_length:2593, max_env_len:5000, task_finished:True
num worker:1, num agv&box:3, env_length:2569, max_env_len:5000, task_finished:True
num worker:1, num agv&box:3, env_length:2583, max_env_len:5000, task_finished:True
num worker:1, num agv&box:3, env_length:2576, max_env_len:5000, task_finished:True
num worker:1, num agv&box:3, env_length:2579, max_env_len:5000, task_finished:True
num worker:1, num agv&box:3, env_length:2568, max_env_len:5000, task_finished:True
num worker:1, num agv&box:3, env_length:2701, max_env_len:5000, task_finished:True
num worker:1, num agv&box:3, env_length:2590, max_env_len:5000, task_finished:True
num worker:1, num agv&box:3, env_length:2682, max_env_len:5000, task_finished:True
num worker:1, num agv&box:3, env_length:2714, max_env_len:5000, task_finished:True
num worker:1, num agv&box:3, env_length:2550, max_env_len:5000, task_finished:True

num worker:2, num agv&box:1, env_length:1098, max_env_len:5000, task_finished:True
num worker:2, num agv&box:1, env_length:1118, max_env_len:5000, task_finished:True
num worker:2, num agv&box:1, env_length:1328, max_env_len:5000, task_finished:True
num worker:2, num agv&box:1, env_length:1193, max_env_len:5000, task_finished:True
num worker:2, num agv&box:1, env_length:1167, max_env_len:5000, task_finished:True
num worker:2, num agv&box:1, env_length:1312, max_env_len:5000, task_finished:True
num worker:2, num agv&box:1, env_length:1149, max_env_len:5000, task_finished:True
num worker:2, num agv&box:1, env_length:1162, max_env_len:5000, task_finished:True
num worker:2, num agv&box:1, env_length:1272, max_env_len:5000, task_finished:True
num worker:2, num agv&box:1, env_length:1126, max_env_len:5000, task_finished:True
num worker:2, num agv&box:1, env_length:1313, max_env_len:5000, task_finished:True
num worker:2, num agv&box:1, env_length:1175, max_env_len:5000, task_finished:True
num worker:2, num agv&box:1, env_length:1206, max_env_len:5000, task_finished:True
num worker:2, num agv&box:1, env_length:1108, max_env_len:5000, task_finished:True


num worker:2, num agv&box:2, env_length:1111, max_env_len:5000, task_finished:True
num worker:2, num agv&box:2, env_length:1096, max_env_len:5000, task_finished:True
num worker:2, num agv&box:2, env_length:1171, max_env_len:5000, task_finished:True
num worker:2, num agv&box:2, env_length:1102, max_env_len:5000, task_finished:True
num worker:2, num agv&box:2, env_length:1083, max_env_len:5000, task_finished:True
num worker:2, num agv&box:2, env_length:1043, max_env_len:5000, task_finished:True
num worker:2, num agv&box:2, env_length:1090, max_env_len:5000, task_finished:True
num worker:2, num agv&box:2, env_length:1092, max_env_len:5000, task_finished:True
num worker:2, num agv&box:2, env_length:1095, max_env_len:5000, task_finished:True
num worker:2, num agv&box:2, env_length:1094, max_env_len:5000, task_finished:True
num worker:2, num agv&box:2, env_length:1122, max_env_len:5000, task_finished:True
num worker:2, num agv&box:2, env_length:1144, max_env_len:5000, task_finished:True
num worker:2, num agv&box:2, env_length:1138, max_env_len:5000, task_finished:True
num worker:2, num agv&box:2, env_length:1082, max_env_len:5000, task_finished:True
num worker:2, num agv&box:2, env_length:1068, max_env_len:5000, task_finished:True
num worker:2, num agv&box:2, env_length:1196, max_env_len:5000, task_finished:True


num worker:2, num agv&box:3, env_length:1114, max_env_len:5000, task_finished:True
num worker:2, num agv&box:3, env_length:1116, max_env_len:5000, task_finished:True
num worker:2, num agv&box:3, env_length:992, max_env_len:5000, task_finished:True
num worker:2, num agv&box:3, env_length:1209, max_env_len:5000, task_finished:True
num worker:2, num agv&box:3, env_length:1080, max_env_len:5000, task_finished:True
num worker:2, num agv&box:3, env_length:1086, max_env_len:5000, task_finished:True
num worker:2, num agv&box:3, env_length:1169, max_env_len:5000, task_finished:True
num worker:2, num agv&box:3, env_length:964, max_env_len:5000, task_finished:True
num worker:2, num agv&box:3, env_length:1107, max_env_len:5000, task_finished:True
num worker:2, num agv&box:3, env_length:1016, max_env_len:5000, task_finished:True
num worker:2, num agv&box:3, env_length:994, max_env_len:5000, task_finished:True
num worker:2, num agv&box:3, env_length:1106, max_env_len:5000, task_finished:True
num worker:2, num agv&box:3, env_length:1075, max_env_len:5000, task_finished:True
num worker:2, num agv&box:3, env_length:1118, max_env_len:5000, task_finished:True
num worker:2, num agv&box:3, env_length:986, max_env_len:5000, task_finished:True


num worker:3, num agv&box:1, env_length:957, max_env_len:5000, task_finished:True
num worker:3, num agv&box:1, env_length:1005, max_env_len:5000, task_finished:True
num worker:3, num agv&box:1, env_length:982, max_env_len:5000, task_finished:True
num worker:3, num agv&box:1, env_length:865, max_env_len:5000, task_finished:True
num worker:3, num agv&box:1, env_length:805, max_env_len:5000, task_finished:True
num worker:3, num agv&box:1, env_length:1026, max_env_len:5000, task_finished:True
num worker:3, num agv&box:1, env_length:1051, max_env_len:5000, task_finished:True
num worker:3, num agv&box:1, env_length:850, max_env_len:5000, task_finished:True
num worker:3, num agv&box:1, env_length:776, max_env_len:5000, task_finished:True
num worker:3, num agv&box:1, env_length:927, max_env_len:5000, task_finished:True
num worker:3, num agv&box:1, env_length:1033, max_env_len:5000, task_finished:True
num worker:3, num agv&box:1, env_length:912, max_env_len:5000, task_finished:True
num worker:3, num agv&box:1, env_length:959, max_env_len:5000, task_finished:True
num worker:3, num agv&box:1, env_length:867, max_env_len:5000, task_finished:True
num worker:3, num agv&box:1, env_length:1055, max_env_len:5000, task_finished:True
num worker:3, num agv&box:1, env_length:958, max_env_len:5000, task_finished:True

num worker:3, num agv&box:2, env_length:900, max_env_len:5000, task_finished:True
num worker:3, num agv&box:2, env_length:879, max_env_len:5000, task_finished:True
num worker:3, num agv&box:2, env_length:785, max_env_len:5000, task_finished:True
num worker:3, num agv&box:2, env_length:877, max_env_len:5000, task_finished:True
num worker:3, num agv&box:2, env_length:788, max_env_len:5000, task_finished:True
num worker:3, num agv&box:2, env_length:827, max_env_len:5000, task_finished:True
num worker:3, num agv&box:2, env_length:937, max_env_len:5000, task_finished:True
num worker:3, num agv&box:2, env_length:951, max_env_len:5000, task_finished:True
num worker:3, num agv&box:2, env_length:991, max_env_len:5000, task_finished:True
num worker:3, num agv&box:2, env_length:829, max_env_len:5000, task_finished:True
num worker:3, num agv&box:2, env_length:820, max_env_len:5000, task_finished:True
num worker:3, num agv&box:2, env_length:834, max_env_len:5000, task_finished:True
num worker:3, num agv&box:2, env_length:959, max_env_len:5000, task_finished:True
num worker:3, num agv&box:2, env_length:809, max_env_len:5000, task_finished:True
num worker:3, num agv&box:2, env_length:971, max_env_len:5000, task_finished:True
num worker:3, num agv&box:2, env_length:897, max_env_len:5000, task_finished:True
num worker:3, num agv&box:2, env_length:800, max_env_len:5000, task_finished:True
num worker:3, num agv&box:2, env_length:954, max_env_len:5000, task_finished:True
num worker:3, num agv&box:2, env_length:803, max_env_len:5000, task_finished:True


num worker:3, num agv&box:3, env_length:825, max_env_len:5000, task_finished:True
num worker:3, num agv&box:3, env_length:883, max_env_len:5000, task_finished:True
num worker:3, num agv&box:3, env_length:954, max_env_len:5000, task_finished:True
num worker:3, num agv&box:3, env_length:816, max_env_len:5000, task_finished:True
num worker:3, num agv&box:3, env_length:808, max_env_len:5000, task_finished:True
num worker:3, num agv&box:3, env_length:973, max_env_len:5000, task_finished:True
num worker:3, num agv&box:3, env_length:823, max_env_len:5000, task_finished:True
num worker:3, num agv&box:3, env_length:813, max_env_len:5000, task_finished:True
num worker:3, num agv&box:3, env_length:792, max_env_len:5000, task_finished:True
num worker:3, num agv&box:3, env_length:966, max_env_len:5000, task_finished:True
num worker:3, num agv&box:3, env_length:825, max_env_len:5000, task_finished:True
num worker:3, num agv&box:3, env_length:983, max_env_len:5000, task_finished:True
num worker:3, num agv&box:3, env_length:810, max_env_len:5000, task_finished:True
num worker:3, num agv&box:3, env_length:952, max_env_len:5000, task_finished:True
num worker:3, num agv&box:3, env_length:822, max_env_len:5000, task_finished:True
num worker:3, num agv&box:3, env_length:986, max_env_len:5000, task_finished:True
num worker:3, num agv&box:3, env_length:985, max_env_len:5000, task_finished:True