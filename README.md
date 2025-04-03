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
    4.cost function的数据收集工作（可以先用仿真器训练一个）
    5. 如果是异质的工人，怎么进行训练