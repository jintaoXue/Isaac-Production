Isaac Production

---

# Isaac Production
Isaac Production is a training platform for human-robot task allocation in manufacting. Built on [NVIDIA Isaac Lab](https://isaac-sim.github.io/IsaacLab/main/index.html) and [NVIDIA Isaac Sim](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)


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

self.max_episode_length 是否可以修改?

render的代码学要修改

wandb 

修复自动补全

self.scheduler lr_scheduler schedule_type (请参考~/Repos/miniconda3/envs/isaac-lab/lib/python3.10/site-packages/rl_games/common/a2c_common.py)
解决办法，在gym register 部分order_enforce=False

test setting的num robot character 不一定对


