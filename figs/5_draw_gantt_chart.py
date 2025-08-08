import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


'''=========================================================Main drawing code=========================================================='''
def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)
    # return plt.colormaps.get_cmap(name, n)
                
#ax.arrow(s_time, sum(y_pos_dic[y_label]), 0, -0.5, color=cmap(color), width=1.0, shape='full', head_starts_at_zero=False)
cmap = get_cmap(10)

if __name__ == '__main__':
    import pickle
    with open('figs/gantt/gantt_data.pkl', 'rb') as f:
        dic = pickle.load(f)
    print(dic)
    # fatigue_name_dic = {'none':"none", 'hoop_preparing': "convey flange to storage at side", 'bending_tube_preparing': "Convey bend duct to storage at side", 
    # 'hoop_loading_inner':"load flange to workstation 1", 'bending_tube_loading_inner':"load bend duct to workstation 1", 
    # 'hoop_loading_outer':"load flange to workstation 2", 'bending_tube_loading_outer':"load bend duct to workstation 2", 
    # "cutting_cube":"select and activate controlling code for workstations", 'collect_product': "collect made products", 
    # 'placing_product':"place made products to storage region"}
    fatigue_name_projection_dic = {'none':"task 0", 'hoop_preparing': "task 1", 'bending_tube_preparing': "task 2", 
        'hoop_loading_inner':"task 3", 'bending_tube_loading_inner':"task 4", 
        'hoop_loading_outer':"task 5", 'bending_tube_loading_outer':"task 6", 
        "cutting_cube":"task 7", 'collect_product': "task 8", 
        'placing_product':"task 9"}

    # subfic 1, is the fatigue value change of the worker
    worker_data = dic['worker']
    worker_tasks_dic = dic['worker_tasks_dic']
    num_worker = len(worker_data)
    ftg_thresh_phy = 0.95 # 0.95 is the threshold of the physical fatigue
    worker_task_range = {'none', 'hoop_preparing', 'bending_tube_preparing', 'hoop_loading_inner', 'bending_tube_loading_inner', 'hoop_loading_outer', 'bending_tube_loading_outer', "cutting_cube", 
                           'placing_product'}
    # worker_data_example_explanation = ('state': str, 'subtask': str, 'task': str, "phy_fatigue": float, "psy_fatigue": float, "time_step": int)
    # fig one: plot the fatigue value change of the worker with time step, if have subtask change (prev subtask is different from current subtask), plot the vertical line

    agv_data = dic['agv']
    agv_tasks_dic = dic['agv_tasks_dic']
    agv_task_range = {'none', 'hoop_preparing', 'bending_tube_preparing', 'collect_product','placing_product'}
    num_agv = len(agv_data)
    # agv_data_example_explanation = ('state': str, 'subtask': str, 'task': str, "time_step": int) # state: str, subtask: str, task: str, time_step: int
    # fig two: plot the task-level gantt chart of human and agv with time step
    # requirement: each task is a rectangle, each task is different color, if the task is 'none', not plot

    # 创建包含两个子图的图表
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    
    # 子图1：工人疲劳值变化
    if num_worker > 0:
        worker_0_data = worker_data[0]  # 假设只有一个工人，取第一个工人的数据
        
        # 提取时间和疲劳值 - 数据结构是元组 (state, task, subtask, "phy_fatigue": float, "psy_fatigue": float, time_step)
        time_steps = [entry[-1] for entry in worker_0_data]  # time_step是第6个元素
        tasks = [entry[1] for entry in worker_0_data]    # task是第2个元素
        subtasks = [entry[2] for entry in worker_0_data]    # subtask是第3个元素
        phy_fatigue = [entry[3] for entry in worker_0_data]
        
        # 为不同任务分配颜色
        task_colors_fatigue = {}
        all_tasks_fatigue = list(set(tasks))
        for i, task in enumerate(all_tasks_fatigue):
            if task == 'none':
                task_colors_fatigue[task] = 'black'  # task 0单独用黑色
            else:
                task_colors_fatigue[task] = cmap(i)
        
        # 绘制物理疲劳值变化，按任务分段绘制不同颜色
        current_task = tasks[0]
        start_idx = 0
        
        for i in range(1, len(tasks)):
            if tasks[i] != current_task:
                # 绘制前一个任务段
                ax1.plot(time_steps[start_idx:i], phy_fatigue[start_idx:i], 
                        color=task_colors_fatigue.get(current_task, 'gray'), 
                        linewidth=2)
                
                # 绘制连接线段（从当前任务段结束到下一个任务段开始）
                if i < len(tasks):
                    ax1.plot([time_steps[i-1], time_steps[i]], [phy_fatigue[i-1], phy_fatigue[i]], 
                            color=task_colors_fatigue.get(current_task, 'gray'), 
                            linewidth=2)
                
                # 绘制任务变化垂直线
                ax1.axvline(x=time_steps[i], color='gray', linestyle='--', alpha=0.7)
                
                current_task = tasks[i]
                start_idx = i
        
        # 绘制最后一个任务段
        ax1.plot(time_steps[start_idx:], phy_fatigue[start_idx:], 
                color=task_colors_fatigue.get(current_task, 'gray'), 
                linewidth=2)
        
        # 添加疲劳阈值水平线
        ax1.axhline(y=ftg_thresh_phy, color='red', linestyle='--', alpha=0.8, label=f'Fatigue Threshold ({ftg_thresh_phy})')
        
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Fatigue Value')
        ax1.set_title('Worker Fatigue Changes Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # 子图2：任务级别甘特图
    # 使用与图一相同的颜色映射
    task_colors = task_colors_fatigue.copy()
    
    # 用于跟踪已添加的图例标签
    legend_labels = set()
    
    # 计算甘特图的总行数（工人 + AGV数量）
    total_agents = num_worker + num_agv
    y_positions = list(range(total_agents))
    
    # 绘制工人任务
    current_y_pos = 0
    if num_worker > 0:
        for worker_idx in range(num_worker):
            worker_data_current = worker_data[worker_idx]
            current_task = None
            task_start = 0
            
            for i, entry in enumerate(worker_data_current):
                task = entry[1]  # task是第2个元素，不考虑subtask
                time_step = entry[-1]  # time_step是最后一个元素
                
                if task != current_task:
                    # 绘制前一个任务
                    if current_task and current_task not in ['none', 'free']:
                        ax2.barh(current_y_pos, time_step - task_start, left=task_start, 
                                color=task_colors.get(current_task, 'gray'), 
                                alpha=0.8, label=fatigue_name_projection_dic.get(current_task, current_task) if current_task not in legend_labels else "")
                        legend_labels.add(current_task)
                    
                    current_task = task
                    task_start = time_step
            
            # 绘制最后一个任务
            if current_task and current_task not in ['none', 'free']:
                ax2.barh(current_y_pos, worker_data_current[-1][-1] - task_start, left=task_start, 
                        color=task_colors.get(current_task, 'gray'), 
                        alpha=0.8, label=fatigue_name_projection_dic.get(current_task, current_task) if current_task not in legend_labels else "")
                legend_labels.add(current_task)
            
            current_y_pos += 1
    
    # 绘制AGV任务
    if num_agv > 0:
        for agv_idx in range(num_agv):
            agv_data_current = agv_data[agv_idx]
            current_task = None
            task_start = 0
            
            for i, entry in enumerate(agv_data_current):
                task = entry[2]  # task是第2个元素，不考虑subtask
                time_step = entry[-1]  # time_step是最后一个元素
                
                if task != current_task:
                    # 绘制前一个任务
                    if current_task and current_task not in ['none', 'free']:
                        ax2.barh(current_y_pos, time_step - task_start, left=task_start, 
                                color=task_colors.get(current_task, 'gray'), 
                                alpha=0.8, label=fatigue_name_projection_dic.get(current_task, current_task) if current_task not in legend_labels else "")
                        legend_labels.add(current_task)
                    
                    current_task = task
                    task_start = time_step
            
            # 绘制最后一个任务
            if current_task and current_task not in ['none', 'free']:
                ax2.barh(current_y_pos, agv_data_current[-1][-1] - task_start, left=task_start, 
                        color=task_colors.get(current_task, 'gray'), 
                        alpha=0.8, label=fatigue_name_projection_dic.get(current_task, current_task) if current_task not in legend_labels else "")
                legend_labels.add(current_task)
            
            current_y_pos += 1
    
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Agent')
    ax2.set_title('Task-Level Gantt Chart')
    
    # 设置Y轴标签
    y_labels = []
    for i in range(num_worker):
        y_labels.append(f'Worker {i+1}')
    for i in range(num_agv):
        y_labels.append(f'AGV {i+1}')
    
    ax2.set_yticks(range(total_agents))
    ax2.set_yticklabels(y_labels)
    ax2.grid(True, alpha=0.3)
    
    # 添加图例（去除重复项）
    handles, labels = ax2.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax2.legend(by_label.values(), by_label.keys(), loc='upper right')
    
    plt.tight_layout()
    plt.savefig('figs/gantt_chart.png', dpi=300, bbox_inches='tight')
    plt.show()

