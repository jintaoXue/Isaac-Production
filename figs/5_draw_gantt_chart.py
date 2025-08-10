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

def plot_fatigue_curve(ax, worker_data, algorithm_name, color, ftg_thresh_phy, linestyle='-'):
    """绘制单个算法的疲劳曲线"""
    if len(worker_data) > 0:
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
                ax.plot(time_steps[start_idx:i], phy_fatigue[start_idx:i], 
                        color=task_colors_fatigue.get(current_task, 'gray'), 
                        linewidth=2, alpha=0.8, linestyle=linestyle)
                
                # 绘制连接线段（从当前任务段结束到下一个任务段开始）
                if i < len(tasks):
                    ax.plot([time_steps[i-1], time_steps[i]], [phy_fatigue[i-1], phy_fatigue[i]], 
                            color=task_colors_fatigue.get(current_task, 'gray'), 
                            linewidth=2, alpha=0.8, linestyle=linestyle)
                
                current_task = tasks[i]
                start_idx = i
        
        # 绘制最后一个任务段
        ax.plot(time_steps[start_idx:], phy_fatigue[start_idx:], 
                color=task_colors_fatigue.get(current_task, 'gray'), 
                linewidth=2, alpha=0.8, linestyle=linestyle, label=algorithm_name)
        
        return task_colors_fatigue

def plot_gantt_chart(ax, worker_data, agv_data, task_colors, algorithm_name, show_legend=False):
    """绘制单个算法的甘特图"""
    num_worker = len(worker_data)
    num_agv = len(agv_data)
    
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
                        ax.barh(current_y_pos, time_step - task_start, left=task_start, 
                                color=task_colors.get(current_task, 'gray'), 
                                alpha=0.8, height=0.5, label=fatigue_name_projection_dic.get(current_task, current_task) if current_task not in legend_labels else "")
                        legend_labels.add(current_task)
                    
                    current_task = task
                    task_start = time_step
            
            # 绘制最后一个任务
            if current_task and current_task not in ['none', 'free']:
                ax.barh(current_y_pos, worker_data_current[-1][-1] - task_start, left=task_start, 
                        color=task_colors.get(current_task, 'gray'), 
                        alpha=0.8, height=0.5, label=fatigue_name_projection_dic.get(current_task, current_task) if current_task not in legend_labels else "")
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
                        ax.barh(current_y_pos, time_step - task_start, left=task_start, 
                                color=task_colors.get(current_task, 'gray'), 
                                alpha=0.8, height=0.5, label=fatigue_name_projection_dic.get(current_task, current_task) if current_task not in legend_labels else "")
                        legend_labels.add(current_task)
                    
                    current_task = task
                    task_start = time_step
            
            # 绘制最后一个任务
            if current_task and current_task not in ['none', 'free']:
                ax.barh(current_y_pos, agv_data_current[-1][-1] - task_start, left=task_start, 
                        color=task_colors.get(current_task, 'gray'), 
                        alpha=0.8, height=0.5, label=fatigue_name_projection_dic.get(current_task, current_task) if current_task not in legend_labels else "")
                legend_labels.add(current_task)
            
            current_y_pos += 1
    
    ax.set_xlabel('Time Step', fontsize=12)
    # ax.set_ylabel('Robot', fontsize=12)
    ax.set_title(f'Task-Level Gantt Chart - {algorithm_name}', fontsize=14)
    
    # 设置Y轴标签
    y_labels = []
    for i in range(num_worker):
        y_labels.append(f'Worker {i+1}')
    for i in range(num_agv):
        y_labels.append(f'Robot {i+1}')
    
    ax.set_yticks(range(total_agents))
    ax.set_yticklabels(y_labels)
    # 反转Y轴，让Worker显示在第一行
    ax.invert_yaxis()
    # 设置Y轴范围，减小不同类型之间的间隔
    ax.set_ylim(-0.3, total_agents - 0.3)
    ax.tick_params(axis='both', which='both', labelsize=12)
    ax.grid(True, alpha=0.3)
    
    # 只在需要时添加图例
    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right')

if __name__ == '__main__':
    import pickle
    import matplotlib as mpl
    
    # 设置matplotlib不省略刻度值
    mpl.rcParams['axes.formatter.useoffset'] = False
    mpl.rcParams['axes.formatter.use_mathtext'] = False
    
    algorithm_name_dict = {'D3QN': 'figs/gantt/gantt_data_D3QN.pkl', 'PF-CD3Q': 'figs/gantt/gantt_data_PF-CD3Q.pkl'}
    
    # fatigue_name_dic = {'none':"none", 'hoop_preparing': "convey flange to storage at side", 'bending_tube_preparing': "Convey bend duct to storage at side", 
    # 'hoop_loading_inner':"load flange to workstation 1", 'bending_tube_loading_inner':"load bend duct to workstation 1", 
    # 'hoop_loading_outer':"load flange to workstation 2", 'bending_tube_loading_outer':"load bend duct to workstation 2", 
    # "cutting_cube":"select and activate controlling code for workstations", 'collect_product': "collect made products", 
    # 'placing_product':"place made products to storage region"}
    fatigue_name_projection_dic = {'none':"task 0 (recover)", 'hoop_preparing': "task 1", 'bending_tube_preparing': "task 2", 
        'hoop_loading_inner':"task 3", 'bending_tube_loading_inner':"task 4", 
        'hoop_loading_outer':"task 5", 'bending_tube_loading_outer':"task 6", 
        "cutting_cube":"task 7", 'collect_product': "task 8", 
        'placing_product':"task 9"}

    ftg_thresh_phy = 0.95 # 0.95 is the threshold of the physical fatigue
    
    # 加载两个算法的数据
    algorithm_data = {}
    for alg_name, file_path in algorithm_name_dict.items():
        try:
            with open(file_path, 'rb') as f:
                algorithm_data[alg_name] = pickle.load(f)
        except FileNotFoundError:
            print(f"Warning: {file_path} not found, skipping {alg_name}")
            continue

    if not algorithm_data:
        print("No algorithm data found!")
        exit()

    # 创建包含三个子图的图表花
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 14), height_ratios=[1.5, 1, 1])
    
    # 子图1：两个算法的疲劳值对比
    colors = ['blue', 'red']  # 为两个算法分配不同颜色
    linestyles = ['-', '--']  # 为两个算法分配不同线型
    all_task_colors = {}
    
    for i, (alg_name, data) in enumerate(algorithm_data.items()):
        worker_data = data['worker']
        task_colors = plot_fatigue_curve(ax1, worker_data, alg_name, colors[i], ftg_thresh_phy, linestyles[i])
        all_task_colors[alg_name] = task_colors
    
    # 添加疲劳阈值水平线
    ax1.axhline(y=ftg_thresh_phy, color='red', linestyle='--', alpha=0.8, label=f'Fatigue threshold ({ftg_thresh_phy})')
    
    # 为图一添加所有任务的legend
    # 收集所有任务的颜色信息
    all_tasks = set()
    for task_colors in all_task_colors.values():
        all_tasks.update(task_colors.keys())
    
    # 为图一添加任务legend
    legend_handles = []
    legend_labels = []
    
    # 添加算法legend
    for i, alg_name in enumerate(algorithm_data.keys()):
        linestyle = '--' if i == 1 else '-'  # PF-CD3Q使用虚线，D3QN使用实线
        legend_handles.append(plt.Line2D([0, 2], [0, 0], color='gray', linewidth=3, linestyle=linestyle, label=alg_name))
        legend_labels.append(alg_name)
    
    # 添加任务legend，按照task 0到task 9的顺序
    task_order = ['none', 'hoop_preparing', 'bending_tube_preparing', 'hoop_loading_inner', 
                  'bending_tube_loading_inner', 'hoop_loading_outer', 'bending_tube_loading_outer', 
                  'cutting_cube', 'collect_product', 'placing_product']
    
    for task in task_order:
        if task in all_tasks and task in fatigue_name_projection_dic:
            color = list(all_task_colors.values())[0].get(task, 'gray')  # 使用第一个算法的颜色
            legend_handles.append(plt.Rectangle((0, 0), 1, 1, facecolor=color, alpha=0.8, label=fatigue_name_projection_dic[task]))
            legend_labels.append(fatigue_name_projection_dic[task])
    
    # 添加疲劳阈值legend
    legend_handles.append(plt.Line2D([0, 2], [0, 0], color='red', linestyle='--', linewidth=3, label=f'Fatigue Threshold ({ftg_thresh_phy})'))
    legend_labels.append(f'Fatigue Threshold ({ftg_thresh_phy})')
    
    ax1.set_xlabel('Time Step', fontsize=12)
    ax1.set_ylabel('Fatigue Value', fontsize=12)
    ax1.set_title('Worker Fatigue Changes Comparison', fontsize=14)
    ax1.tick_params(axis='both', which='both', labelsize=12)
    ax1.tick_params(axis='x', rotation=0)
    # 设置X轴刻度不省略
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x)}'))
    # 确保X轴刻度不省略
    ax1.xaxis.set_major_locator(plt.MaxNLocator(nbins=10))
    # 强制显示X轴刻度
    ax1.xaxis.set_tick_params(which='major', labelsize=12)
    ax1.legend(legend_handles, legend_labels, loc='center', bbox_to_anchor=(0.7, 0.4), fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # 子图2和3：两个算法的甘特图
    for i, (alg_name, data) in enumerate(algorithm_data.items()):
        worker_data = data['worker']
        agv_data = data['agv']
        
        if i == 0:
            ax = ax2
        else:
            ax = ax3
            
        plot_gantt_chart(ax, worker_data, agv_data, all_task_colors[alg_name], alg_name, show_legend=False)
        # 为图二和图三设置X轴刻度
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x)}'))
        ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=10))
        ax.xaxis.set_tick_params(which='major', labelsize=12)
    
    plt.tight_layout()
    plt.savefig('figs/gantt_chart_comparison.pdf', dpi=300, bbox_inches='tight', format='pdf')
    plt.show()

