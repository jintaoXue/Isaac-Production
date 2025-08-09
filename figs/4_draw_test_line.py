import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


'''=========================================================Main drawing code=========================================================='''
def create_figure(metric_name_file_dir_list, data_algo_name_dict, groups, title_dict):
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(15, 6))
    gs = GridSpec(1, 2, width_ratios=[2.2, 1])  # 图一更宽，图二窄
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    # 只处理前两个metric
    for idx, (metric_name, file_path) in enumerate(list(metric_name_file_dir_list.items())[:2]):
        # 读取数据
        df = pd.read_csv(file_path)
        # 整理数据为长表格格式
        data = []
        data_names = df.loc[0]
        data_dict = {}
        for i, data_name in enumerate(df.columns):
            if "step" in data_name or "MIN" in data_name or "MAX" in data_name:
                continue
            else:
                data_dict[data_name.split(' ')[0]] = df.iloc[1:, i]
        
        # 根据data_dict和data_algo_name_dict匹配数据
        for algo_key, algo_name in data_algo_name_dict.items():
            if algo_key in data_dict:
                for value in data_dict[algo_key].dropna():
                    data.append({'Algorithm': algo_name, metric_name: value})
        
        # 定义分组颜色
        group_colors = {
            'A': '#1f77b4',  # 蓝色
            'B': '#2ca02c',  # 绿色
            'C': '#9467bd',  # 紫色
            'D': '#e377c2'   # 粉色
        }
        # 算法到分组的映射
        algo_group_map = {}
        for group_name, group_dict in groups:
            for algo_key in group_dict:
                algo_group_map[algo_key] = group_name

        plot_df = pd.DataFrame(data)
        # 保证算法顺序
        algo_order = [algo_name for _, algo_name in data_algo_name_dict.items()]
        # 算法到颜色的映射
        algo_color_map = {}
        for algo_key, algo_name in data_algo_name_dict.items():
            group = algo_group_map.get(algo_key, None)
            if group:
                algo_color_map[algo_name] = group_colors[group]
            else:
                algo_color_map[algo_name] = '#333333'

        # 绘制箱线图或条形图
        if idx == 1:  # 图二 Overwork 用条形图
            bar_vals = []
            for algo in algo_order:
                vals = plot_df[plot_df['Algorithm'] == algo][metric_name]
                # 统计非零次数的和/总长度
                overwork_count = (vals != 0).sum()
                total_count = len(vals)
                mean_overwork = overwork_count / total_count if total_count > 0 else 0
                bar_vals.append(mean_overwork)
            axes[idx].bar(algo_order, bar_vals, color=[algo_color_map[a] for a in algo_order])
            # 在柱子上显示数值
            for i, v in enumerate(bar_vals):
                axes[idx].text(i, v, f'{v:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold', color='black')
        else:
            # 箱线图，包含异常值
            box = sns.boxplot(x='Algorithm', y=metric_name, data=plot_df, ax=axes[idx], order=algo_order,
                        palette=algo_color_map, showmeans=False, meanprops={"marker":"o","markerfacecolor":"white","markeredgecolor":"black"}, showfliers=True)
            # 在箱线图上显示均值，颜色与箱体一致
            for i, algo in enumerate(algo_order):
                vals = plot_df[plot_df['Algorithm'] == algo][metric_name]
                if len(vals) > 0:
                    mean_val = vals.mean()
                    if idx == 2:
                        axes[idx].text(i, mean_val, f'{mean_val:.8f}', ha='center', va='bottom', fontsize=10, fontweight='bold', color='black')
                    elif idx == 0:
                        axes[idx].text(i, mean_val + (vals.max() - vals.min()) * 0.04, f'{mean_val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold', color='black')
                        axes[idx].scatter(i, mean_val, marker='>', color='red', s=120, zorder=5)
                        axes[idx].plot([i-0.13, i+0.13], [mean_val, mean_val], color='red', linewidth=2, zorder=6)
                    else:
                        axes[idx].text(i, mean_val, f'{mean_val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold', color='black')
        axes[idx].set_title(metric_name)
        axes[idx].set_xlabel('')
        axes[idx].set_ylabel(metric_name)
        axes[idx].tick_params(axis='x', rotation=30)
        if idx == 0:  # 第一张图 Makespan
            axes[idx].set_ylim(top=2000)
    plt.tight_layout()
    return fig

def create_human_robot_curves(metric_name_file_dir_list, data_algo_name_dict, groups):
    """
    绘制human/robot数量变化的曲线图
    布局：
    图1: human vs makespan (robot=1)
    图2: human vs makespan (robot=2) 
    图3: human vs overwork (robot=1)
    图4: human vs overwork (robot=2)
    图5: robot vs makespan (human=1)
    图6: robot vs makespan (human=2)
    图7: robot vs overwork (human=1)
    图8: robot vs overwork (human=2)
    """
    from matplotlib.gridspec import GridSpec
    
    # 创建2行2列的子图
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1, 1], height_ratios=[1, 1])
    
    # 定义human和robot的数量
    human_nums = [1, 2, 3]
    robot_nums = [1, 2, 3]
    
    # 定义分组颜色
    group_colors = {
        'A': '#1f77b4',  # 蓝色
        'B': '#2ca02c',  # 绿色
        'C': '#9467bd',  # 紫色
        'D': '#e377c2'   # 粉色
    }
    
    # 算法到分组的映射
    algo_group_map = {}
    for group_name, group_dict in groups:
        for algo_key in group_dict:
            algo_group_map[algo_key] = group_name
    
    # 算法到颜色的映射
    algo_color_map = {}
    for algo_key, algo_name in data_algo_name_dict.items():
        group = algo_group_map.get(algo_key, None)
        if group:
            algo_color_map[algo_name] = group_colors[group]
        else:
            algo_color_map[algo_name] = '#333333'
    
    # 为PF-CD3QP设置特殊颜色
    algo_color_map['PF-CD3QP'] = '#ff7f0e'  # 橙色
    
    # 为每个metric计算统计数据
    makespan_stats = {}
    overwork_stats = {}
    
    # 处理makespan数据
    makespan_file = metric_name_file_dir_list["Makespan (Test)"]
    df_makespan = pd.read_csv(makespan_file)
    
    # 处理overwork数据  
    overwork_file = metric_name_file_dir_list["Overwork (Test)"]
    df_overwork = pd.read_csv(overwork_file)
    
    # 整理数据
    def process_data(df):
        data_dict = {}
        for i, data_name in enumerate(df.columns):
            if "step" in data_name or "MIN" in data_name or "MAX" in data_name:
                continue
            else:
                data_dict[data_name.split(' ')[0]] = df.iloc[0:, i]
        return data_dict
    
    makespan_data = process_data(df_makespan)
    overwork_data = process_data(df_overwork)
    
    # 为每个算法计算不同human/robot组合的统计值
    for algo_key, algo_name in data_algo_name_dict.items():
        if algo_key not in makespan_data or algo_key not in overwork_data:
            continue
            
        makespan_algo_data = makespan_data[algo_key].dropna()
        overwork_algo_data = overwork_data[algo_key].dropna()
        
        if len(makespan_algo_data) != 450 or len(overwork_algo_data) != 450:
            print(f"警告: {algo_name} 数据条数不是450: makespan={len(makespan_algo_data)}, overwork={len(overwork_algo_data)}")
            continue
        
        # 计算makespan统计值
        makespan_stats[algo_name] = {}
        overwork_stats[algo_name] = {}
        
        for h_idx, human_num in enumerate(human_nums):
            for r_idx, robot_num in enumerate(robot_nums):
                # 计算数据索引
                start_idx = h_idx * 150 + r_idx * 50
                end_idx = start_idx + 50
                
                if end_idx <= len(makespan_algo_data):
                    makespan_values = makespan_algo_data.iloc[start_idx:end_idx]
                    overwork_values = overwork_algo_data.iloc[start_idx:end_idx]
                    
                    # makespan计算：平均值
                    makespan_stats[algo_name][(human_num, robot_num)] = makespan_values.mean()
                    
                    # overwork计算：非零次数/总次数
                    overwork_rate = (overwork_values != 0).sum() / len(overwork_values)
                    overwork_stats[algo_name][(human_num, robot_num)] = overwork_rate
    
    # 绘制4张图：2行2列布局
    # 图1: human vs makespan (所有robot情况取平均)
    ax1 = fig.add_subplot(gs[0, 0])
    for algo_name, stats in makespan_stats.items():
        color = algo_color_map.get(algo_name, '#333333')
        linestyle = '--' if 'PF' in algo_name else '-'
        
        x_values = []
        y_values = []
        for human_num in human_nums:
            # 计算该human_num下所有robot情况的平均值
            robot_vals = []
            for robot_num in robot_nums:
                if (human_num, robot_num) in stats:
                    robot_vals.append(stats[(human_num, robot_num)])
            if robot_vals:  # 如果有数据
                x_values.append(human_num)
                y_values.append(np.mean(robot_vals))
        
        if x_values and y_values:
            ax1.plot(x_values, y_values, marker='o', label=algo_name, color=color, linewidth=2, markersize=6, linestyle=linestyle)
    
    ax1.set_xlabel('Human num')
    ax1.set_ylabel('Makespan (Test)')
    ax1.set_title('Makespan vs Human Number')
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    # 图2: human vs overwork (所有robot情况取平均)
    ax2 = fig.add_subplot(gs[0, 1])
    for algo_name, stats in overwork_stats.items():
        color = algo_color_map.get(algo_name, '#333333')
        linestyle = '--' if 'PF' in algo_name else '-'
        
        x_values = []
        y_values = []
        for human_num in human_nums:
            # 计算该human_num下所有robot情况的平均值
            robot_vals = []
            for robot_num in robot_nums:
                if (human_num, robot_num) in stats:
                    robot_vals.append(stats[(human_num, robot_num)])
            if robot_vals:  # 如果有数据
                x_values.append(human_num)
                y_values.append(np.mean(robot_vals))
        
        if x_values and y_values:
            ax2.plot(x_values, y_values, marker='o', label=algo_name, color=color, linewidth=2, markersize=6, linestyle=linestyle)
    
    ax2.set_xlabel('Human num')
    ax2.set_ylabel('Overwork (Test)')
    ax2.set_title('Overwork vs Human Number')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12, handlelength=4, handleheight=2)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    # 图3: robot vs makespan (所有human情况取平均)
    ax3 = fig.add_subplot(gs[1, 0])
    for algo_name, stats in makespan_stats.items():
        color = algo_color_map.get(algo_name, '#333333')
        linestyle = '--' if 'PF' in algo_name else '-'
        
        x_values = []
        y_values = []
        for robot_num in robot_nums:
            # 计算该robot_num下所有human情况的平均值
            human_vals = []
            for human_num in human_nums:
                if (human_num, robot_num) in stats:
                    human_vals.append(stats[(human_num, robot_num)])
            if human_vals:  # 如果有数据
                x_values.append(robot_num)
                y_values.append(np.mean(human_vals))
        
        if x_values and y_values:
            ax3.plot(x_values, y_values, marker='o', label=algo_name, color=color, linewidth=2, markersize=6, linestyle=linestyle)
    
    ax3.set_xlabel('Robot num')
    ax3.set_ylabel('Makespan (Test)')
    ax3.set_title('Makespan vs Robot Number')
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    # 图4: robot vs overwork (所有human情况取平均)
    ax4 = fig.add_subplot(gs[1, 1])
    for algo_name, stats in overwork_stats.items():
        color = algo_color_map.get(algo_name, '#333333')
        linestyle = '--' if 'PF' in algo_name else '-'
        
        x_values = []
        y_values = []
        for robot_num in robot_nums:
            # 计算该robot_num下所有human情况的平均值
            human_vals = []
            for human_num in human_nums:
                if (human_num, robot_num) in stats:
                    human_vals.append(stats[(human_num, robot_num)])
            if human_vals:  # 如果有数据
                x_values.append(robot_num)
                y_values.append(np.mean(human_vals))
        
        if x_values and y_values:
            ax4.plot(x_values, y_values, marker='o', label=algo_name, color=color, linewidth=2, markersize=6, linestyle=linestyle)
    
    ax4.set_xlabel('Robot num')
    ax4.set_ylabel('Overwork (Test)')
    ax4.set_title('Overwork vs Robot Number')
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12, handlelength=4, handleheight=2)
    ax4.grid(True, alpha=0.3)
    ax4.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    plt.tight_layout()
    return fig

if __name__ == '__main__':
    ## 3 metric for 3 subfigure, each subfigure has 9 algorithms, draw the boxplot
    ## data source
    metric_name_file_dir_list = {
        "Makespan (Test)": os.path.dirname(__file__) + "/test" + "/EpEnvLen.csv",
        "Overwork (Test)": os.path.dirname(__file__) + "/test" + "/EpOverCost.csv",
        "Progress (Test)": os.path.dirname(__file__) + "/test" + "/EpProgress.csv"
    }
    title_dict = {
        "Makespan (Test)": "Makespan (Test)",
        "Overwork (Test)": "Overwork (Test)",
        "Progress (Test)": "Progress (Test)"
    }
    data_algo_name_dict = {
        # 1_test_rl_filter_test_49600_2025-07-25_15-02-16  D3QN
        "2_test_rl_filter_49600_2025-07-29_22-22-18": "D3QN",
        "3_test_rl_filter_49600_2025-07-20_12-17-12": "PF-CD3Q",
        # "4_test_rl_filter_49600_2025-07-27_14-41-12": "PF-CD3QP",
        "5_test_dqn_49600_2025-07-27_11-39-32": "DQN",
        "6_test_dqn_test_49600_2025-07-29_13-21-06": "PF-DQN",
        "7_test_ppo_dis_49600_2025-07-31_13-37-58": "PPO",
        "8_test_ppo_dis_49600_2025-07-30_13-18-07": "PF-PPO",
        "9_test_ppolag_filter_dis_49600_2025-07-23_22-24-04": "PPO-Lag",
        "10_test_ppolag_filter_dis_42800_2025-07-21_23-34-32": "PF-PPO-Lag"
    }
    # 定义算法分组
    group_A = {
        "2_test_rl_filter_49600_2025-07-29_22-22-18": "D3QN",
        "3_test_rl_filter_49600_2025-07-20_12-17-12": "PF-CD3Q",
        # "4_test_rl_filter_49600_2025-07-27_14-41-12": "PF-CD3QP",
    }
    group_B = {
        "5_test_dqn_49600_2025-07-27_11-39-32": "DQN",
        "6_test_dqn_test_49600_2025-07-29_13-21-06": "PF-DQN",
    }
    group_C = {
        "7_test_ppo_dis_49600_2025-07-31_13-37-58": "PPO",
        "8_test_ppo_dis_49600_2025-07-30_13-18-07": "PF-PPO",
    }
    group_D = {
        "9_test_ppolag_filter_dis_49600_2025-07-23_22-24-04": "PPO-Lag",
        "10_test_ppolag_filter_dis_42800_2025-07-21_23-34-32": "PF-PPO-Lag"
    }
    
    # 合并所有算法字典
    data_algo_name_dict = {**group_A, **group_B, **group_C, **group_D}
    
    # 定义groups
    groups = [('A', group_A), ('B', group_B), ('C', group_C), ('D', group_D)]
    
    # # 创建原始箱线图
    # fig = create_figure(metric_name_file_dir_list, data_algo_name_dict, groups, title_dict)
    
    # # 保存箱线图
    # output_path = os.path.dirname(__file__) + "/test_boxplot.pdf"
    # plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')
    # print(f"箱线图已保存到: {output_path}")
    
    # # 显示箱线图
    # plt.show()
    
    # 创建human/robot曲线图
    fig2 = create_human_robot_curves(metric_name_file_dir_list, data_algo_name_dict, groups)
    
    # 保存曲线图
    output_path2 = os.path.dirname(__file__) + "/test_human_robot_curves.pdf"
    plt.savefig(output_path2, dpi=300, bbox_inches='tight', format='pdf')
    print(f"曲线图已保存到: {output_path2}")
    
    # 显示曲线图
    plt.show()
    
    
    

