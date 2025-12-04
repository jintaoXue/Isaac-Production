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
                data_dict[data_name.split(' ')[0]] = df.iloc[0:, i]
        
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
                axes[idx].text(i, v, f'{v:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold', color='black')
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
                        axes[idx].plot([i-0.13, i+0.13], [mean_val, mean_val], color='red', linewidth=1, zorder=6)
                    else:
                        axes[idx].text(i, mean_val, f'{mean_val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold', color='black')
        axes[idx].set_title(metric_name)
        axes[idx].set_xlabel('')
        axes[idx].set_ylabel(metric_name)
        axes[idx].tick_params(axis='x', rotation=30)
        
        # 只在第一个子图添加标记解释
        if idx == 0:
            # 在解释文本位置绘制示例图标
            legend_x, legend_y = 0.02, 0.95
            # 绘制示例三角形
            axes[idx].scatter(legend_x + 0.05, legend_y, marker='>', color='red', s=80, 
                             transform=axes[idx].transAxes, zorder=10)
            # 绘制示例直线
            axes[idx].plot([legend_x + 0.02, legend_x + 0.08], [legend_y, legend_y], 
                          color='red', linewidth=1, transform=axes[idx].transAxes, zorder=10)
            # 添加文字说明
            axes[idx].text(legend_x + 0.12, legend_y, ': Mean value', 
                          transform=axes[idx].transAxes, fontsize=10, verticalalignment='center', fontweight='bold',
                          bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            axes[idx].set_ylim(top=2000)
    plt.tight_layout()
    return fig

if __name__ == '__main__':
    ## 3 metric for 3 subfigure, each subfigure has 9 algorithms, draw the boxplot
    ## data source
    metric_name_file_dir_list = {
        "Makespan (Test)": os.path.dirname(__file__) + "/test" + "/EpEnvLen.csv",
        "Overwork (Test)": os.path.dirname(__file__) + "/test" + "/EpOverCost.csv",
        # "Progress (Test)": os.path.dirname(__file__) + "/test" + "/EpProgress.csv"
    }
    title_dict = {
        "Makespan (Test)": "Makespan (Test)",
        "Overwork (Test)": "Overwork (Test)",
        # "Progress (Test)": "Progress (Test)"
    }
    data_algo_name_dict = {
        # 1_test_rl_filter_test_49600_2025-07-25_15-02-16  D3QN
        "2_test_rl_filter_49600_2025-07-29_22-22-18": "D3QN",
        "3_test_rl_filter_49600_2025-07-20_12-17-12": "PF-CD3Q",
        # "4_test_rl_filter_49600_2025-07-27_14-41-12": "PF-CD3QP",
        "5_test_dqn_49600_2025-07-27_11-39-32": "DQN",
        "6_test_dqn_49600_2025-07-29_13-21-06": "PF-DQN",
        "7_test_ppo_dis_49600_2025-07-31_13-37-58": "PPO",
        "8_test_ppo_dis_49600_2025-07-30_13-18-07": "PF-PPO",
        "9_test_ppolag_filter_dis_49600_2025-08-08_13-49-16": "PPO-Lag",
        "10_test_ppolag_filter_dis_49600_2025-08-08_13-46-57": "PF-PPO-Lag"
    }
    # 定义算法分组
    group_A = {
        "2_test_rl_filter_49600_2025-07-29_22-22-18": "D3QN",
        "3_test_rl_filter_49600_2025-07-20_12-17-12": "PF-CD3Q",
        # "4_test_rl_filter_49600_2025-07-27_14-41-12": "PF-CD3QP",
    }
    group_B = {
        "5_test_dqn_49600_2025-07-27_11-39-32": "DQN",
        "6_test_dqn_49600_2025-07-29_13-21-06": "PF-DQN",
    }
    group_C = {
        "7_test_ppo_dis_49600_2025-07-31_13-37-58": "PPO",
        "8_test_ppo_dis_49600_2025-07-30_13-18-07": "PF-PPO",
    }
    group_D = {
        "9_test_ppolag_filter_dis_49600_2025-08-08_13-49-16": "PPO-Lag",
        "10_test_ppolag_filter_dis_49600_2025-08-08_13-46-57": "PF-PPO-Lag"
    }
    
    # 合并所有算法字典
    data_algo_name_dict = {**group_A, **group_B, **group_C, **group_D}
    
    # 定义groups
    groups = [('A', group_A), ('B', group_B), ('C', group_C), ('D', group_D)]
    
    # 创建图表
    fig = create_figure(metric_name_file_dir_list, data_algo_name_dict, groups, title_dict)
    
    # 保存图表
    output_path = os.path.dirname(__file__) + "/test_boxplot.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')
    print(f"图表已保存到: {output_path}")
    
    # 显示图表
    plt.show()
    a = 1
    
    
    

