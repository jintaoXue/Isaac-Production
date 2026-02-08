import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def perform_ttest_and_annotate(ax, plot_df, algo_order, metric_name, baseline_algo='PF-CD3Q', idx=0, is_bar_plot=False):
    """对每个算法与基准算法进行独立样本T检验，并在图表上显示P值和T值"""
    # 获取基准算法的数据
    baseline_data = plot_df[plot_df['Algorithm'] == baseline_algo][metric_name].dropna().values
    
    if len(baseline_data) == 0:
        print(f"警告: 未找到基准算法 {baseline_algo} 的数据")
        return
    
    # 获取当前y轴范围
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    
    # 找到基准算法在algo_order中的位置
    try:
        baseline_idx = algo_order.index(baseline_algo)
    except ValueError:
        print(f"警告: 基准算法 {baseline_algo} 不在算法列表中")
        return
    
    # 收集所有需要显示的算法索引（排除基准算法）
    algo_indices = []
    for i, algo in enumerate(algo_order):
        if algo != baseline_algo:
            algo_data = plot_df[plot_df['Algorithm'] == algo][metric_name].dropna().values
            if len(algo_data) > 0:
                algo_indices.append(i)
    
    # 在PF-CD3Q位置显示注释（图一和图二都显示）
    if is_bar_plot:
        # 图二（条形图）：在PF-CD3Q位置显示注释，放在0.1高度
        ax.text(baseline_idx, 0.1, "T-test\nbaseline", 
               ha='center', va='bottom', fontsize=9, 
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.8, edgecolor='#666666', linewidth=1),
               zorder=100)
    else:
        # 图一（箱线图）：在PF-CD3Q位置显示注释，放在800高度
        ax.text(baseline_idx, 800, "T-test\nbaseline", 
               ha='center', va='center', fontsize=9, 
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.8, edgecolor='#666666', linewidth=1),
               zorder=100)
    
    # 对每个其他算法进行T检验并显示统计信息
    for idx_in_list, i in enumerate(algo_indices):
        algo = algo_order[i]
        algo_data = plot_df[plot_df['Algorithm'] == algo][metric_name].dropna().values
        
        # 执行独立样本T检验（假设方差不等）
        t_stat, p_value = stats.ttest_ind(baseline_data, algo_data, equal_var=False)
        
        # 打印T检验结果
        print(f"{baseline_algo} vs {algo}: t={t_stat:.4f}, p={p_value:.4f}")
        
        # 格式化显示文本
        t_text = f"t={t_stat:.3f}"
        # 如果p值保留三位小数后为0.000，则显示约等于符号
        p_formatted = f"{p_value:.3f}"
        if p_formatted == "0.000":
            p_text = "p≈0"
        else:
            p_text = f"p={p_formatted}"
        stat_text = f"{t_text}\n{p_text}"
        
        if is_bar_plot:
            # 图二（条形图）：交错放在0.15和0.1高度
            # 根据算法在algo_order中的位置索引i来决定高度
            # PF-CD3Q已经在baseline_idx位置显示"T-test baseline"在0.1高度
            # 其他算法根据位置索引交错：偶数索引用0.15，奇数索引用0.1
            # 由于每个算法x坐标不同，即使都在0.1也不会重叠
            if i % 2 == 0:
                text_y = 0.15  # 偶数位置索引放在0.15
            else:
                text_y = 0.1   # 奇数位置索引放在0.1（与PF-CD3Q的T-test注释同一高度，但x坐标不同）
        else:
            # 图一（箱线图）：统一放在800高度
            text_y = 800
        
        # 绘制文本
        ax.text(i, text_y, stat_text, 
               ha='center', va='center' if not is_bar_plot else 'bottom', fontsize=9, 
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='#333333', linewidth=1),
               zorder=100)  # 使用更高的zorder确保在最上层


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
        
        # 执行T检验并标注显著性
        print(f"\n=== {metric_name} 的T检验结果 (基准: PF-CD3Q) ===")
        is_bar_plot = (idx == 1)  # Overwork使用条形图
        
        # 先调整y轴范围，为统计信息留出空间
        y_min_orig, y_max_orig = axes[idx].get_ylim()
        y_range_orig = y_max_orig - y_min_orig
        
        if not is_bar_plot:
            # 图一（箱线图）：y轴上界设为2000，下界设为700
            new_y_max = 2000
            new_y_min = 700
            axes[idx].set_ylim(bottom=new_y_min, top=new_y_max)
        else:
            # 图二（条形图）：确保y轴下界为0，并留出空间显示0.1和0.15
            current_y_max = max(y_max_orig, 0.2)  # 至少到0.2
            axes[idx].set_ylim(bottom=0.0, top=current_y_max + 0.05)
        
        # 执行T检验并显示统计信息（在y轴调整之后）
        perform_ttest_and_annotate(axes[idx], plot_df, algo_order, metric_name, baseline_algo='PF-CD3Q', idx=idx, is_bar_plot=is_bar_plot)
        
        # 只在第一个子图添加标记解释（在T检验之后）
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
    
    
    

