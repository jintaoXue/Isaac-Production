import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def extract_data_from_csv(df, algo_key, metric_name, fatigue_values):
    """
    从CSV文件中提取指定算法在不同fatigue值下的指标数据
    
    Args:
        df: pandas DataFrame
        algo_key: 算法标识符（如 "test_rl_filter_49600_2025-07-20_12-17-12"）
        metric_name: 指标名称（如 "EpEnvLen", "EpOverCost", "EpProgress"）
        fatigue_values: fatigue值列表
    
    Returns:
        dict: {fatigue_value: mean_value} 的字典
    """
    results = {}
    
    for fatigue in fatigue_values:
        # 构建列名模式：{algo_key}_{fatigue} - Evaluate/{metric_name}
        pattern = f"{algo_key}_ftg_{fatigue} - Evaluate/{metric_name}"
        
        # 查找匹配的列
        matching_cols = [col for col in df.columns if pattern in col]
        
        if matching_cols:
            # 取第一个匹配的列（忽略MIN和MAX列）
            col_name = matching_cols[0]
            # 提取数据（跳过第一行，因为第一行可能是列名）
            values = df[col_name].iloc[1:].astype(float)
            # 计算平均值（也可以使用其他统计量，如最后一个值）
            mean_value = values.mean()
            results[fatigue] = mean_value
        else:
            # 如果找不到数据，设为NaN
            results[fatigue] = np.nan
    
    return results


if __name__ == '__main__':
    
    #     PF inference at every step, per-human and per-subtask parameter tracking, and attention-based 
    # Q-networks add overhead. There’s no report of step latency or throughput. Please provide (i) 
    # per-step inference time, (ii) cost vs. number of humans/subtasks/particles, and (iii) whether 
    # decisions meet cycle-time constraints in your simulated line. 
    
    # 配置
    metric_name_file_dir_list = {
        "Makespan": os.path.dirname(__file__) + "/revise_r1/env_len_fatigue.csv",
        "Overwork": os.path.dirname(__file__) + "/revise_r1/overwork_fatigue.csv",
        "Progress": os.path.dirname(__file__) + "/revise_r1/succes_fatigue.csv",
        "ValueDistribution": os.path.dirname(__file__) + "/revise_r1/fatigue_values.csv"
    }
    
    # 指标名称映射（CSV中的列名）
    metric_column_mapping = {
        "Makespan": "EpEnvLen",
        "Overwork": "EpOverCost",
        "Progress": "EpProgress",
        "ValueDistribution": "EpOverworkPhyValues"
    }
    
    data_algo_name_dict = {
        "test_rl_filter_49600_2025-07-20_12-17-12": "PF-CD3Q",
        "test_ppolag_filter_dis_49600_2025-08-08_13-49-16": "PPO-Lag"
    }
    
    # fatigue 从 0 到 1，间隔 0.05
    fatigue_values = [round(x * 0.05, 2) for x in range(0, 21)]
    
    # 构建绘图配置：常规指标各一个子图，ValueDistribution 为每个算法单独一个子图
    plot_config = []
    for metric_name in metric_name_file_dir_list.keys():
        if metric_name == "ValueDistribution":
            for algo_key, algo_name in data_algo_name_dict.items():
                plot_config.append(
                    {"metric": metric_name, "algo_key": algo_key, "algo_name": algo_name}
                )
        else:
            plot_config.append({"metric": metric_name, "algo_key": None, "algo_name": None})
    
    num_plots = len(plot_config)
    num_cols = 3  # 第一行 3 张图
    num_rows = 2  # 固定两行
    fig = plt.figure(figsize=(5 * num_cols, 5 * num_rows))
    gridspec = fig.add_gridspec(
        num_rows,
        num_cols * 2,
        height_ratios=[0.8, 1.0]
    )  # 第一行高 0.8，第二行保持 1.0
    axes = []
    for idx in range(num_plots):
        if idx < num_cols:
            # 第一行：每个图占据 2 列（总共 6 列）
            col_start = idx * 2
            col_end = col_start + 2
            ax = fig.add_subplot(gridspec[0, col_start:col_end])
        else:
            # 第二行：两张图平均分担 6 列
            second_row_idx = idx - num_cols
            col_span = 3  # 每个图占 3 列
            col_start = second_row_idx * col_span
            col_end = col_start + col_span
            ax = fig.add_subplot(gridspec[1, col_start:col_end])
        axes.append(ax)
    
    # 算法颜色和线型（与2_draw_training_curve.py保持一致）
    algo_styles = {
        "PF-CD3Q": {
            "color": "#1f77b4",  # 蓝色 (Group A)
            "linestyle": "--"    # 虚线（因为以PF-开头）
        },
        "PPO-Lag": {
            "color": "#e377c2",  # 粉色 (Group D)
            "linestyle": "-"     # 实线（因为不以PF-开头）
        }
    }
    
    # 处理每个指标
    for idx, config in enumerate(plot_config):
        metric_name = config["metric"]
        file_path = metric_name_file_dir_list[metric_name]
        ax = axes[idx]
        
        # 读取CSV文件
        df = pd.read_csv(file_path)
        
        # 获取对应的列名
        metric_column = metric_column_mapping[metric_name]
        
        if metric_name != "ValueDistribution":
            # 普通指标：按 fatigue 画平均曲线
            for algo_key, algo_name in data_algo_name_dict.items():
                data_dict = extract_data_from_csv(df, algo_key, metric_column, fatigue_values)
                
                # 提取x和y值
                x_values = []
                y_values = []
                for fv in fatigue_values:
                    if fv in data_dict and not np.isnan(data_dict[fv]):
                        x_values.append(fv)
                        y_values.append(data_dict[fv])
                
                # 绘制曲线
                if len(x_values) > 0:
                    style = algo_styles[algo_name]
                    ax.plot(x_values, y_values, marker='o', linewidth=2, markersize=6, 
                           label=algo_name, color=style["color"], linestyle=style["linestyle"])
        else:
            # fatigue_values.csv：每个 fatigue 上指定算法的 overwork values 箱线图
            algo_key = config["algo_key"]
            algo_name = config["algo_name"]
            style = algo_styles[algo_name]
            boxplot_data = []
            positions = []
            for fv in fatigue_values:
                pattern = f"{algo_key}_ftg_{fv} - Evaluate2/{metric_column}"
                matching_cols = [col for col in df.columns if pattern in col]
                if matching_cols:
                    col_name = matching_cols[0]
                    values = df[col_name].iloc[1:].astype(float).dropna()
                    if len(values) > 0:
                        boxplot_data.append(values.tolist())
                        positions.append(fv)
            
            if len(boxplot_data) > 0:
                bp = ax.boxplot(
                    boxplot_data,
                    positions=positions,
                    widths=0.03,
                    patch_artist=True,
                    showfliers=False
                )
                # 统一配色
                for box in bp['boxes']:
                    box.set(facecolor=style["color"], alpha=0.4)
                for median in bp['medians']:
                    median.set(color=style["color"], linewidth=2)
                # 为每个 fatigue 值添加短横线（显示 fatigue limit）
                limit_label = 'Fatigue constraint value'
                label_used = False
                for pos in positions:
                    ax.hlines(
                        y=pos,
                        xmin=pos - 0.015,
                        xmax=pos + 0.015,
                        colors="#ff7f0e",
                        linewidth=2,
                        label=limit_label if not label_used else None
                    )
                    label_used = True
        
        # 设置标签和标题
        ax.set_xlabel('Fatigue constraint', fontsize=14)
        if metric_name == "ValueDistribution":
            ax.set_ylabel('Overwork values', fontsize=14)
            vd_title = 'Overwork Value Distribution'
            if config["algo_name"]:
                vd_title += f" - {config['algo_name']}"
            ax.set_title(vd_title, fontsize=16)
        else:
            ax.set_ylabel(metric_name, fontsize=14)
            ax.set_title(metric_name, fontsize=16)
        ax.grid(True, alpha=0.3, linestyle='--')
        if metric_name != "ValueDistribution":
            ax.legend(fontsize=12)
        elif len(positions) > 0:
            ax.legend(fontsize=12, loc='lower right', bbox_to_anchor=(1, 0.2))
        row_idx = 0 if idx < num_cols else 1
        if row_idx == 0:
            ax.set_xlim(-0.05, 1.05)
            ax.set_xticks(np.arange(0, 1.01, 0.1))
        else:
            ax.set_xlim(-0.05, 1)
            ticks = np.arange(0, 1.01, 0.1)
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{t:.1f}" for t in ticks])
            ax.set_ylim(0, 1)
            ax.set_yticks(np.arange(0, 1.01, 0.1))
        ax.tick_params(labelsize=12)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存曲线图
    output_path = os.path.dirname(__file__) + "/fatigue_sensitivity.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')
    print(f"曲线图已保存到: {output_path}")
    
    # 显示曲线图
    plt.show()
