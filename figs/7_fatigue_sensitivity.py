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
        pattern = f"{algo_key}_{fatigue} - Evaluate/{metric_name}"
        
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
    # 配置
    metric_name_file_dir_list = {
        "Makespan": os.path.dirname(__file__) + "/revise_r1/env_len_fatigue.csv",
        "Overwork": os.path.dirname(__file__) + "/revise_r1/overwork_fatigue.csv",
        "Progress": os.path.dirname(__file__) + "/revise_r1/succes_fatigue.csv"
    }
    
    # 指标名称映射（CSV中的列名）
    metric_column_mapping = {
        "Makespan": "EpEnvLen",
        "Overwork": "EpOverCost",
        "Progress": "EpProgress"
    }
    
    data_algo_name_dict = {
        "test_rl_filter_49600_2025-07-20_12-17-12": "PF-CD3Q",
        "test_ppolag_filter_dis_49600_2025-08-08_13-49-16": "PPO-Lag"
    }
    
    fatigue_values = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    # 创建图形，3个子图
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
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
    for idx, (metric_name, file_path) in enumerate(metric_name_file_dir_list.items()):
        ax = axes[idx]
        
        # 读取CSV文件
        df = pd.read_csv(file_path)
        
        # 获取对应的列名
        metric_column = metric_column_mapping[metric_name]
        
        # 为每个算法提取数据
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
        
        # 设置标签和标题
        ax.set_xlabel('Fatigue', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric_name, fontsize=12, fontweight='bold')
        ax.set_title(metric_name, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=10)
        ax.set_xlim(-0.05, 1.05)
        ax.tick_params(labelsize=10)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存曲线图
    output_path = os.path.dirname(__file__) + "/fatigue_sensitivity.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')
    print(f"曲线图已保存到: {output_path}")
    
    # 显示曲线图
    plt.show()
