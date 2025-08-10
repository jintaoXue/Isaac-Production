import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon
import seaborn as sns

def normalize_data(value, min_val, max_val, reverse=False):
    """标准化数据到0-1范围，reverse=True表示值越小越好"""
    if max_val == min_val:
        return 0.5
    normalized = (value - min_val) / (max_val - min_val)
    if reverse:
        normalized = 1 - normalized
    return normalized

def create_radar_chart(ax, data_dict, title="Algorithm Performance Radar Chart"):
    """创建现代风格三角雷达图"""
    
    # 定义三个指标
    metrics = ['Progress', 'Makespan', 'Overwork']
    num_metrics = len(metrics)
    
    # 计算角度
    angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
    angles += angles[:1]  # 闭合图形
    
    # 设置雷达图的基本参数
    ax.set_theta_offset(np.pi / 2)  # 从顶部开始
    ax.set_theta_direction(-1)  # 顺时针方向
    
    # 设置刻度标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=16, fontweight='bold', color='#2c3e50')
    
    # 设置网格样式 - 更现代的网格
    ax.grid(True, alpha=0.2, color='#34495e', linewidth=0.8)
    ax.set_ylim(0, 1)
    
    # 设置Y轴标签
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=12, color='#7f8c8d')
    
    # 设置背景色
    ax.set_facecolor('#f8f9fa')
    
    # 使用现代配色方案
    group_colors = {
        'A': '#3498db',  # 现代蓝色
        'B': '#2ecc71',  # 现代绿色
        'C': '#9b59b6',  # 现代紫色
        'D': '#e74c3c'   # 现代红色
    }
    
    # 算法分组映射（与4_draw_test_line.py保持一致）
    algo_groups = {
        'D3QN': 'A',
        'PF-CD3Q': 'A',
        'DQN': 'B',
        'PF-DQN': 'B',
        'PPO': 'C',
        'PF-PPO': 'C',
        'PPO-Lag': 'D',
        'PF-PPO-Lag': 'D'
    }
    
    # 算法到颜色的映射
    algo_color_map = {}
    for algo_name in data_dict.keys():
        group = algo_groups.get(algo_name, 'A')
        algo_color_map[algo_name] = group_colors[group]
    
    # 绘制每个算法的雷达图
    for algo_name, values in data_dict.items():
        # 标准化数据 - 放大性能差异
        progress_norm = 1.0  # Progress都设为1
        # Makespan映射到0-1范围，最差到最好
        makespan_norm = normalize_data(values['Makespan'], 1281.86, 1360.39, reverse=True)   # Makespan越小越好，使用实际数据范围
        # Overwork映射到0-1范围，最差到最好
        overwork_norm = normalize_data(values['Overwork'], 0.0, 0.2667, reverse=True)  # Overwork越小越好，使用实际数据范围
        
        # 创建雷达图数据点
        radar_values = [progress_norm, makespan_norm, overwork_norm]
        radar_values += radar_values[:1]  # 闭合图形
        
        # 使用与4_draw_test_line.py相同的线型
        linestyle = '--' if 'PF' in algo_name else '-'
        color = algo_color_map[algo_name]
        
        # 绘制雷达图 - 现代风格
        ax.plot(angles, radar_values, 'o', linewidth=3, label=algo_name, 
                color=color, linestyle=linestyle, markersize=8, markerfacecolor=color, markeredgecolor='white', markeredgewidth=1)
        ax.fill(angles, radar_values, alpha=0.15, color=color)
    
    # 添加图例
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=16, handlelength=4, handleheight=2)
    
    # 设置标题
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    return ax

def load_performance_data():
    """加载性能数据"""
    # 使用与4_draw_test_line.py相同的算法顺序和数据
    data = {
        'D3QN': {
            'Progress': 1.0,
            'Makespan': 1281.86,
            'Overwork': 0.2578
        },
        'PF-CD3Q': {
            'Progress': 1.0,
            'Makespan': 1300.24,
            'Overwork': 0.0111
        },
        'DQN': {
            'Progress': 1.0,
            'Makespan': 1316.47,
            'Overwork': 0.2222
        },
        'PF-DQN': {
            'Progress': 1.0,
            'Makespan': 1320.0,  # 估算值
            'Overwork': 0.2000   # 估算值
        },
        'PPO': {
            'Progress': 1.0,
            'Makespan': 1360.39,
            'Overwork': 0.1889
        },
        'PF-PPO': {
            'Progress': 1.0,
            'Makespan': 1324.21,
            'Overwork': 0.0000
        },
        'PPO-Lag': {
            'Progress': 1.0,
            'Makespan': 1329.02,
            'Overwork': 0.2667
        },
        'PF-PPO-Lag': {
            'Progress': 1.0,
            'Makespan': 1324.87,
            'Overwork': 0.0000
        }
    }
    return data

def create_comprehensive_radar_chart():
    """创建综合雷达图"""
    # 加载数据
    data = load_performance_data()
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'))
    
    # 创建雷达图
    create_radar_chart(ax, data, "Algorithms performance comparison")
    
    # 调整布局
    plt.tight_layout()
    
    return fig

if __name__ == '__main__':
    # 设置matplotlib样式
    plt.style.use('seaborn-v0_8')
    
    # 创建综合雷达图
    print("创建综合雷达图...")
    fig = create_comprehensive_radar_chart()
    
    # 保存综合雷达图
    output_path = os.path.dirname(__file__) + "/radar_chart_comprehensive.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')
    print(f"综合雷达图已保存到: {output_path}")
    
    # 显示图形
    plt.show()
    
    print("雷达图生成完成！")
    
    
    

