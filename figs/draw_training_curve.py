import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Function x**(1/2)
def forward(x):
    return x**(2)

def inverse(x):
    return x**(1/2)

def draw_training_curve(ax, data_file, algo_name, x_label, y_label, x_range, y_range, y_log=False, y_log_func=None):
    """绘制训练曲线"""
    if os.path.exists(data_file):
        df = pd.read_csv(data_file, header=None)
        data_names = df.loc[0]
        
        for data_name, i in zip(data_names, range(len(data_names))):
            if "step" in data_name or "MIN" in data_name or "MAX" in data_name:
                continue
            else:
                # 检查是否包含算法名称
                if any(algo_key in data_name for algo_key in data_algo_name_dict.keys()):
                    values = df[i][1:].dropna().astype(float)
                    if len(values) > 0:
                        x_data = np.arange(len(values))
                        ax.plot(x_data, values, label=algo_name, linewidth=2)
    
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    
    if y_log:
        if y_log_func:
            ax.set_yscale('function', functions=y_log_func)
        else:
            ax.set_yscale('log')
    
    ax.grid(True, alpha=0.3)
    ax.legend()

def draw_boxplot(ax, data_dict, title, y_label):
    """绘制箱线图"""
    box_data = []
    labels = []
    
    for key, data_list in data_dict.items():
        for data_file in data_list:
            if os.path.exists(data_file):
                df = pd.read_csv(data_file, header=None)
                data_names = df.loc[0]
                
                for data_name, i in zip(data_names, range(len(data_names))):
                    if "step" in data_name or "MIN" in data_name or "MAX" in data_name:
                        continue
                    else:
                        # 检查是否包含算法名称
                        for algo_key, algo_name in data_algo_name_dict.items():
                            if algo_key in data_name:
                                values = df[i][1:].dropna().astype(float)
                                if len(values) > 0:
                                    box_data.append(values)
                                    labels.append(algo_name)
                                break
    
    if box_data:
        bp = ax.boxplot(box_data, labels=labels, patch_artist=True)
        
        # 设置颜色
        colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 
                 'lightpink', 'lightgray', 'lightsteelblue', 'lightseagreen', 'lightgoldenrodyellow']
        for patch, color in zip(bp['boxes'], colors * (len(bp['boxes']) // len(colors) + 1)):
            patch.set_facecolor(color)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylabel(y_label, fontsize=12)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        # 添加平均值标记
        for i, (data, label) in enumerate(zip(box_data, labels)):
            mean_val = np.mean(data)
            ax.text(i+1, mean_val, f'{mean_val:.3f}', 
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

def create_figure(metric_data, algo_dict):
    """创建包含4个子图的图表"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # 子图1: Return (Training)
    draw_training_curve(
        axes[0], 
        metric_data["Return (Training)"], 
        "Training Return",
        "Training Steps", 
        "Episodic Return", 
        [0, int(2.8e6)], 
        [-100, 1], 
        y_log=False, 
        y_log_func=None
    )
    axes[0].set_title("Return (Training)", fontsize=14, fontweight='bold')
    
    # 子图2: Makespan (Evaluate)
    draw_training_curve(
        axes[1], 
        metric_data["Makespan (Evaluate)"], 
        "Evaluation Makespan",
        "Evaluate Episode", 
        "Makespan", 
        [0, 2000], 
        [0, 3000], 
        y_log=True, 
        y_log_func=(inverse, forward)
    )
    axes[1].set_title("Makespan (Evaluate)", fontsize=14, fontweight='bold')
    
    # 子图3: Overwork (Evaluate) - 箱线图
    draw_boxplot(
        axes[2], 
        {"Overwork": [metric_data["Overwork (Evaluate)"]]}, 
        "Overwork (Evaluate)", 
        "Overwork"
    )
    axes[2].set_ylim([0, 1])
    
    # 子图4: Progress (Evaluate)
    draw_training_curve(
        axes[3], 
        metric_data["Progress (Evaluate)"], 
        "Evaluation Progress",
        "Evaluate Episode", 
        "Progress", 
        [0, 2000], 
        [0, 1], 
        y_log=True, 
        y_log_func=(inverse, forward)
    )
    axes[3].set_title("Progress (Evaluate)", fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig

'''=========================================================Main drawing code=========================================================='''
if __name__ == '__main__':
    ## 4 metric for 4 subfigure, each subfigure has 9 algorithms, draw the line using Time weighted EMA
    ## data source
    metric_name_file_dir_list = {
        "Return (Training)": os.path.dirname(__file__) + "/train" + "/Mrewards.csv",
        "Makespan (Evaluate)": os.path.dirname(__file__) + "/train" + "/EpEnvLen.csv",
        "Overwork (Evaluate)": os.path.dirname(__file__) + "/train" + "/EpOverCost.csv",
        "Progress (Evaluate)": os.path.dirname(__file__) + "/train" + "/EpProgress.csv"
    }
    
    data_algo_name_dict = {
        # 1_test_rl_filter_test_49600_2025-07-25_15-02-16  D3QN
        "penalty_4070_rl_filter_2025-07-29_22-22-18": "D3QN",
        "4070_rl_filter_2025-07-20_12-17-12": "PF-CD3Q",
        "mask_penalty_4090_rl_filter_2025-07-27_14-41-12": "PF-CD3QP",
        "penalty_4070_dqn_2025-07-27_11-39-32": "DQN",
        "4090_dqn_2025-07-29_13-21-06": "PF-DQN",
        "4070_penalty_ppo_dis_2025-07-31_13-37-58": "PPO",
        "4090_ppo_dis_2025-07-30_13-18-07": "PF-PPO",
        "nomask_4070_ppolag_filter_dis_2025-07-23_22-24-04": "PPO-lag",
        "4070_ppolag_filter_dis_2025-07-21_23-34-32": "PF-PPO-lag"
    }
    # 创建图表
    fig = create_figure(metric_name_file_dir_list, data_algo_name_dict)
    
    # 保存图表
    output_path = os.path.dirname(__file__) + "/training_curves.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"图表已保存到: {output_path}")
    
    # 显示图表
    plt.show()
    
    
    

