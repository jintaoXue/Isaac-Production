import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def draw_one_sub_pic(ax, data, title, x_lable, algo_dict, color_dict, log_x, y_tick_f, y_tick, alpha):
    '''loss curve plot'''
    df = pd.read_csv(data, header=None)
    ax.set_title(title, fontsize=20)
    ax.set_xlabel(x_lable, fontsize=15)
    ax.tick_params(axis='both', which='both', labelsize=15)
    if log_x:
        ax.set_xscale('log')
    if y_tick_f == "1":
        # ax.set_yscale('symlog')
        # ax.set_yticks(y_tick)
        # ax.set_yscale('log')  # or 'logit'
        ax.set_yscale('function', functions=(forward, inverse))
        ax.set_ylim(y_tick)
    elif y_tick_f == "2":
        ax.set_yscale('function', functions=(inverse, forward))
        ax.set_ylim(y_tick)
    elif y_tick_f == "3":
        # ax.set_yscale('function', functions=(inverse, forward))
        ax.set_ylim(y_tick)
    # ax.tick_params(axis='both', size=12)
    # ax.tick_params(axis='both', size=12)
    algo_dict_rev = {v: k for k, v in algo_dict.items()}
    x = np.array(df[0][1:].to_list(), dtype=np.float32)
    data_names = df.loc[0]
    data_dict = {}
    for data_name, i in zip(data_names, range(len(data_names))):
        if "step" in data_name or "MIN" in data_name or "MAX" in data_name:
            pass
        else:
            data_dict[data_name.split(' ')[0]] = df[i][1:]
    #one data
    for name in algo_dict.values():
        label = algo_dict_rev[name]
        raw_y, color = data_dict[name], color_dict[label]
        # raw_y, color = np.array(data_dict[name], dtype=np.float32), color_dict[label]
        smoothed_y = smooth_line(raw_y, alpha)
        ax.plot(x, smoothed_y, '-', color=color, label=label, ms=5, linewidth=2)
    # ax.legend(
    #     fontsize="x-large",
    #     handlelength=5.0)
        # handleheight=3)
    # get the legend object
    leg = ax.legend(ncol=2)
    # change the line width for the legend
    for line in leg.get_lines():
        line.set_linewidth(6.0)

    return 

def smooth_line(data, alpha=0.005):

    smoothed_d = data.ewm(alpha=alpha,adjust=False).mean()
    np_data = np.array(smoothed_d.to_list(), dtype=np.float32)
    return np_data

# Function x**(1/2)
def forward(x):
    return x**(2)


def inverse(x):
    return x**(1/2)

def draw_training_curve(ax, data_file, title, x_label, y_label, x_range, y_range, y_log=False, y_log_func=None, alpha=0.005, groups=None, add_zoom=False):
    """绘制训练曲线"""
    if os.path.exists(data_file):
        df = pd.read_csv(data_file, header=None)
        data_names = df.loc[0]
        
        # 设置图表属性
        ax.set_title(title, fontsize=18)
        ax.set_xlabel(x_label, fontsize=14)
        ax.set_ylabel(y_label, fontsize=14)
        ax.tick_params(axis='both', which='both', labelsize=14)
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        
        # 为第四张图设置自定义x轴刻度
        if x_range[1] == 2100 and y_range[0] == 0.2 and y_range[1] == 1.1:  # 第四张图
            # 在[0, 500]区间更密集的刻度，但只显示部分标签避免重叠
            custom_x_ticks = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000, 1200, 1400, 1600, 1800, 2000, 2100]
            custom_x_labels = ['0', '', '', '', '200', '', '', '', '400', '', '', '600', '', '', '', '1000', '', '', '', '', '2000', '']
            ax.set_xticks(custom_x_ticks)
            ax.set_xticklabels(custom_x_labels)
        
        if y_log:
            if y_log_func:
                ax.set_yscale('function', functions=y_log_func)
            else:
                # 使用自定义的非均匀刻度
                if y_range[0] < 0 and y_range[1] > 0:
                    y_min, y_max = y_range
                    if y_min == -100 and y_max == 1.5:
                        custom_ticks = [-1, 0, 0.5, 1, 1.5]
                        ax.set_yticks(custom_ticks)
                        ax.set_yticklabels([str(tick) for tick in custom_ticks])
                elif y_range[0] == 1100 and y_range[1] == 3000:
                    # 为Makespan图设置自定义刻度，在1200-1600区间更密集
                    custom_ticks = [1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2200, 2400, 2600, 2800, 3000]
                    ax.set_yticks(custom_ticks)
                    ax.set_yticklabels([str(tick) for tick in custom_ticks])
                else:
                    ax.set_yscale('log')
        
        # 准备数据
        x = np.array(df[0][1:].to_list(), dtype=np.float32)
        data_dict = {}
        for data_name, i in zip(data_names, range(len(data_names))):
            if "step" in data_name or "MIN" in data_name or "MAX" in data_name:
                continue
            else:
                data_dict[data_name.split(' ')[0]] = df[i][1:]
        
        # 过滤x>2000的数据
        if x_range[1] == 2100:  # 如果是第二张图或第四张图
            mask = x <= 2000
            x = x[mask]
            for key in data_dict:
                data_dict[key] = data_dict[key][mask]
        
        # 定义group颜色和线型
        group_colors = {
            'A': '#1f77b4',  # 蓝色
            'B': '#2ca02c',  # 绿色
            'C': '#9467bd',  # 紫色
            'D': '#e377c2'   # 粉色
        }
        
        # 按group顺序绘制数据
        if groups is None:
            # 如果没有提供groups，使用默认的data_algo_name_dict
            for i, (algo_key, algo_name) in enumerate(data_algo_name_dict.items()):
                # 在data_dict中查找包含algo_key的数据
                found_data = False
                for data_key in data_dict.keys():
                    if algo_key in data_key:
                        raw_y = data_dict[data_key]
                        smoothed_y = smooth_line(raw_y, alpha)
                        color = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan'][i % 10]
                        ax.plot(x, smoothed_y, '-', color=color, label=algo_name, linewidth=2)
                        found_data = True
                        break
                
                if not found_data:
                    print(f"警告: 未找到算法 {algo_name} ({algo_key}) 的数据")
        else:
            for group_name, group_dict in groups:
                for i, (algo_key, algo_name) in enumerate(group_dict.items()):
                    # 在data_dict中查找包含algo_key的数据
                    found_data = False
                    for data_key in data_dict.keys():
                        if algo_key in data_key:
                            raw_y = data_dict[data_key]
                            smoothed_y = smooth_line(raw_y, alpha)
                            
                            # 为group A特殊处理：D3QN和PF-CD3Q使用相同颜色
                            if group_name == 'A':
                                if algo_name in ['D3QN', 'PF-CD3Q']:
                                    color = group_colors[group_name]  # 使用group A的主颜色
                                else:
                                    color = '#ff7f0e'  # 橙色，用于PF-CD3QP
                            else:
                                color = group_colors[group_name]  # 其他group使用单一颜色
                            
                            # 根据是否带PF选择线型
                            if algo_name.startswith('PF-'):
                                linestyle = '--'  # 虚线表示带PF
                            else:
                                linestyle = '-'   # 实线表示不带PF
                            
                            ax.plot(x, smoothed_y, linestyle, color=color, label=algo_name, linewidth=2)
                            found_data = True
                            break
                    
                    if not found_data:
                        print(f"警告: 未找到算法 {algo_name} ({algo_key}) 的数据")
        
        # 设置图例 - 按group分组显示
        handles, labels = ax.get_legend_handles_labels()
        
        # 重新组织handles和labels，按group顺序
        new_handles = []
        new_labels = []
        
        for group_name, group_dict in groups:
            for algo_key, algo_name in group_dict.items():
                # 找到对应的handle
                for i, label in enumerate(labels):
                    if label == algo_name:
                        new_handles.append(handles[i])
                        new_labels.append(labels[i])
                        break
        
        # 创建新的图例
        leg = ax.legend(new_handles, new_labels, ncol=2, fontsize=14, handlelength=3.0)
        for line in leg.get_lines():
            line.set_linewidth(2.0)
        
        ax.grid(True, alpha=0.3)
        
        # 如果需要添加放大框
        if add_zoom and x_range[1] == 2100:  # 只对图二和图四添加放大框
            # 创建放大框
            if title == "Progress (Evaluate during training)":
                zoom_ax = ax.inset_axes([0.25, 0.45, 0.5, 0.3])  # 图四：放大尺寸，位置调整
            else:
                if title == "Makespan (Evaluate during training)":
                    zoom_ax = ax.inset_axes([0.3, 0.3, 0.6, 0.4])  # 图二：拉长宽度
                else:
                    zoom_ax = ax.inset_axes([0.3, 0.3, 0.4, 0.4])  # 其他图：正常尺寸
            
            # 在放大框中重新绘制x>500的数据，直接使用已经smooth处理过的数据
            mask = x > 500
            if np.any(mask):
                x_zoom = x[mask]
                # 重新绘制所有线条，使用已经smooth处理过的数据
                for line in ax.get_lines():
                    # 获取原始数据
                    x_data = line.get_xdata()
                    y_data = line.get_ydata()
                    
                    # 过滤x>500的数据
                    mask_zoom = x_data > 500
                    if np.any(mask_zoom):
                        x_zoom_data = x_data[mask_zoom]
                        y_zoom_data = y_data[mask_zoom]
                        
                        # 使用相同的颜色和线型
                        color = line.get_color()
                        linestyle = line.get_linestyle()
                        
                        # 图二的zoom框中减小线宽并进一步平滑
                        if title == "Makespan (Evaluate during training)":
                            # 对图二的zoom框数据进一步平滑
                            y_zoom_series = pd.Series(y_zoom_data)
                            y_zoom_smoothed = smooth_line(y_zoom_series, alpha=0.06)  # 使用更小的alpha值
                            zoom_ax.plot(x_zoom_data, y_zoom_smoothed, linestyle, color=color, linewidth=1.5)
                        else:
                            zoom_ax.plot(x_zoom_data, y_zoom_data, linestyle, color=color, linewidth=1.5)
            
            # 设置放大框的属性
            zoom_ax.set_xlim(500, 2000)
            if title == "Makespan (Evaluate during training)":
                zoom_ax.set_ylim(1275, 1450)  # 图二的放大范围
            elif title == "Progress (Evaluate during training)":
                zoom_ax.set_ylim(0.97, 1.02)  # 图四的放大范围
            
            zoom_ax.grid(True, alpha=0.3)
            zoom_ax.tick_params(labelsize=10)
            zoom_ax.set_title('Zoom', fontsize=12)
            
            # 添加连接线
            ax.indicate_inset_zoom(zoom_ax, edgecolor='black', alpha=0.5)

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
        
        ax.set_title(title, fontsize=14)
        ax.set_ylabel(y_label, fontsize=12)
        ax.tick_params(axis='x', rotation=45, labelsize=12)
        ax.tick_params(axis='y', labelsize=12)
        ax.grid(True, alpha=0.3)
        
        # 添加平均值标记
        for i, (data, label) in enumerate(zip(box_data, labels)):
            mean_val = np.mean(data)
            ax.text(i+1, mean_val, f'{mean_val:.3f}', 
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

def create_figure(metric_data, algo_dict, groups=None):
    """创建包含4个子图的图表"""
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    axes = axes.flatten()
    
    # 子图1: Return (Training) - 使用EMA平滑
    draw_training_curve(
        axes[0], 
        metric_data["Return (Training)"], 
        "Return (Training)",
        "Training Steps", 
        "Episodic Return", 
        [0, int(2.8e6)], 
        [-1.5, 1.5], 
        y_log=True, 
        y_log_func=None,
        alpha=0.0025,
        groups=groups
    )
    
    # 子图2: Makespan (Evaluate during training)
    draw_training_curve(
        axes[1], 
        metric_data["Makespan (Evaluate during training)"], 
        "Makespan (Evaluate during training)",
        "Evaluate Episode", 
        "Makespan", 
        [0, 2100], 
        [1100, 3000], 
        y_log=True, 
        y_log_func=(inverse, forward),
        alpha=0.008,
        groups=groups,
        add_zoom=True
    )
    
    # 子图3: Overwork (Evaluate during training) - 曲线图
    draw_training_curve(
        axes[2], 
        metric_data["Overwork (Evaluate during training)"], 
        "Overwork (Evaluate during training)",
        "Evaluate Episode", 
        "Overwork", 
        [0, 2100], 
        [0, 5.0], 
        y_log=False, 
        y_log_func=None,
        alpha=0.008,
        groups=groups
    )
    
    # 子图4: Progress (Evaluate during training)
    draw_training_curve(
        axes[3], 
        metric_data["Progress (Evaluate during training)"], 
        "Progress (Evaluate during training)",
        "Evaluate Episode", 
        "Progress", 
        [0, 2100], 
        [0.4, 1.1], 
        y_log=True, 
        y_log_func=(inverse, forward),
        alpha=0.03,
        groups=groups,
        add_zoom=True
    )
    
    plt.tight_layout()
    return fig

'''=========================================================Main drawing code=========================================================='''
if __name__ == '__main__':
    ## 4 metric for 4 subfigure, each subfigure has 9 algorithms, draw the line using Time weighted EMA
    ## data source
    metric_name_file_dir_list = {
        "Return (Training)": os.path.dirname(__file__) + "/train" + "/Mrewards.csv",
        "Makespan (Evaluate during training)": os.path.dirname(__file__) + "/train" + "/EpEnvLen.csv",
        "Overwork (Evaluate during training)": os.path.dirname(__file__) + "/train" + "/EpOverCost.csv",
        "Progress (Evaluate during training)": os.path.dirname(__file__) + "/train" + "/EpProgress.csv"
    }
    
    # 定义算法分组
    group_A = {
        "penalty_4070_rl_filter_2025-07-29_22-22-18": "D3QN",
        "4070_rl_filter_2025-07-20_12-17-12": "PF-CD3Q",
        # "mask_penalty_4090_rl_filter_2025-07-27_14-41-12": "PF-CD3QP",
    }
    group_B = {
        "penalty_4070_dqn_2025-07-27_11-39-32": "DQN",
        "4090_dqn_2025-07-29_13-21-06": "PF-DQN",
    }
    group_C = {
        "4070_penalty_ppo_dis_2025-07-31_13-37-58": "PPO",
        "4090_ppo_dis_2025-07-30_13-18-07": "PF-PPO",
    }
    group_D = {
        "4070_9_ppolag_filter_dis_2025-08-08_13-49-16": "PPO-Lag",
        "4090_10_ppolag_filter_dis_2025-08-08_13-46-57": "PF-PPO-Lag"
    }
    
    # 合并所有算法字典
    data_algo_name_dict = {**group_A, **group_B, **group_C, **group_D}
    
    # 定义groups
    groups = [('A', group_A), ('B', group_B), ('C', group_C), ('D', group_D)]
    
    # 创建图表
    fig = create_figure(metric_name_file_dir_list, data_algo_name_dict, groups)
    
    # 保存图表
    output_path = os.path.dirname(__file__) + "/training_curves.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')
    print(f"图表已保存到: {output_path}")
    
    # 显示图表
    plt.show()
    a = 1
    
    
    

