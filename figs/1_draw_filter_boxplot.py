import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def draw_one_sub_pic(ax, data_dict, title, plot_type="predict_loss"):
    '''绘制箱线图'''
    # 准备数据
    box_data = []
    labels = []
    
    if plot_type == "predict_loss":
        # 第一个子图：显示PF、KF、EKF、True_value预测损失
        # 需要处理True_value、PF-predict、KF-predict、EKF-predict
        filter_data = {"True_value": [], "PF": [], "KF": [], "EKF": []}
        
        for key, data_list in data_dict.items():
            for data_file in data_list:
                if os.path.exists(data_file):
                    df = pd.read_csv(data_file, header=None)
                    data_names = df.loc[0]
                    
                    for data_name, i in zip(data_names, range(len(data_names))):
                        if "step" in data_name or "MIN" in data_name or "MAX" in data_name:
                            continue
                        else:
                            algo_name = data_name.split(' ')[0]
                            if algo_name in data_algo_name_dict:
                                values = df[i][1:].dropna().astype(float)
                                if len(values) > 0:
                                    # 根据key确定过滤器类型
                                    if "True_value" in key:
                                        filter_data["True_value"].extend(values)
                                    elif "PF-predict" in key:
                                        filter_data["PF"].extend(values)
                                    elif "EKF-predict" in key:
                                        filter_data["EKF"].extend(values)
                                    elif "KF-predict" in key:
                                        filter_data["KF"].extend(values)
                                    else:
                                        assert False, "Unknown key: " + key

        # 为每个过滤器类型创建箱线图
        for filter_type, values in filter_data.items():
            if values:
                box_data.append(values)
                labels.append(filter_type)
    
    else:
        # 第二和第三个子图：显示PF、KF、EKF的精度
        # 收集所有算法的数据并按过滤器类型分组
        filter_groups = {"PF": [], "KF": [], "EKF": []}
        
        for key, data_list in data_dict.items():
            for data_file in data_list:
                if os.path.exists(data_file):
                    df = pd.read_csv(data_file, header=None)
                    data_names = df.loc[0]
                    
                    for data_name, i in zip(data_names, range(len(data_names))):
                        if "step" in data_name or "MIN" in data_name or "MAX" in data_name:
                            continue
                        else:
                            algo_name = data_name.split(' ')[0]
                            if algo_name in data_algo_name_dict:
                                values = df[i][1:].dropna().astype(float)
                                if len(values) > 0:
                                    # 根据key确定过滤器类型
                                    if "PF" in key and ("fatigue" in key or "recover" in key):
                                        filter_groups["PF"].extend(values)
                                    elif ("EKF" in key or "ekf" in key) and ("fatigue" in key or "recover" in key):
                                        filter_groups["EKF"].extend(values)
                                    elif ("KF" in key or "kf" in key) and ("fatigue" in key or "recover" in key):
                                        filter_groups["KF"].extend(values)

        
        # 为每个过滤器类型创建箱线图
        for filter_type, values in filter_groups.items():
            if values:
                box_data.append(values)
                labels.append(filter_type)
    
    # 绘制箱线图
    if box_data:
        bp = ax.boxplot(box_data, labels=labels, patch_artist=True, widths=0.65)
        
        # 设置颜色
        colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 
                 'lightpink', 'lightgray', 'lightsteelblue', 'lightseagreen', 'lightgoldenrodyellow']
        for patch, color in zip(bp['boxes'], colors * (len(bp['boxes']) // len(colors) + 1)):
            patch.set_facecolor(color)
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        # ax.set_xlabel('Category', fontsize=14)
        ax.set_ylabel('Value', fontsize=16)
        ax.tick_params(axis='x', rotation=45, labelsize=14)
        ax.tick_params(axis='y', labelsize=14)
        ax.grid(True, alpha=0.3)
        
        # 只在第一个子图添加标记解释
        if plot_type == "predict_loss":
            # 在解释文本位置绘制示例图标
            legend_x, legend_y = 0.02, 0.95
            # 绘制示例三角形
            ax.scatter(legend_x + 0.05, legend_y, marker='>', color='red', s=80, 
                      transform=ax.transAxes, zorder=10)
            # 绘制示例直线
            ax.plot([legend_x + 0.02, legend_x + 0.08], [legend_y, legend_y], 
                   color='red', linewidth=1, transform=ax.transAxes, zorder=10)
            # 添加文字说明
            ax.text(legend_x + 0.12, legend_y, ': Mean value', 
                   transform=ax.transAxes, fontsize=14, verticalalignment='center', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # 添加平均值标记
        for i, (data, label) in enumerate(zip(box_data, labels)):
            mean_val = np.mean(data)
            # 根据数据类型调整显示精度
            if plot_type == "predict_loss":
                ax.text(i+1, mean_val + (max(data) - min(data)) * 0.04, f'{mean_val:.5f}', 
                       ha='center', va='bottom', fontsize=14, fontweight='bold', color='black')
            else:
                ax.text(i+1, mean_val + (max(data) - min(data)) * 0.04, f'{mean_val:.4f}', 
                       ha='center', va='bottom', fontsize=14, fontweight='bold', color='black')
            
            # 添加朝右的三角形标记
            ax.scatter(i+1, mean_val, marker='>', color='red', s=120, zorder=5)
            # 添加穿越三角形的直线
            ax.plot([i+1-0.13, i+1+0.13], [mean_val, mean_val], color='red', linewidth=1, zorder=6)
    
    return ax

def create_boxplot_figure(fig_data_dict, fig_names, main_title):
    '''创建包含多个子图的箱线图'''
    fig, axes = plt.subplots(1, len(fig_data_dict), figsize=(6*len(fig_data_dict), 6))
    
    if len(fig_data_dict) == 1:
        axes = [axes]
    
    plot_types = ["predict_loss", "fatigue_accuracy", "recovery_accuracy"]
    
    for i, (name, data_dict) in enumerate(fig_data_dict.items()):
        plot_type = plot_types[i] if i < len(plot_types) else "predict_loss"
        draw_one_sub_pic(axes[i], data_dict, fig_names[i], plot_type)
    
    # plt.suptitle(main_title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig

'''=========================================================Main drawing code=========================================================='''
if __name__ == '__main__':
    ## type one: task-level fatigue value predict loss
    # wandb.define_metric("Evaluate/EpPredictLossCompare", step_metric="Evaluate/step_episode")
    # wandb.define_metric("Evaluate/EpFilterPredictLoss", step_metric="Evaluate/step_episode")
    # wandb.define_metric("Evaluate/EpFilterPredictLoss_kf", step_metric="Evaluate/step_episode")
    # wandb.define_metric("Evaluate/EpFilterPredictLoss_ekf", step_metric="Evaluate/step_episode")
   
    ## Type two: fatigue coefficient accuracy
    # wandb.define_metric("Evaluate/EpFilterFatigueCoeAccu", step_metric="Evaluate/step_episode")
    # wandb.define_metric("Evaluate/EpFilterFatigueCoeAccu_kf", step_metric="Evaluate/step_episode")
    # wandb.define_metric("Evaluate/EpFilterFatigueCoeAccu_ekf", step_metric="Evaluate/step_episode")

    ## Type three: fatigue recovery coefficient accuracy
    # wandb.define_metric("Evaluate/EpFilterRecoverCoeAccu", step_metric="Evaluate/step_episode")
    # wandb.define_metric("Evaluate/EpFilterRecoverCoeAccu_kf", step_metric="Evaluate/step_episode")
    # wandb.define_metric("Evaluate/EpFilterRecoverCoeAccu_ekf", step_metric="Evaluate/step_episode")

    
    ## data source
    file_dir_list = {
        "EpPredictLossCompare": os.path.dirname(__file__) + "/filter" + "/EpPredictLossCompare.csv",
        "EpFilterPredictLoss": os.path.dirname(__file__) + "/filter" + "/EpFilterPredictLoss.csv",
        "EpFilterPredictLoss_kf": os.path.dirname(__file__) + "/filter" + "/EpFilterPredictLoss_kf.csv",
        "EpFilterPredictLoss_ekf": os.path.dirname(__file__) + "/filter" + "/EpFilterPredictLoss_ekf.csv",
        "EpFilterRecoverCoeAccu": os.path.dirname(__file__) + "/filter" + "/EpFilterRecoverCoeAccu.csv",
        "EpFilterRecoverCoeAccu_kf": os.path.dirname(__file__) + "/filter" + "/EpFilterRecoverCoeAccu_kf.csv",
        "EpFilterRecoverCoeAccu_ekf": os.path.dirname(__file__) + "/filter" + "/EpFilterRecoverCoeAccu_ekf.csv",
        "EpFilterFatigueCoeAccu": os.path.dirname(__file__) + "/filter" + "/EpFilterFatigueCoeAccu.csv",
        "EpFilterFatigueCoeAccu_kf": os.path.dirname(__file__) + "/filter" + "/EpFilterFatigueCoeAccu_kf.csv",
        "EpFilterFatigueCoeAccu_ekf": os.path.dirname(__file__) + "/filter" + "/EpFilterFatigueCoeAccu_ekf.csv",
    }
    
    ## data source, each csv file data is from diverse algorithms
    # example
    # Evaluate/step_episode	
    # 5_test_dqn_49600_2025-07-27_11-39-32 - _step	5_test_dqn_49600_2025-07-27_11-39-32 - _step__MIN	5_test_dqn_49600_2025-07-27_11-39-32 - _step__MAX	
    # 5_test_dqn_49600_2025-07-27_11-39-32 - Evaluate/EpPredictLossCompare	5_test_dqn_49600_2025-07-27_11-39-32 - Evaluate/EpPredictLossCompare__MIN	5_test_dqn_49600_2025-07-27_11-39-32 - Evaluate/EpPredictLossCompare__MAX

    data_algo_name_dict = {
        # 1_test_rl_filter_test_49600_2025-07-25_15-02-16  D3QN
        "2_test_rl_filter_49600_2025-07-29_22-22-18": "D3QN",
        # "3_test_rl_filter_49600_2025-07-20_12-17-12": "PF-CD3Q",
        "3_test_rl_filter_54400_2025-07-20_12-17-12": "PF-CD3Q",
        # "4_test_rl_filter_49600_2025-07-27_14-41-12": "PF-CD3QP",
        "5_test_dqn_49600_2025-07-27_11-39-32": "DQN",
        "6_test_dqn_test_49600_2025-07-29_13-21-06": "PF-DQN",
        "7_test_ppo_dis_49600_2025-07-31_13-37-58": "PPO",
        "8_test_ppo_dis_49600_2025-07-30_13-18-07": "PF-PPO",
        "9_test_ppolag_filter_dis_49600_2025-08-08_13-49-16": "PPO-Lag",
        "10_test_ppolag_filter_dis_49600_2025-08-08_13-46-57": "PF-PPO-Lag"
    }
    
    ## Draw boxplot, three sub-figures
    fig_one_boxplot_name = ["True_value", "PF-predict", "KF-predict", "EKF-predict"]
    fig_two_boxplot_name = ["PF-fatigue", "KF-fatigue", "EKF-fatigue"]
    fig_three_boxplot_name = ["PF-recover", "KF-recover", "EKF-recover"]
    
    fig_one_boxplot_data = {
        "True_value": [file_dir_list["EpPredictLossCompare"]],
        "PF-predict": [file_dir_list["EpFilterPredictLoss"]],
        "KF-predict": [file_dir_list["EpFilterPredictLoss_kf"]],
        "EKF-predict": [file_dir_list["EpFilterPredictLoss_ekf"]]
    }
    
    fig_two_boxplot_data = {
        "PF-fatigue": [file_dir_list["EpFilterFatigueCoeAccu"]],
        "KF-fatigue": [file_dir_list["EpFilterFatigueCoeAccu_kf"]],
        "EKF-fatigue": [file_dir_list["EpFilterFatigueCoeAccu_ekf"]]
    }
    
    fig_three_boxplot_data = {
        "PF-recover": [file_dir_list["EpFilterRecoverCoeAccu"]],
        "KF-recover": [file_dir_list["EpFilterRecoverCoeAccu_kf"]],
        "EKF-recover": [file_dir_list["EpFilterRecoverCoeAccu_ekf"]]
    }
    
    # 设置matplotlib样式
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # 绘制三个子图
    fig_data_dict = {
        "Predict Loss": fig_one_boxplot_data,
        "Fatigue Coefficient": fig_two_boxplot_data,
        "Recovery Coefficient": fig_three_boxplot_data
    }
    
    fig_names = ["Predict loss comparison", "Fatigue parameter accuracy", "Recovery parameter accuracy"]
    
    # 创建箱线图
    fig = create_boxplot_figure(fig_data_dict, fig_names, "")
    
    # 保存图片
    output_path = os.path.dirname(__file__) + "/filter_boxplot.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')
    print(f"箱线图已保存到: {output_path}")
    
    # 显示图片
    plt.show()
    

