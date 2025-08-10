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
        axes[idx].set_title(metric_name, fontweight='bold', fontsize=14)
        axes[idx].set_xlabel('', fontsize=12)
        axes[idx].set_ylabel(metric_name, fontsize=12)
        axes[idx].tick_params(axis='x', rotation=30, labelsize=12)
        axes[idx].tick_params(axis='y', labelsize=12)
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
    
    ax1.set_xlabel('Human num', fontsize=12)
    ax1.set_ylabel('Makespan (Test)', fontsize=12)
    ax1.set_title('Makespan vs Human Number', fontweight='bold', fontsize=16)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax1.tick_params(axis='both', labelsize=12)
    
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
    
    ax2.set_xlabel('Human num', fontsize=12)
    ax2.set_ylabel('Overwork (Test)', fontsize=12)
    ax2.set_title('Overwork vs Human Number', fontweight='bold', fontsize=16)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12, handlelength=4, handleheight=2)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax2.tick_params(axis='both', labelsize=12)
    
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
    
    ax3.set_xlabel('Robot num', fontsize=12)
    ax3.set_ylabel('Makespan (Test)', fontsize=12)
    ax3.set_title('Makespan vs Robot Number', fontweight='bold', fontsize=16)
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax3.tick_params(axis='both', labelsize=12)
    
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
    
    ax4.set_xlabel('Robot num', fontsize=12)
    ax4.set_ylabel('Overwork (Test)', fontsize=12)
    ax4.set_title('Overwork vs Robot Number', fontweight='bold', fontsize=16)
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12, handlelength=4, handleheight=2)
    ax4.grid(True, alpha=0.3)
    ax4.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax4.tick_params(axis='both', labelsize=12)
    
    plt.tight_layout()
    return fig

def generate_statistics_table(metric_name_file_dir_list, data_algo_name_dict, groups):
    """
    生成不同human/robot组合下的统计表格并保存为CSV
    """
    # 定义human和robot的数量
    human_nums = [1, 2, 3]
    robot_nums = [1, 2, 3]
    
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
    
    # 创建统计表格数据
    stats_data = []
    
    # 为每个算法计算不同human/robot组合的统计值
    for algo_key, algo_name in data_algo_name_dict.items():
        if algo_key not in makespan_data or algo_key not in overwork_data:
            continue
            
        makespan_algo_data = makespan_data[algo_key].dropna()
        overwork_algo_data = overwork_data[algo_key].dropna()
        
        if len(makespan_algo_data) != 450 or len(overwork_algo_data) != 450:
            print(f"警告: {algo_name} 数据条数不是450: makespan={len(makespan_algo_data)}, overwork={len(overwork_algo_data)}")
            continue
        
        for h_idx, human_num in enumerate(human_nums):
            for r_idx, robot_num in enumerate(robot_nums):
                # 计算数据索引
                start_idx = h_idx * 150 + r_idx * 50
                end_idx = start_idx + 50
                
                if end_idx <= len(makespan_algo_data):
                    makespan_values = makespan_algo_data.iloc[start_idx:end_idx]
                    overwork_values = overwork_algo_data.iloc[start_idx:end_idx]
                    
                    # makespan计算：平均值和标准差
                    makespan_mean = makespan_values.mean()
                    makespan_std = makespan_values.std()
                    
                    # overwork计算：非零次数/总次数
                    overwork_rate = (overwork_values != 0).sum() / len(overwork_values)
                    
                    # 添加到统计表格
                    stats_data.append({
                        'Algorithm': algo_name,
                        'Human_Num': human_num,
                        'Robot_Num': robot_num,
                        'Makespan_Mean': round(makespan_mean, 2),
                        'Makespan_Std': round(makespan_std, 2),
                        'Overwork_Rate': round(overwork_rate, 4)
                    })
    
    # 创建DataFrame
    stats_df = pd.DataFrame(stats_data)
    
    # 保存为CSV文件
    # output_path = os.path.dirname(__file__) + "/human_robot_statistics.csv"
    # stats_df.to_csv(output_path, index=False)
    # print(f"统计表格已保存到: {output_path}")
    
    # 打印表格摘要
    print("\n=== 统计表格摘要 ===")
    print(f"总记录数: {len(stats_df)}")
    print(f"算法数量: {stats_df['Algorithm'].nunique()}")
    print(f"Human数量范围: {stats_df['Human_Num'].min()} - {stats_df['Human_Num'].max()}")
    print(f"Robot数量范围: {stats_df['Robot_Num'].min()} - {stats_df['Robot_Num'].max()}")
    
    # 显示前几行数据
    print("\n=== 前10行数据 ===")
    print(stats_df.head(10))
    
    # 按算法分组的统计摘要
    print("\n=== 按算法分组的统计摘要 ===")
    for algo in stats_df['Algorithm'].unique():
        algo_data = stats_df[stats_df['Algorithm'] == algo]
        print(f"\n{algo}:")
        print(f"  Makespan平均值范围: {algo_data['Makespan_Mean'].min():.2f} - {algo_data['Makespan_Mean'].max():.2f}")
        print(f"  Overwork率范围: {algo_data['Overwork_Rate'].min():.4f} - {algo_data['Overwork_Rate'].max():.4f}")
    
    return stats_df

def generate_performance_analysis(stats_df):
    """
    基于统计表格生成性能分析和排名
    """
    print("\n" + "="*60)
    print("性能分析报告")
    print("="*60)
    
    # 1. 整体性能排名（基于所有human/robot组合的平均值）
    print("\n1. 整体性能排名（基于所有组合的平均值）")
    print("-" * 50)
    
    # 计算每个算法的整体平均值
    overall_stats = stats_df.groupby('Algorithm').agg({
        'Makespan_Mean': 'mean',
        'Overwork_Rate': 'mean'
    }).round(4)
    
    # Makespan排名（越小越好）
    makespan_ranking = overall_stats.sort_values('Makespan_Mean')
    print("\nMakespan排名（越小越好）:")
    for i, (algo, row) in enumerate(makespan_ranking.iterrows(), 1):
        print(f"  {i}. {algo}: {row['Makespan_Mean']:.2f}")
    
    # Overwork排名（越小越好）
    overwork_ranking = overall_stats.sort_values('Overwork_Rate')
    print("\nOverwork率排名（越小越好）:")
    for i, (algo, row) in enumerate(overwork_ranking.iterrows(), 1):
        print(f"  {i}. {algo}: {row['Overwork_Rate']:.4f}")
    
    # 2. 不同Human数量下的性能分析
    print("\n2. 不同Human数量下的性能分析")
    print("-" * 50)
    
    for human_num in [1, 2, 3]:
        human_data = stats_df[stats_df['Human_Num'] == human_num]
        human_stats = human_data.groupby('Algorithm').agg({
            'Makespan_Mean': 'mean',
            'Overwork_Rate': 'mean'
        }).round(4)
        
        print(f"\nHuman数量 = {human_num}:")
        print("  Makespan排名:")
        makespan_rank = human_stats.sort_values('Makespan_Mean')
        for i, (algo, row) in enumerate(makespan_rank.iterrows(), 1):
            print(f"    {i}. {algo}: {row['Makespan_Mean']:.2f}")
        
        print("  Overwork率排名:")
        overwork_rank = human_stats.sort_values('Overwork_Rate')
        for i, (algo, row) in enumerate(overwork_rank.iterrows(), 1):
            print(f"    {i}. {algo}: {row['Overwork_Rate']:.4f}")
    
    # 3. 不同Robot数量下的性能分析
    print("\n3. 不同Robot数量下的性能分析")
    print("-" * 50)
    
    for robot_num in [1, 2, 3]:
        robot_data = stats_df[stats_df['Robot_Num'] == robot_num]
        robot_stats = robot_data.groupby('Algorithm').agg({
            'Makespan_Mean': 'mean',
            'Overwork_Rate': 'mean'
        }).round(4)
        
        print(f"\nRobot数量 = {robot_num}:")
        print("  Makespan排名:")
        makespan_rank = robot_stats.sort_values('Makespan_Mean')
        for i, (algo, row) in enumerate(makespan_rank.iterrows(), 1):
            print(f"    {i}. {algo}: {row['Makespan_Mean']:.2f}")
        
        print("  Overwork率排名:")
        overwork_rank = robot_stats.sort_values('Overwork_Rate')
        for i, (algo, row) in enumerate(overwork_rank.iterrows(), 1):
            print(f"    {i}. {algo}: {row['Overwork_Rate']:.4f}")
    
    # 4. 最佳配置分析
    print("\n4. 最佳配置分析")
    print("-" * 50)
    
    # 找到每个算法的最佳配置（Makespan最小且Overwork率最低）
    best_configs = []
    for algo in stats_df['Algorithm'].unique():
        algo_data = stats_df[stats_df['Algorithm'] == algo]
        
        # 计算综合得分（Makespan标准化 + Overwork率标准化）
        makespan_range = algo_data['Makespan_Mean'].max() - algo_data['Makespan_Mean'].min()
        overwork_range = algo_data['Overwork_Rate'].max() - algo_data['Overwork_Rate'].min()
        
        # 避免除零错误
        if makespan_range == 0:
            makespan_norm = 0
        else:
            makespan_norm = (algo_data['Makespan_Mean'] - algo_data['Makespan_Mean'].min()) / makespan_range
            
        if overwork_range == 0:
            overwork_norm = 0
        else:
            overwork_norm = (algo_data['Overwork_Rate'] - algo_data['Overwork_Rate'].min()) / overwork_range
        
        # 综合得分（越小越好）
        algo_data = algo_data.copy()
        algo_data['Combined_Score'] = makespan_norm + overwork_norm
        
        best_idx = algo_data['Combined_Score'].idxmin()
        if pd.isna(best_idx):
            # 如果所有得分都是NaN，选择Makespan最小的配置
            best_idx = algo_data['Makespan_Mean'].idxmin()
        best_config = algo_data.loc[best_idx]
        
        best_configs.append({
            'Algorithm': best_config['Algorithm'],
            'Best_Human_Num': best_config['Human_Num'],
            'Best_Robot_Num': best_config['Robot_Num'],
            'Best_Makespan': best_config['Makespan_Mean'],
            'Best_Overwork_Rate': best_config['Overwork_Rate'],
            'Combined_Score': best_config['Combined_Score']
        })
    
    # 按综合得分排序
    best_configs_df = pd.DataFrame(best_configs).sort_values('Combined_Score')
    
    print("每个算法的最佳配置（综合Makespan和Overwork）:")
    for i, (_, row) in enumerate(best_configs_df.iterrows(), 1):
        print(f"  {i}. {row['Algorithm']}: Human={row['Best_Human_Num']}, Robot={row['Best_Robot_Num']}")
        print(f"     Makespan: {row['Best_Makespan']:.2f}, Overwork率: {row['Best_Overwork_Rate']:.4f}")
    
    # 保存最佳配置到CSV
    # best_configs_df.to_csv(os.path.dirname(__file__) + "/best_configurations.csv", index=False)
    # print(f"\n最佳配置已保存到: {os.path.dirname(__file__) + '/best_configurations.csv'}")
    
    return best_configs_df

def generate_academic_table_csv(stats_df):
    """
    生成符合学术论文格式的表格CSV
    参考Table 3 Makespan performance in the testing phase的格式
    """
    print("\n" + "="*60)
    print("生成学术论文格式表格...")
    print("="*60)
    
    # 创建透视表，将数据重新组织为表格格式
    # 行：算法，列：不同的human/robot组合
    makespan_pivot = stats_df.pivot_table(
        values='Makespan_Mean', 
        index='Algorithm', 
        columns=['Human_Num', 'Robot_Num'],
        aggfunc='mean'
    )
    
    overwork_pivot = stats_df.pivot_table(
        values='Overwork_Rate', 
        index='Algorithm', 
        columns=['Human_Num', 'Robot_Num'],
        aggfunc='mean'
    )
    
    # 计算每个算法的平均值
    makespan_mean = stats_df.groupby('Algorithm')['Makespan_Mean'].mean()
    overwork_mean = stats_df.groupby('Algorithm')['Overwork_Rate'].mean()
    
    # 重新排列列名，使其符合H1,R1, H1,R2, H1,R3, H2,R1, H2,R2, H2,R3, H3,R1, H3,R2, H3,R3的顺序
    column_order = [(1,1), (1,2), (1,3), (2,1), (2,2), (2,3), (3,1), (3,2), (3,3)]
    
    # 创建Makespan表格
    makespan_table = pd.DataFrame()
    for col in column_order:
        if col in makespan_pivot.columns:
            makespan_table[f'H{col[0]},R{col[1]}'] = makespan_pivot[col]
    
    # 添加平均值列
    makespan_table['Mean'] = makespan_mean
    
    # 创建Overwork表格
    overwork_table = pd.DataFrame()
    for col in column_order:
        if col in overwork_pivot.columns:
            overwork_table[f'H{col[0]},R{col[1]}'] = overwork_pivot[col]
    
    # 添加平均值列
    overwork_table['Mean'] = overwork_mean
    
    # 按照指定顺序重新排列算法
    desired_order = ['DQN', 'PPO', 'D3QN', 'PPO-Lag', 'PF-DQN', 'PF-PPO', 'PF-CD3Q', 'PF-PPO-Lag']
    
    # 只保留存在的算法
    available_algorithms = [algo for algo in desired_order if algo in makespan_table.index]
    
    # 重新排列Makespan表格
    makespan_table = makespan_table.reindex(available_algorithms)
    
    # 重新排列Overwork表格
    overwork_table = overwork_table.reindex(available_algorithms)
    
    # 标记最佳性能（最小值）
    def mark_best_performance(df):
        df_marked = df.copy()
        for col in df.columns:
            min_val = df[col].min()
            df_marked[col] = df[col].apply(lambda x: f"**{x:.2f}**" if abs(x - min_val) < 0.01 else f"{x:.2f}")
        return df_marked
    
    makespan_table_marked = mark_best_performance(makespan_table)
    overwork_table_marked = mark_best_performance(overwork_table)
    
    # 保存Makespan表格
    # makespan_output_path = os.path.dirname(__file__) + "/table_makespan_performance.csv"
    # makespan_table_marked.to_csv(makespan_output_path)
    # print(f"Makespan性能表格已保存到: {makespan_output_path}")
    
    # 保存Overwork表格
    # overwork_output_path = os.path.dirname(__file__) + "/table_overwork_performance.csv"
    # overwork_table_marked.to_csv(overwork_output_path)
    # print(f"Overwork性能表格已保存到: {overwork_output_path}")
    
    # 保存原始数值表格（用于LaTeX）
    # makespan_table.to_csv(os.path.dirname(__file__) + "/table_makespan_raw.csv")
    # overwork_table.to_csv(os.path.dirname(__file__) + "/table_overwork_raw.csv")
    
    # 打印表格预览
    # print("\n=== Makespan性能表格预览 ===")
    # print("Table 3 Makespan performance in the testing phase")
    # print("-" * 80)
    # print(makespan_table_marked.to_string())
    
    # print("\n=== Overwork性能表格预览 ===")
    # print("Table 4 Overwork performance in the testing phase")
    # print("-" * 80)
    # print(overwork_table_marked.to_string())
    
    # 生成LaTeX表格代码
    def generate_latex_table(df, title, metric_name):
        latex_code = "\\begin{table}[htbp]\n"
        latex_code += "\\centering\n"
        latex_code += "\\caption{" + title + "}\\label{tab:" + metric_name.lower() + "}\n"
        latex_code += f"\\begin{{tabular}}{{{'c' * (len(df.columns) + 1)}}}\n"
        latex_code += "\\hline\n"
        
        # 表头
        header = "Algorithm & " + " & ".join(df.columns) + " \\\\"
        latex_code += header + "\n"
        latex_code += "\\hline\n"
        
        # 数据行
        for idx, row in df.iterrows():
            data_row = f"{idx} & " + " & ".join([f"{val:.2f}" for val in row]) + " \\\\"
            latex_code += data_row + "\n"
        
        latex_code += "\\hline\n"
        latex_code += "\\end{tabular}\n"
        latex_code += "\\end{table}\n"
        
        return latex_code
    
    # 保存LaTeX代码
    # makespan_latex = generate_latex_table(makespan_table, "Makespan performance in the testing phase", "makespan")
    # overwork_latex = generate_latex_table(overwork_table, "Overwork performance in the testing phase", "overwork")
    
    # with open(os.path.dirname(__file__) + "/table_makespan_latex.tex", 'w') as f:
    #     f.write(makespan_latex)
    
    # with open(os.path.dirname(__file__) + "/table_overwork_latex.tex", 'w') as f:
    #     f.write(overwork_latex)
    
    # print(f"\nLaTeX表格代码已保存到:")
    # print(f"  {os.path.dirname(__file__)}/table_makespan_latex.tex")
    # print(f"  {os.path.dirname(__file__)}/table_overwork_latex.tex")
    
    # 生成合并表格
    print("\n" + "="*60)
    print("生成合并表格...")
    print("="*60)
    
    # 创建合并表格
    combined_data = []
    
    # 添加Makespan数据
    for algo in makespan_table.index:
        makespan_row = [algo] + list(makespan_table.loc[algo])
        combined_data.append(makespan_row)
    
    # 添加分隔行
    combined_data.append([''] * len(makespan_table.columns) + [''])
    
    # 添加Overwork数据
    for algo in overwork_table.index:
        overwork_row = [algo] + list(overwork_table.loc[algo])
        combined_data.append(overwork_row)
    
    # 创建合并表格DataFrame
    combined_columns = ['Algorithm'] + list(makespan_table.columns)
    combined_df = pd.DataFrame(combined_data, columns=combined_columns)
    
    # 保存合并表格
    # combined_output_path = os.path.dirname(__file__) + "/table_combined_performance.csv"
    # combined_df.to_csv(combined_output_path, index=False)
    # print(f"合并表格已保存到: {combined_output_path}")
    
    # 生成合并表格的LaTeX代码
    def generate_combined_latex_table(makespan_df, overwork_df):
        latex_code = "\\begin{table*}[htbp]\n"
        latex_code += "\\centering\n"
        latex_code += "\\caption{Combined Makespan and Overwork performance in the testing phase}\\label{tab:combined}\n"
        latex_code += f"\\begin{{tabular}}{{{'c' * (len(makespan_df.columns) + 1)}}}\n"
        latex_code += "\\hline\n"
        
        # 表头
        header = "Algorithm & " + " & ".join(makespan_df.columns) + " \\\\"
        latex_code += header + "\n"
        latex_code += "\\hline\n"
        
        # Makespan数据行
        latex_code += "\\multicolumn{" + str(len(makespan_df.columns) + 1) + "}{c}{\\textbf{Makespan}} \\\\\n"
        latex_code += "\\hline\n"
        
        # 找到每个列的最佳值（最小值）
        makespan_best = {}
        for col in makespan_df.columns:
            if col != 'Mean':  # 排除Mean列
                makespan_best[col] = makespan_df[col].min()
        
        for idx, row in makespan_df.iterrows():
            values = []
            for col in makespan_df.columns:
                val = row[col]
                if col in makespan_best and abs(val - makespan_best[col]) < 0.01:
                    values.append(f"\\textbf{{{val:.2f}}}")
                else:
                    values.append(f"{val:.2f}")
            data_row = f"{idx} & " + " & ".join(values) + " \\\\"
            latex_code += data_row + "\n"
        
        latex_code += "\\hline\n"
        latex_code += "\\multicolumn{" + str(len(overwork_df.columns) + 1) + "}{c}{\\textbf{Overwork}} \\\\\n"
        latex_code += "\\hline\n"
        
        # Overwork数据行（不加粗）
        for idx, row in overwork_df.iterrows():
            values = []
            for col in overwork_df.columns:
                val = row[col]
                if col == 'Mean':
                    values.append(f"{val:.3f}")  # 均值保留3位有效数字
                else:
                    values.append(f"{val:.2f}")   # 其他列保留2位有效数字
            data_row = f"{idx} & " + " & ".join(values) + " \\\\"
            latex_code += data_row + "\n"
        
        latex_code += "\\hline\n"
        latex_code += "\\end{tabular}\n"
        latex_code += "\\end{table*}\n"
        
        return latex_code
    
    # 保存合并表格的LaTeX代码
    combined_latex = generate_combined_latex_table(makespan_table, overwork_table)
    with open(os.path.dirname(__file__) + "/table_combined_latex.tex", 'w') as f:
        f.write(combined_latex)
    
    print(f"合并表格LaTeX代码已保存到: {os.path.dirname(__file__)}/table_combined_latex.tex")
    
    # 打印合并表格预览
    print("\n=== 合并表格预览 ===")
    print("Table 5 Combined Makespan and Overwork performance in the testing phase")
    print("-" * 100)
    print(combined_df.to_string(index=False))
    
    return makespan_table, overwork_table, combined_df

if __name__ == '__main__':
    ## 2 metric for 2 subfigure, each subfigure has 9 algorithms, draw the boxplot
    ## data source
    metric_name_file_dir_list = {
        "Makespan (Test)": os.path.dirname(__file__) + "/test" + "/EpEnvLen.csv",
        "Overwork (Test)": os.path.dirname(__file__) + "/test" + "/EpOverCost.csv",
        # "Progress (Test)": os.path.dirname(__file__) + "/test" + "/EpProgress.csv"
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
        "6_test_dqn_test_49600_2025-07-29_13-21-06": "PF-DQN",
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
    
    # 生成统计表格
    print("\n" + "="*50)
    print("开始生成统计表格...")
    stats_df = generate_statistics_table(metric_name_file_dir_list, data_algo_name_dict, groups)
    print("统计表格生成完成！")
    print("="*50)
    
    # 生成性能分析报告
    generate_performance_analysis(stats_df)
    
    # 生成学术论文格式表格
    generate_academic_table_csv(stats_df)
    
    
    

