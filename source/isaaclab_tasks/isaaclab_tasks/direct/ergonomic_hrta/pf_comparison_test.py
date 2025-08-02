import numpy as np
import matplotlib.pyplot as plt
from pf_filter import ParticleFilter
from pf_filter_improved import ImprovedParticleFilter

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def run_comparison_test():
    """运行对比测试"""
    
    # 测试参数
    dt = 0.1
    num_steps = 300
    true_lambda = 0.5
    F0 = 0.0
    num_particles = 1000
    sigma_w = 0.005
    sigma_v = 0.005
    lamda_init = 0.3
    upper_bound = 2.0
    lower_bound = 0.01
    
    print("=== 粒子滤波Lambda参数估计对比测试 ===")
    print(f"真实lambda值: {true_lambda}")
    print(f"粒子数量: {num_particles}")
    print(f"时间步数: {num_steps}")
    print(f"过程噪声: {sigma_w}")
    print(f"测量噪声: {sigma_v}")
    print()
    
    # 运行原始粒子滤波
    print("运行原始粒子滤波...")
    pf_original = ParticleFilter(dt, num_steps, true_lambda, F0, num_particles, 
                                sigma_w, sigma_v, lamda_init, upper_bound, lower_bound)
    times_orig, F_estimates_orig, lambda_estimates_orig = pf_original.run()
    
    # 运行改进的粒子滤波
    print("运行改进的粒子滤波...")
    pf_improved = ImprovedParticleFilter(dt, num_steps, true_lambda, F0, num_particles, 
                                        sigma_w, sigma_v, lamda_init, upper_bound, lower_bound)
    times_imp, F_estimates_imp, lambda_estimates_imp, lambda_uncertainties_imp = pf_improved.run()
    
    # 计算性能指标
    def calculate_metrics(lambda_estimates, true_lambda):
        """计算性能指标"""
        # 去除前10个估计（初始阶段）
        stable_estimates = lambda_estimates[10:]
        
        mean_est = np.mean(stable_estimates)
        std_est = np.std(stable_estimates)
        variance_est = np.var(stable_estimates)
        relative_error = abs(mean_est - true_lambda) / true_lambda * 100
        
        # 计算收敛速度（方差随时间的变化）
        window_size = 20
        convergence_speed = []
        for i in range(window_size, len(stable_estimates)):
            window_var = np.var(stable_estimates[i-window_size:i])
            convergence_speed.append(window_var)
        
        return {
            'mean': mean_est,
            'std': std_est,
            'variance': variance_est,
            'relative_error': relative_error,
            'convergence_speed': convergence_speed,
            'estimates': stable_estimates
        }
    
    # 计算原始方法的指标
    metrics_orig = calculate_metrics(lambda_estimates_orig, true_lambda)
    
    # 计算改进方法的指标
    metrics_imp = calculate_metrics(lambda_estimates_imp, true_lambda)
    
    # 打印结果
    print("\n=== 性能对比结果 ===")
    print("原始粒子滤波:")
    print(f"  均值: {metrics_orig['mean']:.4f}")
    print(f"  标准差: {metrics_orig['std']:.4f}")
    print(f"  方差: {metrics_orig['variance']:.6f}")
    print(f"  相对误差: {metrics_orig['relative_error']:.2f}%")
    
    print("\n改进粒子滤波:")
    print(f"  均值: {metrics_imp['mean']:.4f}")
    print(f"  标准差: {metrics_imp['std']:.4f}")
    print(f"  方差: {metrics_imp['variance']:.6f}")
    print(f"  相对误差: {metrics_imp['relative_error']:.2f}%")
    
    # 计算改进程度
    variance_improvement = (metrics_orig['variance'] - metrics_imp['variance']) / metrics_orig['variance'] * 100
    error_improvement = (metrics_orig['relative_error'] - metrics_imp['relative_error']) / metrics_orig['relative_error'] * 100
    
    print(f"\n改进程度:")
    print(f"  方差减少: {variance_improvement:.2f}%")
    print(f"  误差减少: {error_improvement:.2f}%")
    
    # 绘制对比图
    plt.figure(figsize=(20, 12))
    
    # 1. Lambda估计对比
    plt.subplot(2, 3, 1)
    # 确保维度匹配
    lambda_orig_times = times_orig[:len(lambda_estimates_orig[1:])]
    lambda_imp_times = times_imp[:len(lambda_estimates_imp[1:])]
    plt.plot(lambda_orig_times, lambda_estimates_orig[1:], label='原始PF', color='red', alpha=0.7)
    plt.plot(lambda_imp_times, lambda_estimates_imp[1:], label='改进PF', color='blue', alpha=0.7)
    plt.axhline(y=true_lambda, color='black', linestyle='--', label='真实值')
    plt.xlabel('时间')
    plt.ylabel('Lambda估计')
    plt.legend()
    plt.grid(True)
    plt.title('Lambda估计对比')
    
    # 2. 方差随时间变化
    plt.subplot(2, 3, 2)
    window_size = 20
    orig_variances = []
    imp_variances = []
    time_windows = []
    
    lambda_orig_data = lambda_estimates_orig[1:]
    lambda_imp_data = lambda_estimates_imp[1:]
    
    for i in range(window_size, len(lambda_orig_data)):
        orig_var = np.var(lambda_orig_data[i-window_size:i])
        imp_var = np.var(lambda_imp_data[i-window_size:i])
        orig_variances.append(orig_var)
        imp_variances.append(imp_var)
        time_windows.append(lambda_orig_times[i])
    
    plt.plot(time_windows, orig_variances, label='原始PF', color='red')
    plt.plot(time_windows, imp_variances, label='改进PF', color='blue')
    plt.xlabel('时间')
    plt.ylabel('滑动窗口方差')
    plt.legend()
    plt.grid(True)
    plt.title('估计方差随时间变化')
    
    # 3. 有效粒子比例对比
    plt.subplot(2, 3, 3)
    if hasattr(pf_improved, 'effective_particle_ratios') and len(pf_improved.effective_particle_ratios) > 0:
        # 确保维度匹配
        min_len = min(len(lambda_imp_times), len(pf_improved.effective_particle_ratios))
        effective_times = lambda_imp_times[:min_len]
        effective_ratios = pf_improved.effective_particle_ratios[:min_len]
        plt.plot(effective_times, effective_ratios, label='改进PF', color='blue')
        plt.axhline(y=0.5, color='red', linestyle='--', label='重采样阈值')
    plt.xlabel('时间')
    plt.ylabel('有效粒子比例')
    plt.legend()
    plt.grid(True)
    plt.title('有效粒子比例')
    
    # 4. 估计分布对比
    plt.subplot(2, 3, 4)
    plt.hist(metrics_orig['estimates'], bins=30, alpha=0.7, label='原始PF', color='red', density=True)
    plt.hist(metrics_imp['estimates'], bins=30, alpha=0.7, label='改进PF', color='blue', density=True)
    plt.axvline(x=true_lambda, color='black', linestyle='--', label='真实值')
    plt.xlabel('Lambda估计值')
    plt.ylabel('密度')
    plt.legend()
    plt.grid(True)
    plt.title('估计分布对比')
    
    # 5. 收敛速度对比
    plt.subplot(2, 3, 5)
    plt.plot(metrics_orig['convergence_speed'], label='原始PF', color='red')
    plt.plot(metrics_imp['convergence_speed'], label='改进PF', color='blue')
    plt.xlabel('时间窗口')
    plt.ylabel('方差')
    plt.legend()
    plt.grid(True)
    plt.title('收敛速度对比')
    
    # 6. 不确定性估计
    plt.subplot(2, 3, 6)
    if lambda_uncertainties_imp is not None:
        lambda_estimates_array = np.array(lambda_estimates_imp[1:])
        lambda_uncertainties_array = np.array(lambda_uncertainties_imp)
        # 确保维度匹配
        min_len = min(len(lambda_imp_times), len(lambda_estimates_array), len(lambda_uncertainties_array))
        plot_times = lambda_imp_times[:min_len]
        plot_estimates = lambda_estimates_array[:min_len]
        plot_uncertainties = lambda_uncertainties_array[:min_len]
        
        plt.fill_between(plot_times, 
                        plot_estimates - 2*plot_uncertainties,
                        plot_estimates + 2*plot_uncertainties,
                        alpha=0.3, color='blue', label='95%置信区间')
        plt.plot(plot_times, plot_estimates, color='blue', linewidth=2, label='改进PF估计')
        plt.axhline(y=true_lambda, color='black', linestyle='--', label='真实值')
        plt.xlabel('时间')
        plt.ylabel('Lambda估计')
        plt.legend()
        plt.grid(True)
        plt.title('不确定性估计')
    
    plt.tight_layout()
    plt.savefig('pf_comparison_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return metrics_orig, metrics_imp

def test_different_scenarios():
    """测试不同场景下的性能"""
    
    scenarios = [
        {
            'name': '低噪声场景',
            'sigma_w': 0.001,
            'sigma_v': 0.001,
            'num_particles': 500
        },
        {
            'name': '高噪声场景',
            'sigma_w': 0.02,
            'sigma_v': 0.02,
            'num_particles': 2000
        },
        {
            'name': '少粒子场景',
            'sigma_w': 0.005,
            'sigma_v': 0.005,
            'num_particles': 200
        },
        {
            'name': '多粒子场景',
            'sigma_w': 0.005,
            'sigma_v': 0.005,
            'num_particles': 5000
        }
    ]
    
    results = {}
    
    for scenario in scenarios:
        print(f"\n=== 测试场景: {scenario['name']} ===")
        
        # 参数设置
        dt = 0.1
        num_steps = 200
        true_lambda = 0.5
        F0 = 0.0
        lamda_init = 0.3
        upper_bound = 2.0
        lower_bound = 0.01
        
        # 运行测试
        pf_original = ParticleFilter(dt, num_steps, true_lambda, F0, scenario['num_particles'], 
                                   scenario['sigma_w'], scenario['sigma_v'], lamda_init, upper_bound, lower_bound)
        times_orig, F_estimates_orig, lambda_estimates_orig = pf_original.run()
        
        pf_improved = ImprovedParticleFilter(dt, num_steps, true_lambda, F0, scenario['num_particles'], 
                                            scenario['sigma_w'], scenario['sigma_v'], lamda_init, upper_bound, lower_bound)
        times_imp, F_estimates_imp, lambda_estimates_imp, lambda_uncertainties_imp = pf_improved.run()
        
        # 计算指标
        def calculate_metrics(lambda_estimates, true_lambda):
            stable_estimates = lambda_estimates[10:]
            mean_est = np.mean(stable_estimates)
            variance_est = np.var(stable_estimates)
            relative_error = abs(mean_est - true_lambda) / true_lambda * 100
            return {'mean': mean_est, 'variance': variance_est, 'relative_error': relative_error}
        
        metrics_orig = calculate_metrics(lambda_estimates_orig, true_lambda)
        metrics_imp = calculate_metrics(lambda_estimates_imp, true_lambda)
        
        print(f"原始PF - 均值: {metrics_orig['mean']:.4f}, 方差: {metrics_orig['variance']:.6f}, 误差: {metrics_orig['relative_error']:.2f}%")
        print(f"改进PF - 均值: {metrics_imp['mean']:.4f}, 方差: {metrics_imp['variance']:.6f}, 误差: {metrics_imp['relative_error']:.2f}%")
        
        variance_improvement = (metrics_orig['variance'] - metrics_imp['variance']) / metrics_orig['variance'] * 100
        error_improvement = (metrics_orig['relative_error'] - metrics_imp['relative_error']) / metrics_orig['relative_error'] * 100
        
        print(f"改进程度 - 方差减少: {variance_improvement:.2f}%, 误差减少: {error_improvement:.2f}%")
        
        results[scenario['name']] = {
            'original': metrics_orig,
            'improved': metrics_imp,
            'variance_improvement': variance_improvement,
            'error_improvement': error_improvement
        }
    
    return results

if __name__ == "__main__":
    # 运行主要对比测试
    metrics_orig, metrics_imp = run_comparison_test()
    
    # 运行不同场景测试
    scenario_results = test_different_scenarios()
    
    print("\n=== 总结 ===")
    print("改进的粒子滤波主要通过以下方式解决lambda参数估计波动不明显的问题:")
    print("1. 自适应状态转移 - 根据有效粒子数量动态调整噪声")
    print("2. 创新序列更新 - 使用预测误差序列来改进权重更新")
    print("3. 系统重采样 - 使用更稳定的重采样策略")
    print("4. 重采样后抖动 - 增加粒子多样性")
    print("5. 不确定性估计 - 提供估计的置信区间")
    print("6. 收敛性分析 - 监控和评估估计质量") 