import numpy as np
import matplotlib.pyplot as plt

class ParticleFilter:
    def __init__(self, dt, num_steps, true_lambda, F0, num_particles, sigma_w, sigma_v, lamda_init, upper_bound, lower_bound):
        self.dt = dt  # 时间间隔
        self.num_steps = num_steps  # 时间步数
        self.true_lambda = true_lambda  # 真实的 lambda 值
        self.F0 = F0  # 初始 F(t) 值
        self.num_particles = num_particles  # 粒子数量
        self.sigma_w = sigma_w  # 过程噪声标准差
        self.sigma_v = sigma_v  # 测量噪声标准差

        # 初始化粒子
        self.l_bound = lower_bound
        self.u_bound = upper_bound
        self.particles = np.random.uniform(lower_bound, upper_bound, num_particles)  # 初始粒子分布
        self.weights = np.ones(num_particles) / num_particles  # 初始权重

        # 添加自适应参数
        self.adaptive_noise = True  # 启用自适应噪声
        self.min_effective_particles = 0.1  # 最小有效粒子比例
        self.innovation_threshold = 0.01  # 创新阈值
        
        self.prev_time_step = -2
        self.F_estimates = []
        self.lambda_estimates = [np.sum(self.particles * self.weights)]
        self.measurements = []
        self.true_F = []
        self.times = []
        self.innovation_history = []  # 记录创新序列
        self.effective_particle_ratios = []  # 记录有效粒子比例
    
    def reinit(self, time_step, F0):
        self.F_estimates.append(F0)
        if len(self.lambda_estimates) > 0:
            self.lambda_estimates.append(self.lambda_estimates[-1])
        self.prev_time_step = time_step
        self.F0 = F0

    def adaptive_state_transition(self):
        """自适应状态转移，根据有效粒子数量调整噪声"""
        # 计算有效粒子数量
        N_eff = 1 / np.sum(self.weights**2)
        effective_ratio = N_eff / self.num_particles
        
        # 根据有效粒子比例调整噪声
        if effective_ratio < self.min_effective_particles:
            # 有效粒子太少，增加噪声以增加多样性
            adaptive_sigma = self.sigma_w * (1 + 2 * (self.min_effective_particles - effective_ratio))
        else:
            # 有效粒子充足，使用标准噪声
            adaptive_sigma = self.sigma_w
            
        # 添加额外的扰动以增加多样性
        diversity_noise = np.random.normal(0, adaptive_sigma * 0.1, self.num_particles)
        
        self.particles = self.particles + np.random.normal(0, adaptive_sigma, self.num_particles) + diversity_noise
        self.particles = np.clip(self.particles, self.l_bound, self.u_bound)

    def update_weights_with_innovation(self, F_prev, measurement_t):
        """使用创新序列更新权重"""
        F_pred = F_prev + (1 - F_prev) * (1 - np.exp(-self.particles * self.dt))
        
        # 计算创新（预测误差）
        innovation = measurement_t - np.mean(F_pred)
        self.innovation_history.append(innovation)
        
        # 使用自适应似然函数
        if len(self.innovation_history) > 5:
            # 使用创新序列的方差来调整似然函数
            innovation_var = np.var(self.innovation_history[-5:])
            adaptive_sigma_v = max(self.sigma_v, innovation_var * 0.5)
        else:
            adaptive_sigma_v = self.sigma_v
            
        # 计算似然
        likelihood = np.exp(-0.5 * (measurement_t - F_pred)**2 / adaptive_sigma_v**2)
        
        # 添加正则化项以防止权重退化
        regularization = 1e-6
        likelihood = likelihood + regularization
        
        self.weights = self.weights * likelihood
        self.weights = self.weights / np.sum(self.weights)  # 归一化权重

    def improved_resample(self):
        """改进的重采样策略"""
        N_eff = 1 / np.sum(self.weights**2)
        self.effective_particle_ratios.append(N_eff / self.num_particles)
        
        if N_eff < self.num_particles / 2:
            # 使用系统重采样而不是随机重采样
            indices = self.systematic_resample()
            self.particles = self.particles[indices]
            self.weights = np.ones(self.num_particles) / self.num_particles
            
            # 添加重采样后的抖动
            jitter = np.random.normal(0, self.sigma_w * 0.1, self.num_particles)
            self.particles = self.particles + jitter
            self.particles = np.clip(self.particles, self.l_bound, self.u_bound)

    def systematic_resample(self):
        """系统重采样，比随机重采样更稳定"""
        N = self.num_particles
        positions = (np.arange(N) + np.random.uniform(0, 1)) / N
        cumulative_sum = np.cumsum(self.weights)
        cumulative_sum[-1] = 1.0  # 确保最后一个元素为1
        
        indices = np.zeros(N, dtype=int)
        i, j = 0, 0
        while i < N:
            if positions[i] < cumulative_sum[j]:
                indices[i] = j
                i += 1
            else:
                j += 1
        return indices

    def estimate_lambda_with_uncertainty(self):
        """估计lambda并计算不确定性"""
        lambda_est = np.sum(self.particles * self.weights)
        
        # 计算估计的不确定性
        lambda_var = np.sum(self.weights * (self.particles - lambda_est)**2)
        lambda_std = np.sqrt(lambda_var)
        
        return lambda_est, lambda_std

    def step(self, measurement, true_F, time_step):
        self.measurements.append(measurement)
        self.true_F.append(true_F)
        self.times.append(time_step)
        
        if time_step != self.prev_time_step + 1:
            self.reinit(time_step, true_F)
            return
        
        # 使用改进的状态转移
        self.adaptive_state_transition()
        
        # 更新权重
        F_prev = self.F_estimates[-1]
        self.update_weights_with_innovation(F_prev, measurement)

        # 改进的重采样
        self.improved_resample()

        # 估计 lambda 和不确定性
        lambda_est, lambda_std = self.estimate_lambda_with_uncertainty()
        self.lambda_estimates.append(lambda_est)

        # 估计 F(t) 用于下一时刻
        F_est = F_prev + (1 - F_prev) * (1 - np.exp(-lambda_est * self.dt))
        self.F_estimates.append(F_est)

    def run(self):
        F_estimates = [self.F0]
        lambda_estimates = []
        lambda_uncertainties = []
        
        # 生成模拟数据
        self.times = np.arange(0, self.num_steps * self.dt, self.dt)
        self.true_F = 1 - (1 - self.F0) * np.exp(-self.true_lambda * self.times)
        self.measurements = self.true_F + np.random.normal(0, self.sigma_v, size=self.true_F.shape)

        for t in range(1, self.num_steps):
            # 预测步骤
            self.adaptive_state_transition()

            # 更新权重
            F_prev = F_estimates[-1]
            self.update_weights_with_innovation(F_prev, self.measurements[t])

            # 重采样
            self.improved_resample()

            # 估计 lambda 和不确定性
            lambda_est, lambda_std = self.estimate_lambda_with_uncertainty()
            lambda_estimates.append(lambda_est)
            lambda_uncertainties.append(lambda_std)

            # 估计 F(t) 用于下一时刻
            F_est = F_prev + (1 - F_prev) * (1 - np.exp(-lambda_est * self.dt))
            F_estimates.append(F_est)

        return self.times, F_estimates, lambda_estimates, lambda_uncertainties

    def plot_results(self, times, F_estimates, lambda_estimates, lambda_uncertainties=None, name=''):
        plt.figure(figsize=(15, 10))

        # 绘制 F(t)
        plt.subplot(3, 1, 1)
        plt.plot(times, self.measurements, 'x', label='Measurements', color='red', alpha=0.5)
        plt.plot(times, F_estimates, label='Estimated F(t)', color='green', linewidth=2)
        plt.xlabel('Time')
        plt.ylabel('F(t)')
        plt.legend()
        plt.grid(True)

        # 绘制 lambda 估计
        plt.subplot(3, 1, 2)
        # 确保维度匹配
        lambda_plot_times = times[:len(lambda_estimates[1:])]
        plt.plot(lambda_plot_times, lambda_estimates[1:], label='Estimated lambda', color='green', linewidth=2)
        plt.axhline(y=self.true_lambda, color='blue', linestyle='--', label='True lambda')
        
        # 如果有不确定性信息，绘制置信区间
        if lambda_uncertainties is not None:
            lambda_estimates_array = np.array(lambda_estimates[1:])
            lambda_uncertainties_array = np.array(lambda_uncertainties)
            # 确保不确定性数组长度匹配
            if len(lambda_uncertainties_array) == len(lambda_estimates_array):
                plt.fill_between(lambda_plot_times, 
                               lambda_estimates_array - 2*lambda_uncertainties_array,
                               lambda_estimates_array + 2*lambda_uncertainties_array,
                               alpha=0.3, color='green', label='95% Confidence Interval')
        
        plt.xlabel('Time')
        plt.ylabel('lambda')
        plt.legend()
        plt.grid(True)

        # 绘制有效粒子比例
        plt.subplot(3, 1, 3)
        if len(self.effective_particle_ratios) > 0:
            effective_times = times[1:len(self.effective_particle_ratios)+1]
            plt.plot(effective_times, self.effective_particle_ratios, label='Effective Particle Ratio', color='orange')
            plt.axhline(y=0.5, color='red', linestyle='--', label='Resampling Threshold')
        plt.xlabel('Time')
        plt.ylabel('Effective Particle Ratio')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(name + '_improved_pf_results.png', dpi=300, bbox_inches='tight')
        plt.show()

    def analyze_convergence(self):
        """分析收敛性"""
        if len(self.lambda_estimates) < 10:
            return "数据不足，无法分析收敛性"
        
        # 计算lambda估计的方差
        lambda_var = np.var(self.lambda_estimates[-20:])  # 最后20个估计的方差
        lambda_mean = np.mean(self.lambda_estimates[-20:])
        
        # 计算收敛指标
        convergence_metric = lambda_var / (lambda_mean**2 + 1e-6)
        
        print(f"Lambda估计均值: {lambda_mean:.4f}")
        print(f"Lambda估计方差: {lambda_var:.6f}")
        print(f"收敛指标: {convergence_metric:.6f}")
        print(f"真实lambda: {self.true_lambda:.4f}")
        print(f"相对误差: {abs(lambda_mean - self.true_lambda) / self.true_lambda * 100:.2f}%")
        
        return {
            'mean': lambda_mean,
            'variance': lambda_var,
            'convergence_metric': convergence_metric,
            'relative_error': abs(lambda_mean - self.true_lambda) / self.true_lambda * 100
        }


class RecParticleFilter(ParticleFilter):
    
    def update_weights_with_innovation(self, F_prev, measurement_t):

        """使用创新序列更新权重"""
        F_pred = F_prev*np.exp(-self.particles * self.dt)
        
        # 计算创新（预测误差）
        innovation = measurement_t - np.mean(F_pred)
        self.innovation_history.append(innovation)
        
        # 使用自适应似然函数
        if len(self.innovation_history) > 5:
            # 使用创新序列的方差来调整似然函数
            innovation_var = np.var(self.innovation_history[-5:])
            adaptive_sigma_v = max(self.sigma_v, innovation_var * 0.5)
        else:
            adaptive_sigma_v = self.sigma_v
            
        # 计算似然
        likelihood = np.exp(-0.5 * (measurement_t - F_pred)**2 / adaptive_sigma_v**2)
        
        # 添加正则化项以防止权重退化
        regularization = 1e-6
        likelihood = likelihood + regularization
        
        self.weights = self.weights * likelihood
        self.weights = self.weights / np.sum(self.weights)  # 归一化权重


# 使用示例和测试
if __name__ == "__main__":
    # 参数设置
    dt = 0.1
    num_steps = 200  # 增加时间步数
    true_lambda = 0.5
    F0 = 0.0
    num_particles = 2000  # 增加粒子数量
    sigma_w = 0.01  # 增加过程噪声
    sigma_v = 0.01  # 增加测量噪声
    lamda_init = 0.3
    upper_bound = 2.0
    lower_bound = 0.01

    # 创建改进的粒子滤波实例
    pf = ImprovedParticleFilter(dt, num_steps, true_lambda, F0, num_particles, 
                               sigma_w, sigma_v, lamda_init, upper_bound, lower_bound)

    # 运行粒子滤波
    times, F_estimates, lambda_estimates, lambda_uncertainties = pf.run()

    # 绘制结果
    pf.plot_results(times, F_estimates, lambda_estimates, lambda_uncertainties, 'improved')

    # 分析收敛性
    convergence_analysis = pf.analyze_convergence() 