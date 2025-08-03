import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

class ParticleFilter:
    def __init__(self, dt, num_steps, true_lambda, F0, num_particles, sigma_w, sigma_v, 
                 lamda_init, upper_bound, lower_bound, resample_method='systematic'):
        self.dt = dt  # 时间间隔
        self.num_steps = num_steps  # 时间步数
        self.true_lambda = true_lambda  # 真实的 lambda 值
        self.F0 = F0  # 初始 F(t) 值
        self.num_particles = num_particles  # 粒子数量
        self.sigma_w = sigma_w  # 过程噪声标准差
        self.sigma_v = sigma_v  # 测量噪声标准差
        self.resample_method = resample_method  # 重采样方法

        # 收敛改进参数
        self.adaptive_noise = True  # 自适应噪声
        self.regularization = True  # 正则化
        self.multi_scale = True  # 多尺度采样
        self.convergence_threshold = 0.01  # 收敛阈值
        self.min_ess_threshold = 0.3  # 最小有效样本大小阈值
        
        # 自适应参数
        self.adaptive_sigma_w = sigma_w
        self.adaptive_sigma_v = sigma_v
        self.convergence_history = []
        self.ess_history = []
        
        # 初始化粒子
        self.l_bound = lower_bound
        self.u_bound = upper_bound
        self.particles = np.random.uniform(lower_bound, upper_bound, num_particles)
        self.weights = np.ones(num_particles) / num_particles
        
        # 修复初始化估计
        self.lambda_estimates = [lamda_init]  # 使用传入的初始值
        self.prev_time_step = -2
        self.F_estimates = [F0]
        self.measurements = []
        self.true_F = []
        self.times = []
        
        # 多尺度参数
        self.scale_factors = [0.5, 1.0, 2.0]  # 多尺度因子
        self.scale_weights = [0.2, 0.6, 0.2]  # 各尺度权重
    
    def reinit(self, time_step, F0):
        self.F_estimates.append(F0)
        if len(self.lambda_estimates) > 0:
            self.lambda_estimates.append(self.lambda_estimates[-1])
        self.prev_time_step = time_step
        self.F0 = F0

    def error_data_engineering(self, measurement_t, F_pred):
        """
        误差数据工程：对误差值进行预处理和特征工程
        
        参数:
        - measurement_t: 测量值
        - F_pred: 预测值
        
        返回:
        - processed_error: 处理后的误差
        - error_features: 误差特征字典
        """
        # 1. 基础误差计算
        raw_error = measurement_t - F_pred
        abs_error = abs(raw_error)
        
        # 2. 归一化误差（相对于测量噪声）
        normalized_error = abs_error / (self.sigma_v + 1e-8)
        
        # 3. 误差分类
        if normalized_error < 0.1:
            error_category = 'very_small'
        elif normalized_error < 0.5:
            error_category = 'small'
        elif normalized_error < 1.0:
            error_category = 'medium'
        elif normalized_error < 2.0:
            error_category = 'large'
        else:
            error_category = 'very_large'
        
        # 4. 误差变换
        # 对数变换
        log_error = np.log(1 + abs_error)
        
        # 平方根变换
        sqrt_error = np.sqrt(abs_error)
        
        # 倒数变换（防止除零）
        inverse_error = 1.0 / (1.0 + abs_error)
        
        # 5. 误差统计特征
        error_features = {
            'raw_error': raw_error,
            'abs_error': abs_error,
            'normalized_error': normalized_error,
            'error_category': error_category,
            'log_error': log_error,
            'sqrt_error': sqrt_error,
            'inverse_error': inverse_error,
            'error_sign': np.sign(raw_error),
            'is_small_error': normalized_error < 0.5,
            'is_very_small_error': normalized_error < 0.1
        }
        
        return abs_error, error_features

    def adaptive_state_transition(self):
        """自适应状态转移 - 改进版本"""
        if self.adaptive_noise:
            # 基于有效样本大小调整噪声
            ess = self.effective_sample_size()
            ess_ratio = ess / self.num_particles
            
            if ess_ratio < 0.5:
                # 有效样本大小小时，增加噪声以增加多样性
                adaptive_noise = self.sigma_w * (2.0 - ess_ratio)  # 增加噪声幅度
            else:
                # 有效样本大小大时，减少噪声以提高精度
                adaptive_noise = self.sigma_w * (0.3 + ess_ratio * 0.7)
            
            self.adaptive_sigma_w = np.clip(adaptive_noise, self.sigma_w * 0.1, self.sigma_w * 5.0)
        
        # 改进的状态转移模型
        if self.multi_scale:
            new_particles = np.zeros_like(self.particles)
            for scale, weight in zip(self.scale_factors, self.scale_weights):
                # 添加随机游走成分
                random_walk = np.random.normal(0, self.adaptive_sigma_w * scale, self.num_particles)
                scale_particles = self.particles + random_walk
                new_particles += scale_particles * weight
            self.particles = new_particles
        else:
            # 添加随机游走
            self.particles = self.particles + np.random.normal(0, self.adaptive_sigma_w, self.num_particles)
        
        # 边界约束
        self.particles = np.clip(self.particles, self.l_bound, self.u_bound)

    def inverse_error_likelihood(self, measurement_t, F_pred, alpha=2.0, beta=1.0):
        """
        反比误差likelihood函数：误差越小，概率越大，且斜率越大
        
        参数:
        - measurement_t: 测量值
        - F_pred: 预测值
        - alpha: 控制函数陡峭程度
        - beta: 控制函数形状
        """
        error = abs(measurement_t - F_pred)
        
        # 使用反比函数，误差越小概率越大
        # 添加小常数防止除零
        likelihood = 1.0 / (1.0 + alpha * (error ** beta))
        
        return likelihood

    def exponential_sensitive_likelihood(self, measurement_t, F_pred, sensitivity=3.0):
        """
        指数敏感likelihood函数：误差越小，斜率越大
        
        参数:
        - measurement_t: 测量值
        - F_pred: 预测值
        - sensitivity: 敏感度参数，越大对小误差越敏感
        """
        error = abs(measurement_t - F_pred)
        
        # 使用指数函数，误差越小斜率越大
        likelihood = np.exp(-sensitivity * error)
        
        return likelihood

    def power_sensitive_likelihood(self, measurement_t, F_pred, power=0.3):
        """
        幂函数敏感likelihood：使用幂次小于1的函数，误差越小斜率越大
        
        参数:
        - measurement_t: 测量值
        - F_pred: 预测值
        - power: 幂次，小于1时误差越小斜率越大
        """
        error = abs(measurement_t - F_pred)
        
        # 使用幂函数，当power < 1时，误差越小斜率越大
        likelihood = np.exp(-error ** power)
        
        return likelihood

    def logarithmic_sensitive_likelihood(self, measurement_t, F_pred, scale=2.0):
        """
        对数敏感likelihood函数：使用对数函数，误差越小斜率越大
        
        参数:
        - measurement_t: 测量值
        - F_pred: 预测值
        - scale: 缩放因子
        """
        error = abs(measurement_t - F_pred)
        
        # 使用对数函数，误差越小斜率越大
        likelihood = 1.0 / (1.0 + scale * np.log(1 + error))
        
        return likelihood

    def hyperbolic_sensitive_likelihood(self, measurement_t, F_pred, k=2.0):
        """
        双曲线敏感likelihood函数：误差越小斜率越大
        
        参数:
        - measurement_t: 测量值
        - F_pred: 预测值
        - k: 控制函数陡峭程度
        """
        error = abs(measurement_t - F_pred)
        
        # 使用双曲线函数，误差越小斜率越大
        likelihood = 1.0 / (1.0 + k * error)
        
        return likelihood

    def adaptive_sensitive_likelihood(self, measurement_t, F_pred):
        """
        自适应敏感likelihood函数：根据误差大小动态调整敏感度
        
        参数:
        - measurement_t: 测量值
        - F_pred: 预测值
        """
        error = abs(measurement_t - F_pred)
        normalized_error = error / (self.sigma_v + 1e-8)
        
        if normalized_error < 0.1:
            # 极小误差：使用高敏感度指数函数
            likelihood = np.exp(-5.0 * normalized_error)
        elif normalized_error < 0.5:
            # 小误差：使用中等敏感度幂函数
            likelihood = np.exp(-2.0 * (normalized_error ** 0.5))
        elif normalized_error < 1.0:
            # 中等误差：使用标准指数函数
            likelihood = np.exp(-normalized_error)
        else:
            # 大误差：使用平缓函数
            likelihood = np.exp(-0.5 * normalized_error)
        
        return likelihood

    def optimized_likelihood_calculation(self, F_prev, measurement_t, method='adaptive_sensitive'):
        """优化的likelihood计算方法，支持多种敏感函数"""
        # 改进的预测模型
        F_pred = F_prev + (1 - F_prev) * (1 - np.exp(-self.particles * self.dt))
        
        # 根据选择的方法计算likelihood
        if method == 'inverse_error':
            likelihood = np.array([self.inverse_error_likelihood(measurement_t, pred) 
                                 for pred in F_pred])
        elif method == 'exponential_sensitive':
            likelihood = np.array([self.exponential_sensitive_likelihood(measurement_t, pred) 
                                 for pred in F_pred])
        elif method == 'power_sensitive':
            likelihood = np.array([self.power_sensitive_likelihood(measurement_t, pred) 
                                 for pred in F_pred])
        elif method == 'logarithmic_sensitive':
            likelihood = np.array([self.logarithmic_sensitive_likelihood(measurement_t, pred) 
                                 for pred in F_pred])
        elif method == 'hyperbolic_sensitive':
            likelihood = np.array([self.hyperbolic_sensitive_likelihood(measurement_t, pred) 
                                 for pred in F_pred])
        elif method == 'adaptive_sensitive':
            likelihood = np.array([self.adaptive_sensitive_likelihood(measurement_t, pred) 
                                 for pred in F_pred])
        else:
            # 标准高斯likelihood
            squared_diff = (measurement_t - F_pred)**2
            likelihood = np.exp(-0.5 * squared_diff / (self.sigma_v**2 + 1e-8))
        
        # 正则化处理
        if self.regularization:
            regularization_factor = 1e-8
            likelihood = likelihood + regularization_factor
        
        return likelihood

    def robust_weight_update(self, F_prev, measurement_t, likelihood_method='adaptive_sensitive'):
        """鲁棒的权重更新方法"""
        # 使用优化的likelihood计算
        likelihood = self.optimized_likelihood_calculation(F_prev, measurement_t, method=likelihood_method)
        
        # 更新权重
        self.weights = self.weights * likelihood
        
        # 防止权重退化
        min_weight = 1e-10
        self.weights = np.maximum(self.weights, min_weight)
        
        # 归一化权重
        weight_sum = np.sum(self.weights)
        if weight_sum > 0:
            self.weights = self.weights / weight_sum
        else:
            # 如果所有权重都为零，重新初始化
            self.weights = np.ones(self.num_particles) / self.num_particles

    def regularized_update_weights(self, F_prev, measurement_t):
        """正则化权重更新 - 使用优化的likelihood计算"""
        return self.robust_weight_update(F_prev, measurement_t, likelihood_method='adaptive_sensitive')

    def effective_sample_size(self):
        """计算有效样本大小"""
        return 1.0 / np.sum(self.weights**2)

    def systematic_resample(self):
        """系统重采样"""
        cumsum_weights = np.cumsum(self.weights)
        u = np.random.uniform(0, 1.0 / self.num_particles)
        
        indices = np.zeros(self.num_particles, dtype=int)
        j = 0
        for i in range(self.num_particles):
            u_i = u + i / self.num_particles
            while u_i > cumsum_weights[j]:
                j += 1
            indices[i] = j
        
        return indices

    def stratified_resample(self):
        """分层重采样"""
        cumsum_weights = np.cumsum(self.weights)
        u = np.random.uniform(0, 1.0 / self.num_particles, self.num_particles)
        u += np.arange(self.num_particles) / self.num_particles
        
        indices = np.zeros(self.num_particles, dtype=int)
        j = 0
        for i in range(self.num_particles):
            while u[i] > cumsum_weights[j]:
                j += 1
            indices[i] = j
        
        return indices

    def residual_resample(self):
        """残差重采样"""
        expected_counts = self.weights * self.num_particles
        deterministic_counts = np.floor(expected_counts).astype(int)
        residual_weights = expected_counts - deterministic_counts
        residual_weights = residual_weights / np.sum(residual_weights)
        
        remaining_particles = self.num_particles - np.sum(deterministic_counts)
        random_indices = np.random.choice(self.num_particles, remaining_particles, p=residual_weights)
        
        indices = []
        for i in range(self.num_particles):
            indices.extend([i] * deterministic_counts[i])
        indices.extend(random_indices)
        
        return np.array(indices)

    def improved_resample(self):
        """改进的重采样方法"""
        N_eff = self.effective_sample_size()
        self.ess_history.append(N_eff)
        
        # 自适应重采样阈值
        adaptive_threshold = self.num_particles * self.min_ess_threshold
        
        if N_eff < adaptive_threshold:
            if self.resample_method == 'systematic':
                indices = self.systematic_resample()
            elif self.resample_method == 'stratified':
                indices = self.stratified_resample()
            elif self.resample_method == 'residual':
                indices = self.residual_resample()
            else:
                indices = self.systematic_resample()
            
            # 重采样粒子和权重
            self.particles = self.particles[indices]
            self.weights = np.ones(self.num_particles) / self.num_particles
            
            # 智能扰动：基于粒子分布进行扰动
            particle_std = np.std(self.particles)
            if particle_std > 0:
                # 扰动幅度与粒子分布标准差相关
                perturbation_scale = min(0.2, particle_std * 0.2)  # 增加扰动
                self.particles += np.random.normal(0, perturbation_scale, self.num_particles)
                self.particles = np.clip(self.particles, self.l_bound, self.u_bound)

    def check_convergence(self):
        """检查收敛性"""
        if len(self.lambda_estimates) < 5:
            return False
        
        # 计算最近几个估计值的方差
        recent_estimates = self.lambda_estimates[-5:]
        variance = np.var(recent_estimates)
        
        # 计算收敛指标
        convergence_metric = variance / (self.true_lambda ** 2 + 1e-8)
        self.convergence_history.append(convergence_metric)
        
        return convergence_metric < self.convergence_threshold

    def step(self, measurement, true_F, time_step):
        self.measurements.append(measurement)
        self.true_F.append(true_F)
        self.times.append(time_step)
        
        if time_step != self.prev_time_step + 1:
            self.reinit(time_step, true_F)
            return
        
        # 自适应状态转移
        self.adaptive_state_transition()
        
        # 使用优化的权重更新
        F_prev = self.F_estimates[-1]
        self.robust_weight_update(F_prev, measurement, likelihood_method='adaptive_sensitive')
        
        # 改进的重采样
        self.improved_resample()
        
        # 估计 lambda
        lambda_est = np.sum(self.particles * self.weights)
        self.lambda_estimates.append(lambda_est)
        
        # 估计 F(t) 用于下一时刻
        F_est = measurement + (1 - measurement) * (1 - np.exp(-lambda_est * self.dt))
        self.F_estimates.append(F_est)
        
        # 检查收敛性
        if self.check_convergence():
            print(f"在时间步 {time_step} 达到收敛")

    def run(self):
        F_estimates = [self.F0]
        lambda_estimates = [self.lambda_estimates[0]]  # 使用初始估计
        
        # 生成模拟数据
        self.times = np.arange(0, self.num_steps * self.dt, self.dt)
        self.true_F = 1 - (1 - self.F0) * np.exp(-self.true_lambda * self.times)
        self.measurements = self.true_F + np.random.normal(0, self.sigma_v, size=self.true_F.shape)
        
        convergence_reached = False
        
        for t in range(1, self.num_steps):
            # 自适应状态转移
            self.adaptive_state_transition()
            
            # 使用优化的权重更新
            F_prev = F_estimates[-1]
            self.robust_weight_update(F_prev, self.measurements[t], likelihood_method='adaptive_sensitive')
            
            # 改进的重采样
            self.improved_resample()
            
            # 估计 lambda
            lambda_est = np.sum(self.particles * self.weights)
            lambda_estimates.append(lambda_est)
            
            # 估计 F(t) 用于下一时刻
            F_est = F_prev + (1 - F_prev) * (1 - np.exp(-lambda_est * self.dt))
            F_estimates.append(F_est)
            
            # 检查收敛性
            if not convergence_reached and self.check_convergence():
                print(f"在时间步 {t} 达到收敛")
                convergence_reached = True
        
        return self.times, F_estimates, lambda_estimates

    def plot_convergence_analysis(self, times, F_estimates, lambda_estimates, name=''):
        """绘制收敛性分析"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # F(t) 估计
        axes[0, 0].plot(times, self.measurements, 'x', label='Measurements', color='red', alpha=0.5)
        axes[0, 0].plot(times, F_estimates, label='Estimated F(t)', color='green')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('F(t)')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        axes[0, 0].set_title('F(t) Estimation')
        
        # Lambda 估计
        axes[0, 1].plot(times, lambda_estimates[1:], label='Estimated lambda', color='green')
        axes[0, 1].axhline(y=self.true_lambda, color='blue', linestyle='--', label='True lambda')
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('lambda')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        axes[0, 1].set_title('Lambda Estimation')
        
        # 有效样本大小
        if len(self.ess_history) > 0:
            ess_times = range(len(self.ess_history))
            axes[1, 0].plot(ess_times, self.ess_history, 'b-', linewidth=2)
            axes[1, 0].axhline(y=self.num_particles * self.min_ess_threshold, color='r', linestyle='--', 
                               label='Resampling threshold')
            axes[1, 0].set_xlabel('Time Step')
            axes[1, 0].set_ylabel('Effective Sample Size')
            axes[1, 0].set_title('Effective Sample Size Over Time')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # 收敛指标
        if len(self.convergence_history) > 0:
            conv_times = range(len(self.convergence_history))
            axes[1, 1].plot(conv_times, self.convergence_history, 'g-', linewidth=2)
            axes[1, 1].axhline(y=self.convergence_threshold, color='r', linestyle='--', 
                               label='Convergence threshold')
            axes[1, 1].set_xlabel('Time Step')
            axes[1, 1].set_ylabel('Convergence Metric')
            axes[1, 1].set_title('Convergence Analysis')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(name + '_convergence_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_results(self, times, F_estimates, lambda_estimates, name=''):
        """绘制结果"""
        plt.figure(figsize=(10, 6))
        
        # 绘制 F(t)
        plt.subplot(2, 1, 1)
        plt.plot(times, self.measurements, 'x', label='Measurements', color='red', alpha=0.5)
        plt.plot(times, F_estimates, label='Estimated F(t)', color='green')
        plt.xlabel('Time')
        plt.ylabel('F(t)')
        plt.legend()
        plt.grid(True)
        
        # 绘制 lambda
        plt.subplot(2, 1, 2)
        plt.plot(times, lambda_estimates[1:], label='Estimated lambda', color='green')
        plt.axhline(y=self.true_lambda, color='blue', linestyle='--', label='True lambda')
        plt.xlabel('Time')
        plt.ylabel('lambda')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(name + '_improved_pf_results.png')

    def plot_likelihood_comparison(self, measurement_t=0.5, F_pred_range=np.linspace(0.1, 0.9, 100)):
        """绘制不同likelihood函数的比较"""
        plt.figure(figsize=(12, 8))
        
        error_range = np.abs(measurement_t - F_pred_range)
        
        # 标准高斯函数
        standard_likelihood = np.exp(-0.5 * error_range**2)
        plt.plot(error_range, standard_likelihood, 'b-', label='Standard Gaussian', linewidth=2)
        
        # 反比误差函数
        inverse_likelihood = 1.0 / (1.0 + 2.0 * error_range)
        plt.plot(error_range, inverse_likelihood, 'r-', label='Inverse Error', linewidth=2)
        
        # 指数敏感函数
        exp_sensitive_likelihood = np.exp(-3.0 * error_range)
        plt.plot(error_range, exp_sensitive_likelihood, 'g-', label='Exponential Sensitive', linewidth=2)
        
        # 幂函数敏感
        power_sensitive_likelihood = np.exp(-error_range**0.3)
        plt.plot(error_range, power_sensitive_likelihood, 'm-', label='Power Sensitive', linewidth=2)
        
        # 自适应敏感函数
        adaptive_likelihood = []
        for error in error_range:
            normalized_error = error / 0.01  # 假设sigma_v=0.01
            if normalized_error < 0.1:
                likelihood = np.exp(-5.0 * normalized_error)
            elif normalized_error < 0.5:
                likelihood = np.exp(-2.0 * (normalized_error ** 0.5))
            elif normalized_error < 1.0:
                likelihood = np.exp(-normalized_error)
            else:
                likelihood = np.exp(-0.5 * normalized_error)
            adaptive_likelihood.append(likelihood)
        
        plt.plot(error_range, adaptive_likelihood, 'c-', label='Adaptive Sensitive', linewidth=2)
        
        plt.xlabel('Error')
        plt.ylabel('Likelihood')
        plt.title('Comparison of Different Likelihood Functions')
        plt.legend()
        plt.grid(True)
        plt.xlim(0, 0.5)
        plt.ylim(0, 1)
        plt.savefig('likelihood_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

class RecParticleFilter(ParticleFilter):
    def optimized_likelihood_calculation(self, F_prev, measurement_t, method='adaptive_sensitive'):
        """递归粒子滤波器的优化likelihood计算"""
        # 递归模型的预测
        F_pred = F_prev * np.exp(-self.particles * self.dt)
        
        # 根据选择的方法计算likelihood
        if method == 'inverse_error':
            likelihood = np.array([self.inverse_error_likelihood(measurement_t, pred) 
                                 for pred in F_pred])
        elif method == 'exponential_sensitive':
            likelihood = np.array([self.exponential_sensitive_likelihood(measurement_t, pred) 
                                 for pred in F_pred])
        elif method == 'power_sensitive':
            likelihood = np.array([self.power_sensitive_likelihood(measurement_t, pred) 
                                 for pred in F_pred])
        elif method == 'logarithmic_sensitive':
            likelihood = np.array([self.logarithmic_sensitive_likelihood(measurement_t, pred) 
                                 for pred in F_pred])
        elif method == 'hyperbolic_sensitive':
            likelihood = np.array([self.hyperbolic_sensitive_likelihood(measurement_t, pred) 
                                 for pred in F_pred])
        elif method == 'adaptive_sensitive':
            likelihood = np.array([self.adaptive_sensitive_likelihood(measurement_t, pred) 
                                 for pred in F_pred])
        else:
            # 标准高斯likelihood
            squared_diff = (measurement_t - F_pred)**2
            likelihood = np.exp(-0.5 * squared_diff / (self.sigma_v**2 + 1e-8))
        
        # 正则化处理
        if self.regularization:
            regularization_factor = 1e-8
            likelihood = likelihood + regularization_factor
        
        return likelihood

    def regularized_update_weights(self, F_prev, measurement_t):
        """递归粒子滤波器的正则化权重更新"""
        return self.robust_weight_update(F_prev, measurement_t, likelihood_method='adaptive_sensitive')

# 使用示例
if __name__ == "__main__":
    # 参数设置 - 增加噪声参数
    dt = 0.1
    num_steps = 100
    true_lambda = 0.5
    F0 = 0.0
    num_particles = 1000
    sigma_w = 0.01  # 增加过程噪声
    sigma_v = 0.01  # 增加测量噪声
    
    # 创建改进的粒子滤波器
    pf = ParticleFilter(dt, num_steps, true_lambda, F0, num_particles, sigma_w, sigma_v,
                       lamda_init=0.3, upper_bound=2.0, lower_bound=0.0,  # 使用不同的初始值
                       resample_method='systematic')
    
    # 绘制likelihood函数比较
    pf.plot_likelihood_comparison()
    
    # 运行改进的粒子滤波
    times, F_estimates, lambda_estimates = pf.run()
    
    # 绘制结果和收敛性分析
    pf.plot_results(times, F_estimates, lambda_estimates, 'improved')
    pf.plot_convergence_analysis(times, F_estimates, lambda_estimates, 'improved') 