import numpy as np
import matplotlib.pyplot as plt

class ParticleFilter:
    def __init__(self, dt, num_steps, true_lambda, F0, num_particles, sigma_w, sigma_v):
        self.dt = dt  # 时间间隔
        self.num_steps = num_steps  # 时间步数
        self.true_lambda = true_lambda  # 真实的 lambda 值
        self.F0 = F0  # 初始 F(t) 值
        self.num_particles = num_particles  # 粒子数量
        self.sigma_w = sigma_w  # 过程噪声标准差
        self.sigma_v = sigma_v  # 测量噪声标准差

        # 初始化粒子
        # np.random.seed(42)
        self.particles = np.random.uniform(0.1, 1.0, num_particles)  # 初始粒子分布
        self.weights = np.ones(num_particles) / num_particles  # 初始权重

        self.prev_time_step = -2
        self.F_estimates = []
        self.lambda_estimates = []
        self.measurements = []
        self.true_F = []
        self.times = []
    
    def reinit(self, time_step, F0, _lambda):
        self.prev_time_step = time_step
        self.F0 = F0
        self.x_hat = np.array([F0, _lambda])

    def state_transition(self):
        self.particles = self.particles + np.random.normal(0, self.sigma_w, self.num_particles)

    def update_weights(self, F_prev, measurement_t):
        F_pred = F_prev + (1 - F_prev) * (1 - np.exp(-self.particles * self.dt))
        likelihood = np.exp(-0.5 * (measurement_t - F_pred)**2 / self.sigma_v**2)
        self.weights = self.weights * likelihood
        self.weights = self.weights / np.sum(self.weights)  # 归一化权重

    def resample(self):
        N_eff = 1 / np.sum(self.weights**2)
        if N_eff < self.num_particles / 2:
            indices = np.random.choice(self.num_particles, self.num_particles, p=self.weights)
            self.particles = self.particles[indices]
            self.weights = np.ones(self.num_particles) / self.num_particles

    def step(self, measurement, true_F, time_step):

        self.state_transition()

        # 更新权重
        F_prev = self.F_estimates[-1]
        self.update_weights(F_prev, measurement)
        self.measurements.append(measurement)
        self.true_F.append(true_F)
        self.times.append(time_step)

        # 重采样
        self.resample()

        # 估计 lambda
        lambda_est = np.sum(self.particles * self.weights)
        self.lambda_estimates.append(lambda_est)

        # 估计 F(t) 用于下一时刻
        F_est = F_prev + (1 - F_prev) * (1 - np.exp(-lambda_est * self.dt))
        self.F_estimates.append(F_est)

    def run(self):
        F_estimates = [self.F0]  # F(t) 的估计值列表，初始值为 F0
        lambda_estimates = []
        # 生成模拟数据
        self.times = np.arange(0, num_steps * dt, dt)
        self.true_F = 1 - (1 - self.F0) * np.exp(-true_lambda * self.times)  # 真实的 F(t)
        self.measurements = self.true_F + np.random.normal(0, sigma_v, size=self.true_F.shape)  # 带噪声的测量

        for t in range(1, self.num_steps):
            # 预测步骤
            self.state_transition()

            # 更新权重
            F_prev = F_estimates[-1]
            self.update_weights(F_prev, self.measurements[t])

            # 重采样
            self.resample()

            # 估计 lambda
            lambda_est = np.sum(self.particles * self.weights)
            lambda_estimates.append(lambda_est)

            # 估计 F(t) 用于下一时刻
            F_est = F_prev + (1 - F_prev) * (1 - np.exp(-lambda_est * self.dt))
            F_estimates.append(F_est)

        return self.times, F_estimates, lambda_estimates

    def plot_results(self, times, F_estimates, lambda_estimates):
        plt.figure(figsize=(10, 6))

        # 绘制 F(t)
        plt.subplot(2, 1, 1)
        plt.plot(times, self.true_F, label='True F(t)', color='blue')
        plt.plot(times, self.measurements, 'x', label='Measurements', color='red', alpha=0.5)
        plt.plot(times, F_estimates, label='Estimated F(t)', color='green')
        plt.xlabel('Time')
        plt.ylabel('F(t)')
        plt.legend()
        plt.grid(True)

        # 绘制 lambda
        plt.subplot(2, 1, 2)
        plt.plot(times[1:], lambda_estimates, label='Estimated lambda', color='green')
        plt.axhline(y=self.true_lambda, color='blue', linestyle='--', label='True lambda')
        plt.xlabel('Time')
        plt.ylabel('lambda')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig('pf_class_results.png')

# 使用示例
if __name__ == "__main__":
    # 参数设置
    dt = 0.1
    num_steps = 100
    true_lambda = 0.5
    F0 = 0.0
    num_particles = 1000
    sigma_w = 0.01
    sigma_v = 0.1

    # 创建粒子滤波实例
    pf = ParticleFilter(dt, num_steps, true_lambda, F0, num_particles, sigma_w, sigma_v)

    # 运行粒子滤波
    times, F_estimates, lambda_estimates = pf.run()

    # 绘制结果
    pf.plot_results(times, F_estimates, lambda_estimates)