import numpy as np
import matplotlib.pyplot as plt

# 参数设置
np.random.seed(42)
dt = 0.1  # 时间间隔
num_steps = 100  # 时间步数
true_lambda = 0.5  # 真实的 lambda 值
F0 = 0.0  # 初始 F(t) 值

# 过程噪声和测量噪声协方差
Q = np.diag([0.01, 0.0001])  # 过程噪声协方差（对 F 和 lambda）
R = np.array([[0.1]])  # 测量噪声协方差

# 生成模拟数据
times = np.arange(0, num_steps * dt, dt)
true_F = 1 - (1 - F0) * np.exp(-true_lambda * times)  # 真实的 F(t)
measurements = true_F + np.random.normal(0, np.sqrt(R[0, 0]), size=true_F.shape)  # 带噪声的测量

# 状态转移函数
def state_transition(x, dt):
    F, lam = x
    F_new = F + (1 - F) * (1 - np.exp(-lam * dt))
    lam_new = lam
    return np.array([F_new, lam_new])

# 雅可比矩阵
def jacobian(x, dt):
    F, lam = x
    return np.array([
        [np.exp(-lam * dt), (1 - F) * (-dt * np.exp(-lam * dt))],
        [0, 1]
    ])

# 测量矩阵
H = np.array([[1, 0]])

# 初始状态和协方差
x_hat = np.array([F0, 0.6])  # 初始估计 [F(0), lambda]
P = np.diag([1.0, 1.0])  # 初始协方差

# 存储估计结果
F_estimates = []
lambda_estimates = []

# 扩展卡尔曼滤波
for i in range(num_steps):
    # 预测步骤
    x_pred = state_transition(x_hat, dt)
    F_jac = jacobian(x_hat, dt)
    P_pred = F_jac @ P @ F_jac.T + Q

    # 更新步骤
    z = np.array([measurements[i]])
    y = z - H @ x_pred  # 测量残差
    S = H @ P_pred @ H.T + R  # 残差协方差
    K = P_pred @ H.T @ np.linalg.inv(S)  # 卡尔曼增益
    x_hat = x_pred + K @ y  # 更新状态估计
    P = (np.eye(2) - K @ H) @ P_pred  # 更新协方差

    # 存储估计值
    F_estimates.append(x_hat[0])
    lambda_estimates.append(x_hat[1])

# 可视化结果
plt.figure(figsize=(10, 6))

# 绘制 F(t)
plt.subplot(2, 1, 1)
plt.plot(times, true_F, label='True F(t)', color='blue')
plt.plot(times, measurements, 'x', label='Measurements', color='red', alpha=0.5)
plt.plot(times, F_estimates, label='Estimated F(t)', color='green')
plt.xlabel('Time')
plt.ylabel('F(t)')
plt.legend()
plt.grid(True)

# 绘制 lambda
plt.subplot(2, 1, 2)
plt.plot(times, lambda_estimates, label='Estimated lambda', color='green')
plt.axhline(y=true_lambda, color='blue', linestyle='--', label='True lambda')
plt.xlabel('Time')
plt.ylabel('lambda')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('ekf_lambda_results.png')