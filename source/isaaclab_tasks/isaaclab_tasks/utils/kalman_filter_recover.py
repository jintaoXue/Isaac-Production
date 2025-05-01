import numpy as np
import matplotlib.pyplot as plt

# 参数设置
np.random.seed(42)
dt = 0.1  # 时间间隔
num_steps = 100  # 时间步数
true_mu = 0.02  # 真实的 mu 值
R0 = 10.0  # 初始 R(t) 值

# 过程噪声和测量噪声协方差
Q = np.diag([0.01, 0.0001])  # 过程噪声协方差（对 R 和 mu）
R = np.array([[0.1]])  # 测量噪声协方差

# 生成模拟数据
times = np.arange(0, num_steps * dt, dt)
true_R = R0 * np.exp(-true_mu * times)  # 真实的 R(t)
measurements = true_R + np.random.normal(0, np.sqrt(R[0, 0]), size=true_R.shape)  # 带噪声的测量

# 状态转移函数
def state_transition(x, dt):
    R, mu = x
    R_new = R * np.exp(-mu * dt)
    mu_new = mu
    return np.array([R_new, mu_new])

# 雅可比矩阵
def jacobian(x, dt):
    R, mu = x
    return np.array([
        [np.exp(-mu * dt), -R * dt * np.exp(-mu * dt)],
        [0, 1]
    ])

# 测量矩阵
H = np.array([[1, 0]])

# 初始状态和协方差
x_hat = np.array([R0, 0.1])  # 初始估计 [R(0), mu]
P = np.diag([1.0, 1.0])  # 初始协方差

# 存储估计结果
R_estimates = []
mu_estimates = []

# 扩展卡尔曼滤波
for i in range(num_steps):
    # 预测步骤
    x_pred = state_transition(x_hat, dt)
    F = jacobian(x_hat, dt)
    P_pred = F @ P @ F.T + Q

    # 更新步骤
    z = np.array([measurements[i]])
    y = z - H @ x_pred  # 测量残差
    S = H @ P_pred @ H.T + R  # 残差协方差
    K = P_pred @ H.T @ np.linalg.inv(S)  # 卡尔曼增益
    x_hat = x_pred + K @ y  # 更新状态估计
    P = (np.eye(2) - K @ H) @ P_pred  # 更新协方差

    # 存储估计值
    R_estimates.append(x_hat[0])
    mu_estimates.append(x_hat[1])

# 可视化结果
plt.figure(figsize=(10, 6))

# 绘制 R(t)
plt.subplot(2, 1, 1)
plt.plot(times, true_R, label='True R(t)', color='blue')
plt.plot(times, measurements, 'x', label='Measurements', color='red', alpha=0.5)
plt.plot(times, R_estimates, label='Estimated R(t)', color='green')
plt.xlabel('Time')
plt.ylabel('R(t)')
plt.legend()
plt.grid(True)

# 绘制 mu
plt.subplot(2, 1, 2)
plt.plot(times, mu_estimates, label='Estimated mu', color='green')
plt.axhline(y=true_mu, color='blue', linestyle='--', label='True mu')
plt.xlabel('Time')
plt.ylabel('mu')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('ekf_results.png')













# import numpy as np

# # 定义卡尔曼滤波类
# class KalmanFilter:
#     def __init__(self, A, H, Q, R, x0, P0):
#         self.A = A  # 状态转移矩阵
#         self.H = H  # 观测矩阵
#         self.Q = Q  # 过程噪声协方差
#         self.R = R  # 测量噪声协方差
#         self.x = x0  # 初始状态估计
#         self.P = P0  # 初始估计误差协方差

#     def predict(self):
#         # 预测步骤
#         self.x = np.dot(self.A, self.x)
#         self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
#         return self.x

#     def update(self, z, H):
#         # 更新步骤
#         self.H = H
#         y = z - np.dot(self.H, self.x)
#         S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
#         K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
#         self.x = self.x + np.dot(K, y)
#         self.P = np.dot((np.eye(self.P.shape[0]) - np.dot(K, self.H)), self.P)
#         return self.x

# # 真实参数值
# true_A = 2.0
# true_B = 0.1

# # 时间点
# times = np.linspace(0, 10, 1000)

# # 生成测量值（添加噪声）
# measurements = []
# for t in times:
#     y = true_A * np.exp(true_B * t)
#     noise = np.random.normal(0, 0.1)
#     measurements.append(y + noise)

# # 初始化卡尔曼滤波器
# A = np.eye(2)  # 状态转移矩阵
# # Q = np.diag([0.00000001, 0.00000000001])  # 过程噪声协方差
# Q = np.diag([0.01, 0.01])  # 过程噪声协方差
# # R = 0.0  # 测量噪声协方差
# R = 0.1  # 测量噪声协方差
# x0 = np.array([1.0, 0.05])  # 初始状态估计
# P0 = np.diag([1.0, 1.0])  # 初始估计误差协方差

# kf = KalmanFilter(A, None, Q, R, x0, P0)

# # 进行卡尔曼滤波迭代
# estimated_params = []
# for t, z in zip(times, measurements):
#     # 构建观测矩阵
#     H = np.array([[np.exp(t * kf.x[1]), kf.x[0] * t * np.exp(t * kf.x[1])]])
#     kf.predict()
#     estimated_x = kf.update(z, H)
#     estimated_params.append(estimated_x)

# # 输出估计结果
# estimated_A = [param[0] for param in estimated_params]
# estimated_B = [param[1] for param in estimated_params]

# print("Final estimated A:", estimated_A[-1])
# print("Final estimated B:", estimated_B[-1])