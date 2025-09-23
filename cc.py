import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. 用pandas读取train.csv中的数据
df = pd.read_csv('train.csv')

# 检查数据
print("数据前5行：")
print(df.head())
print(f"\n数据形状：{df.shape}")

# 提取x和y数据
x = df['x'].values
y = df['y'].values


# 2. 训练线性回归模型 y = wx + b
def linear_regression(x, y, learning_rate=0.0001, epochs=1000):
    # 初始化参数
    w = 0.0
    b = 0.0
    n = len(x)

    # 存储训练过程中的参数和损失
    w_history = []
    b_history = []
    loss_history = []

    # 梯度下降
    for epoch in range(epochs):
        # 前向传播
        y_pred = w * x + b

        # 计算损失（均方误差）
        loss = np.mean((y_pred - y) ** 2)

        # 计算梯度
        dw = (2 / n) * np.sum((y_pred - y) * x)
        db = (2 / n) * np.sum(y_pred - y)

        # 更新参数
        w = w - learning_rate * dw
        b = b - learning_rate * db

        # 记录历史值
        if epoch % 10 == 0:  # 每10个epoch记录一次
            w_history.append(w)
            b_history.append(b)
            loss_history.append(loss)

    return w, b, w_history, b_history, loss_history


# 训练模型
w_final, b_final, w_history, b_history, loss_history = linear_regression(x, y, learning_rate=0.0001, epochs=10000)

print(f"\n训练完成！")
print(f"最终参数：w = {w_final:.4f}, b = {b_final:.4f}")

# 3. 用matplotlib绘制w和loss之间的关系、b和loss之间的关系
plt.figure(figsize=(15, 5))

# 子图1：w和loss之间的关系
plt.subplot(1, 2, 1)
plt.plot(w_history, loss_history, 'b-', alpha=0.7)
plt.scatter(w_history[0], loss_history[0], color='red', s=50, label='起点', zorder=5)
plt.scatter(w_history[-1], loss_history[-1], color='green', s=50, label='终点', zorder=5)
plt.xlabel('权重 w')
plt.ylabel('损失 Loss')
plt.title('权重 w 与损失 Loss 的关系')
plt.grid(True, alpha=0.3)
plt.legend()

# 子图2：b和loss之间的关系
plt.subplot(1, 2, 2)
plt.plot(b_history, loss_history, 'r-', alpha=0.7)
plt.scatter(b_history[0], loss_history[0], color='red', s=50, label='起点', zorder=5)
plt.scatter(b_history[-1], loss_history[-1], color='green', s=50, label='终点', zorder=5)
plt.xlabel('偏置 b')
plt.ylabel('损失 Loss')
plt.title('偏置 b 与损失 Loss 的关系')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()

# 绘制最终拟合结果
plt.figure(figsize=(10, 6))
plt.scatter(x, y, alpha=0.6, label='实际数据')
plt.plot(x, w_final * x + b_final, 'r-', linewidth=2, label=f'拟合直线: y = {w_final:.4f}x + {b_final:.4f}')
plt.xlabel('x')
plt.ylabel('y')
plt.title('线性回归拟合结果')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 打印最终损失
final_loss = np.mean((w_final * x + b_final - y) ** 2)
print(f"最终损失值：{final_loss:.4f}")

# 绘制3D损失曲面（可选）
from mpl_toolkits.mplot3d import Axes3D

# 创建网格数据
w_range = np.linspace(min(w_history) - 0.5, max(w_history) + 0.5, 50)
b_range = np.linspace(min(b_history) - 0.5, max(b_history) + 0.5, 50)
W, B = np.meshgrid(w_range, b_range)

# 计算每个点的损失
Loss = np.zeros_like(W)
for i in range(len(w_range)):
    for j in range(len(b_range)):
        y_pred = W[j, i] * x + B[j, i]
        Loss[j, i] = np.mean((y_pred - y) ** 2)

# 绘制3D图
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(W, B, Loss, cmap='viridis', alpha=0.7)
ax.plot(w_history, b_history, loss_history, 'r-', linewidth=2, label='梯度下降路径')
ax.scatter(w_history[0], b_history[0], loss_history[0], color='red', s=50, label='起点')
ax.scatter(w_history[-1], b_history[-1], loss_history[-1], color='green', s=50, label='终点')
ax.set_xlabel('权重 w')
ax.set_ylabel('偏置 b')
ax.set_zlabel('损失 Loss')
ax.set_title('损失函数曲面与梯度下降路径')
plt.legend()
plt.show()