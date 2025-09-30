# main_pytorch.py
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False


class PyTorchLinearRegression(nn.Module):
    def __init__(self):
        super(PyTorchLinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)  # 输入维度1，输出维度1

    def forward(self, x):
        return self.linear(x)


class LinearRegressionTrainer:
    def __init__(self):
        self.model = PyTorchLinearRegression()
        self.criterion = nn.MSELoss()
        self.loss_history = []
        self.w_history = []
        self.b_history = []

    def compute_loss(self, X, y):
        """计算损失"""
        with torch.no_grad():
            predictions = self.model(X)
            loss = self.criterion(predictions, y)
            return loss.item()

    def train(self, X, y, learning_rate=0.01, epochs=1000):
        """使用PyTorch训练模型"""
        # 转换为PyTorch张量
        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X.reshape(-1, 1))
        if isinstance(y, np.ndarray):
            y = torch.FloatTensor(y.reshape(-1, 1))

        # 初始化优化器
        optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)

        # 记录初始参数
        with torch.no_grad():
            w0 = self.model.linear.weight.item()
            b0 = self.model.linear.bias.item()
            self.w_history.append(w0)
            self.b_history.append(b0)
            initial_loss = self.compute_loss(X, y)
            self.loss_history.append(initial_loss)

        print(f"初始参数: w={w0:.4f}, b={b0:.4f}, loss={initial_loss:.4f}")

        # 训练循环
        for epoch in range(epochs):
            # 前向传播
            predictions = self.model(X)
            loss = self.criterion(predictions, y)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 记录损失
            self.loss_history.append(loss.item())

            # 每100轮记录一次参数
            if epoch % 100 == 0:
                with torch.no_grad():
                    current_w = self.model.linear.weight.item()
                    current_b = self.model.linear.bias.item()
                    self.w_history.append(current_w)
                    self.b_history.append(current_b)

            # 每500轮打印一次进度
            if epoch % 500 == 0:
                print(f'Epoch {epoch}: loss={loss.item():.4f}')

        # 记录最终参数
        with torch.no_grad():
            final_w = self.model.linear.weight.item()
            final_b = self.model.linear.bias.item()
            self.w_history.append(final_w)
            self.b_history.append(final_b)

        return self.w_history, self.b_history, self.loss_history

    def predict(self, X):
        """预测"""
        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X.reshape(-1, 1))

        with torch.no_grad():
            predictions = self.model(X)
            return predictions.numpy().flatten()

    def get_parameters(self):
        """获取模型参数"""
        with torch.no_grad():
            w = self.model.linear.weight.item()
            b = self.model.linear.bias.item()
            return w, b


def create_3d_loss_surface(X, y, w_range=(-1, 4), b_range=(-2, 4), points=50):
    """创建3D损失曲面"""
    if isinstance(X, np.ndarray):
        X_tensor = torch.FloatTensor(X.reshape(-1, 1))
    if isinstance(y, np.ndarray):
        y_tensor = torch.FloatTensor(y.reshape(-1, 1))

    # 生成参数网格
    w_values = np.linspace(w_range[0], w_range[1], points)
    b_values = np.linspace(b_range[0], b_range[1], points)
    W, B = np.meshgrid(w_values, b_values)

    # 计算损失
    Z = np.zeros_like(W)
    criterion = nn.MSELoss()

    for i in range(points):
        for j in range(points):
            with torch.no_grad():
                # 创建临时模型
                temp_model = PyTorchLinearRegression()
                temp_model.linear.weight.data.fill_(W[i, j])
                temp_model.linear.bias.data.fill_(B[i, j])

                predictions = temp_model(X_tensor)
                loss = criterion(predictions, y_tensor)
                Z[i, j] = loss.item()

    return W, B, Z


def main():
    print("=== PyTorch线性回归 y = wx + b 训练程序 ===\n")

    # 1. 读取数据
    try:
        df = pd.read_csv('data/train.csv')
        X = df['x'].values
        y = df['y'].values
        print("数据读取成功！")
        print(f"数据形状: X{len(X)}, y{len(y)}")
        print(f"前5个样本: X={X[:5]}, y={y[:5]}")
    except FileNotFoundError:
        print("错误: 找不到 data/train.csv 文件")
        print("请先运行 create_data.py 生成数据")
        return

    # 2. 训练PyTorch线性回归模型
    print("\n=== 开始训练PyTorch模型 ===")
    trainer = LinearRegressionTrainer()
    w_values, b_values, loss_values = trainer.train(X, y, learning_rate=0.01, epochs=5000)

    w_final, b_final = trainer.get_parameters()
    print(f"\n训练完成！")
    print(f"最终参数: w = {w_final:.4f}, b = {b_final:.4f}")
    print(f"最终损失: {loss_values[-1]:.4f}")

    # 3. 可视化结果
    print("\n=== 生成可视化图表 ===")

    # 创建更丰富的可视化
    fig = plt.figure(figsize=(20, 15))

    # 子图1: w和loss之间的关系
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(w_values, loss_values[:len(w_values)], 'b-', linewidth=2, marker='o', markersize=4)
    ax1.set_xlabel('权重 w')
    ax1.set_ylabel('损失 Loss')
    ax1.set_title('权重 w 与损失 Loss 的关系')
    ax1.grid(True, alpha=0.3)

    # 子图2: b和loss之间的关系
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(b_values, loss_values[:len(b_values)], 'r-', linewidth=2, marker='s', markersize=4)
    ax2.set_xlabel('偏置 b')
    ax2.set_ylabel('损失 Loss')
    ax2.set_title('偏置 b 与损失 Loss 的关系')
    ax2.grid(True, alpha=0.3)

    # 子图3: 损失下降曲线
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(loss_values, 'g-', linewidth=2)
    ax3.set_xlabel('训练轮数 Epoch')
    ax3.set_ylabel('损失 Loss')
    ax3.set_title('训练过程中损失下降曲线')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)

    # 子图4: 原始数据和拟合直线
    ax4 = plt.subplot(2, 3, 4)
    ax4.scatter(X, y, alpha=0.6, label='原始数据')
    x_line = np.linspace(X.min(), X.max(), 100)
    y_line = trainer.predict(x_line)
    ax4.plot(x_line, y_line, 'r-', linewidth=3,
             label=f'拟合直线: y = {w_final:.2f}x + {b_final:.2f}')
    ax4.set_xlabel('X')
    ax4.set_ylabel('y')
    ax4.set_title('PyTorch线性回归拟合结果')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 子图5: 参数更新轨迹（2D）
    ax5 = plt.subplot(2, 3, 5)
    # 生成损失等高线
    w_min, w_max = min(w_values) - 0.5, max(w_values) + 0.5
    b_min, b_max = min(b_values) - 0.5, max(b_values) + 0.5

    w_grid = np.linspace(w_min, w_max, 50)
    b_grid = np.linspace(b_min, b_max, 50)
    W_grid, B_grid = np.meshgrid(w_grid, b_grid)

    # 简化计算损失（实际应用中可能需要优化）
    Z_simple = np.zeros_like(W_grid)
    for i in range(len(w_grid)):
        for j in range(len(b_grid)):
            y_pred = W_grid[i, j] * X + B_grid[i, j]
            Z_simple[i, j] = np.mean((y_pred - y) ** 2)

    contour = ax5.contour(W_grid, B_grid, Z_simple, levels=20, alpha=0.6)
    ax5.clabel(contour, inline=True, fontsize=8)
    ax5.plot(w_values, b_values, 'ro-', markersize=4, linewidth=2,
             label='参数更新轨迹')
    ax5.plot(w_values[0], b_values[0], 'go', markersize=8, label='起点')
    ax5.plot(w_values[-1], b_values[-1], 'bo', markersize=8, label='终点')
    ax5.set_xlabel('权重 w')
    ax5.set_ylabel('偏置 b')
    ax5.set_title('参数空间中的优化轨迹')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 子图6: 梯度下降动态效果（简化版）
    ax6 = plt.subplot(2, 3, 6)
    ax6.plot(range(len(loss_values)), loss_values, 'purple', linewidth=2)
    ax6.set_xlabel('训练步数')
    ax6.set_ylabel('损失值')
    ax6.set_title('梯度下降过程')
    ax6.set_yscale('log')
    ax6.grid(True, alpha=0.3)

    # 标记关键点
    key_points = [0, len(loss_values) // 4, len(loss_values) // 2,
                  3 * len(loss_values) // 4, len(loss_values) - 1]
    for point in key_points:
        if point < len(loss_values):
            ax6.plot(point, loss_values[point], 'ro', markersize=6)
            ax6.annotate(f'Step {point}', (point, loss_values[point]),
                         xytext=(10, 10), textcoords='offset points')

    plt.tight_layout()
    plt.savefig('pytorch_training_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 4. 生成3D损失曲面（可选，需要更多计算资源）
    print("正在生成3D损失曲面...")
    try:
        fig_3d = plt.figure(figsize=(12, 10))
        ax_3d = fig_3d.add_subplot(111, projection='3d')

        # 使用简化版本生成3D曲面（减少点数以提高速度）
        W_3d, B_3d, Z_3d = create_3d_loss_surface(X, y, points=30)

        surf = ax_3d.plot_surface(W_3d, B_3d, Z_3d, cmap='viridis',
                                  alpha=0.8, linewidth=0, antialiased=True)

        # 绘制优化路径
        ax_3d.plot(w_values, b_values, loss_values[:len(w_values)],
                   'r.-', markersize=8, linewidth=2, label='优化路径')

        ax_3d.set_xlabel('权重 w')
        ax_3d.set_ylabel('偏置 b')
        ax_3d.set_zlabel('损失 Loss')
        ax_3d.set_title('3D损失曲面与优化路径')
        ax_3d.legend()

        fig_3d.colorbar(surf, ax=ax_3d, shrink=0.5, aspect=5)
        plt.savefig('3d_loss_surface.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("3D损失曲面已保存为 '3d_loss_surface.png'")
    except Exception as e:
        print(f"生成3D曲面时出错: {e}")

    print("PyTorch训练结果图表已保存为 'pytorch_training_results.png'")


if __name__ == "__main__":
    main()