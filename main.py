# main.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False


class LinearRegression:
    def __init__(self):
        self.w = 0.0  # 权重
        self.b = 0.0  # 偏置
        self.loss_history = []  # 记录损失历史

    def compute_loss(self, X, y, w, b):
        """计算均方误差损失"""
        n = len(y)
        y_pred = w * X + b
        loss = np.mean((y_pred - y) ** 2)
        return loss

    def train(self, X, y, learning_rate=0.01, epochs=1000):
        """使用梯度下降法训练模型"""
        n = len(y)

        # 记录不同参数下的损失
        w_values = []
        b_values = []
        loss_values = []

        # 梯度下降
        for epoch in range(epochs):
            # 前向传播
            y_pred = self.w * X + self.b

            # 计算梯度
            dw = (2 / n) * np.sum((y_pred - y) * X)
            db = (2 / n) * np.sum(y_pred - y)

            # 更新参数
            self.w -= learning_rate * dw
            self.b -= learning_rate * db

            # 计算当前损失
            current_loss = self.compute_loss(X, y, self.w, self.b)
            self.loss_history.append(current_loss)

            # 每100轮记录一次参数和损失（为了绘图，不需要太密集）
            if epoch % 100 == 0:
                w_values.append(self.w)
                b_values.append(self.b)
                loss_values.append(current_loss)

            # 每500轮打印一次进度
            if epoch % 500 == 0:
                print(f'Epoch {epoch}: w={self.w:.4f}, b={self.b:.4f}, loss={current_loss:.4f}')

        return w_values, b_values, loss_values

    def predict(self, X):
        """预测"""
        return self.w * X + self.b


def main():
    print("=== 线性回归 y = wx + b 训练程序 ===\n")

    # 1. 用pandas读取train.csv中的数据
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

    # 2. 训练线性回归模型
    print("\n=== 开始训练模型 ===")
    model = LinearRegression()
    w_values, b_values, loss_values = model.train(X, y, learning_rate=0.01, epochs=5000)

    print(f"\n训练完成！")
    print(f"最终参数: w = {model.w:.4f}, b = {model.b:.4f}")
    print(f"最终损失: {model.loss_history[-1]:.4f}")

    # 3. 用matplotlib绘制关系图
    print("\n=== 生成可视化图表 ===")

    # 创建2x2的子图布局
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 子图1: w和loss之间的关系
    axes[0, 0].plot(w_values, loss_values, 'b-', linewidth=2, marker='o', markersize=4)
    axes[0, 0].set_xlabel('权重 w')
    axes[0, 0].set_ylabel('损失 Loss')
    axes[0, 0].set_title('权重 w 与损失 Loss 的关系')
    axes[0, 0].grid(True, alpha=0.3)

    # 子图2: b和loss之间的关系
    axes[0, 1].plot(b_values, loss_values, 'r-', linewidth=2, marker='s', markersize=4)
    axes[0, 1].set_xlabel('偏置 b')
    axes[0, 1].set_ylabel('损失 Loss')
    axes[0, 1].set_title('偏置 b 与损失 Loss 的关系')
    axes[0, 1].grid(True, alpha=0.3)

    # 子图3: 损失下降曲线
    axes[1, 0].plot(model.loss_history, 'g-', linewidth=2)
    axes[1, 0].set_xlabel('训练轮数 Epoch')
    axes[1, 0].set_ylabel('损失 Loss')
    axes[1, 0].set_title('训练过程中损失下降曲线')
    axes[1, 0].set_yscale('log')  # 使用对数坐标更好地观察下降
    axes[1, 0].grid(True, alpha=0.3)

    # 子图4: 原始数据和拟合直线
    axes[1, 1].scatter(X, y, alpha=0.6, label='原始数据')
    x_line = np.linspace(X.min(), X.max(), 100)
    y_line = model.predict(x_line)
    axes[1, 1].plot(x_line, y_line, 'r-', linewidth=3, label=f'拟合直线: y = {model.w:.2f}x + {model.b:.2f}')
    axes[1, 1].set_xlabel('X')
    axes[1, 1].set_ylabel('y')
    axes[1, 1].set_title('线性回归拟合结果')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("图表已保存为 'training_results.png'")


if __name__ == "__main__":
    main()