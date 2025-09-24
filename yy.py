import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 1. 用pandas读取train.csv中的数据（使用绝对路径）
df = pd.read_csv(r'C:\Users\ASUS\Downloads\train.csv')

print("原始数据信息：")
print(f"数据形状：{df.shape}")
print("\n数据前5行：")
print(df.head())
print("\n数据基本信息：")
print(df.info())
print("\n数据描述性统计：")
print(df.describe())


# 2. 数据清洗和异常值处理
def clean_data(df):
    """
    数据清洗函数
    """
    print("\n=== 数据清洗过程 ===")

    # 复制原始数据
    df_clean = df.copy()

    # 2.1 检查缺失值
    print("缺失值统计：")
    missing_values = df_clean.isnull().sum()
    print(missing_values)

    # 处理缺失值 - 删除包含缺失值的行
    if missing_values.sum() > 0:
        print(f"删除 {missing_values.sum()} 个缺失值")
        df_clean = df_clean.dropna()

    # 2.2 检查重复值
    duplicate_count = df_clean.duplicated().sum()
    print(f"重复值数量：{duplicate_count}")

    # 处理重复值 - 删除重复行
    if duplicate_count > 0:
        print(f"删除 {duplicate_count} 个重复值")
        df_clean = df_clean.drop_duplicates()

    # 2.3 异常值检测和处理
    print("\n=== 异常值检测 ===")

    # 对x和y分别进行异常值检测
    for col in ['x', 'y']:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df_clean[(df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)]
        print(f"{col}列异常值数量：{len(outliers)}")
        print(f"{col}列正常值范围：[{lower_bound:.4f}, {upper_bound:.4f}]")

        # 可视化异常值
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.boxplot(df_clean[col])
        plt.title(f'{col}列箱线图')
        plt.ylabel(col)

        plt.subplot(1, 2, 2)
        plt.scatter(range(len(df_clean)), df_clean[col], alpha=0.6)
        plt.axhline(y=lower_bound, color='r', linestyle='--', label='异常值下限')
        plt.axhline(y=upper_bound, color='r', linestyle='--', label='异常值上限')
        plt.title(f'{col}列分布')
        plt.xlabel('样本索引')
        plt.ylabel(col)
        plt.legend()
        plt.tight_layout()
        plt.show()

    # 使用IQR方法删除异常值
    print("\n使用IQR方法处理异常值...")
    initial_shape = df_clean.shape[0]

    # 定义异常值过滤函数
    def remove_outliers_iqr(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

    df_clean = remove_outliers_iqr(df_clean, 'x')
    df_clean = remove_outliers_iqr(df_clean, 'y')

    removed_count = initial_shape - df_clean.shape[0]
    print(f"删除异常值数量：{removed_count}")

    # 2.4 Z-score方法检测异常值（作为参考）
    from scipy import stats
    z_scores = np.abs(stats.zscore(df_clean[['x', 'y']]))
    z_outliers = (z_scores > 3).any(axis=1)
    print(f"Z-score方法检测到的异常值数量：{z_outliers.sum()}")

    # 2.5 数据分布可视化（清洗前后对比）
    plt.figure(figsize=(15, 5))

    # 清洗前
    plt.subplot(1, 3, 1)
    plt.scatter(df['x'], df['y'], alpha=0.6, color='red', label='原始数据')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('清洗前数据分布')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 清洗后
    plt.subplot(1, 3, 2)
    plt.scatter(df_clean['x'], df_clean['y'], alpha=0.6, color='blue', label='清洗后数据')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('清洗后数据分布')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 对比
    plt.subplot(1, 3, 3)
    plt.scatter(df['x'], df['y'], alpha=0.3, color='red', label='原始数据')
    plt.scatter(df_clean['x'], df_clean['y'], alpha=0.6, color='blue', label='清洗后数据')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('数据清洗前后对比')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"\n清洗前数据形状：{df.shape}")
    print(f"清洗后数据形状：{df_clean.shape}")
    print(f"删除数据比例：{(df.shape[0] - df_clean.shape[0]) / df.shape[0] * 100:.2f}%")

    return df_clean


# 执行数据清洗
df_clean = clean_data(df)

# 提取清洗后的x和y数据
x = df_clean['x'].values
y = df_clean['y'].values

print(f"\n清洗后数据描述性统计：")
print(df_clean.describe())


# 3. 训练线性回归模型 y = wx + b
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
print("\n=== 开始训练模型 ===")
w_final, b_final, w_history, b_history, loss_history = linear_regression(x, y, learning_rate=0.0001, epochs=10000)

print(f"训练完成！")
print(f"最终参数：w = {w_final:.4f}, b = {b_final:.4f}")

# 4. 用matplotlib绘制w和loss之间的关系、b和loss之间的关系
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
plt.scatter(x, y, alpha=0.6, label='清洗后数据')
plt.plot(x, w_final * x + b_final, 'r-', linewidth=2, label=f'拟合直线: y = {w_final:.4f}x + {b_final:.4f}')
plt.xlabel('x')
plt.ylabel('y')
plt.title('线性回归拟合结果（使用清洗后数据）')
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

# 对比清洗前后的模型效果
print("\n=== 数据清洗效果对比 ===")
# 在原始数据上训练模型（用于对比）
x_original = df['x'].values
y_original = df['y'].values
w_original, b_original, _, _, _ = linear_regression(x_original, y_original, learning_rate=0.0001, epochs=10000)
loss_original = np.mean((w_original * x_original + b_original - y_original) ** 2)

print(f"使用原始数据训练的模型：")
print(f"参数：w = {w_original:.4f}, b = {b_original:.4f}")
print(f"损失：{loss_original:.4f}")

print(f"\n使用清洗后数据训练的模型：")
print(f"参数：w = {w_final:.4f}, b = {b_final:.4f}")
print(f"损失：{final_loss:.4f}")

print(f"\n数据清洗使损失降低了：{loss_original - final_loss:.4f}")