import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# 读取数据
data = pd.read_csv('train.csv')

print("原始数据信息:")
print(f"数据形状: {data.shape}")
print(f"数据前5行:\n{data.head()}")
print(f"\n数据基本信息:")
print(data.info())
print(f"\n数据描述统计:")
print(data.describe())


# 数据清洗函数
def clean_data(df):
    """
    数据清洗函数
    包括处理缺失值、异常值等
    """
    df_clean = df.copy()

    # 1. 检查缺失值
    print(f"\n缺失值检查:")
    print(df_clean.isnull().sum())

    # 处理缺失值 - 删除包含缺失值的行
    initial_shape = df_clean.shape[0]
    df_clean = df_clean.dropna()
    final_shape = df_clean.shape[0]
    print(f"删除缺失值: {initial_shape} -> {final_shape} 行")

    # 2. 检查重复值
    print(f"\n重复值检查:")
    duplicate_count = df_clean.duplicated().sum()
    print(f"重复行数: {duplicate_count}")

    # 删除重复值
    if duplicate_count > 0:
        initial_shape = df_clean.shape[0]
        df_clean = df_clean.drop_duplicates()
        final_shape = df_clean.shape[0]
        print(f"删除重复值: {initial_shape} -> {final_shape} 行")

    # 3. 异常值检测和处理
    print(f"\n异常值检测:")

    # 使用IQR方法检测异常值
    for column in df_clean.columns:
        if df_clean[column].dtype in ['float64', 'int64']:
            Q1 = df_clean[column].quantile(0.25)
            Q3 = df_clean[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = df_clean[(df_clean[column] < lower_bound) | (df_clean[column] > upper_bound)]
            print(f"{column}: {len(outliers)} 个异常值 (范围: [{lower_bound:.4f}, {upper_bound:.4f}])")

            # 可以选择删除异常值或进行缩尾处理
            # 这里选择缩尾处理 (Winsorization)
            df_clean[column] = np.clip(df_clean[column], lower_bound, upper_bound)

    # 4. 检查数据分布
    print(f"\n清洗后数据描述统计:")
    print(df_clean.describe())

    return df_clean


# 执行数据清洗
print("=" * 50)
print("开始数据清洗...")
print("=" * 50)
cleaned_data = clean_data(data)

# 准备训练数据（只使用清洗后的数据）
x_data = cleaned_data['x'].values.reshape(-1, 1)
y_data = cleaned_data['y'].values.reshape(-1, 1)

print(f"\n清洗后数据形状: x_data: {x_data.shape}, y_data: {y_data.shape}")

# 数据标准化
scaler_x = StandardScaler()
scaler_y = StandardScaler()
x_data_scaled = scaler_x.fit_transform(x_data)
y_data_scaled = scaler_y.fit_transform(y_data)

# 转换为Tensor
x_tensor = torch.Tensor(x_data_scaled)
y_tensor = torch.Tensor(y_data_scaled)

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(
    x_tensor, y_tensor, test_size=0.2, random_state=42
)

print(f"\n训练集大小: {x_train.shape[0]}, 测试集大小: {x_test.shape[0]}")


class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


def train_model(optimizer_class, optimizer_name, lr=0.01, epochs=1000):
    """训练模型并返回结果"""
    model = LinearModel()
    criterion = nn.MSELoss()

    # 根据优化器类型选择不同的参数
    if optimizer_name == 'LBFGS':
        optimizer = optimizer_class(model.parameters(), lr=lr, max_iter=20)
    else:
        optimizer = optimizer_class(model.parameters(), lr=lr)

    train_losses = []
    test_losses = []

    for epoch in range(epochs):
        # 训练
        model.train()
        if optimizer_name == 'LBFGS':
            def closure():
                optimizer.zero_grad()
                y_pred = model(x_train)
                loss = criterion(y_pred, y_train)
                loss.backward()
                return loss

            loss = optimizer.step(closure)
        else:
            optimizer.zero_grad()
            y_pred = model(x_train)
            loss = criterion(y_pred, y_train)
            loss.backward()
            optimizer.step()

        train_losses.append(loss.item())

        # 测试
        model.eval()
        with torch.no_grad():
            y_test_pred = model(x_test)
            test_loss = criterion(y_test_pred, y_test)
            test_losses.append(test_loss.item())

    # 最终评估
    model.eval()
    with torch.no_grad():
        y_pred_final = model(x_test)
        final_loss = criterion(y_pred_final, y_test)

        # 反标准化预测结果
        y_pred_original = scaler_y.inverse_transform(y_pred_final.numpy())
        y_test_original = scaler_y.inverse_transform(y_test.numpy())

        # 计算R2分数
        ss_res = np.sum((y_test_original - y_pred_original) ** 2)
        ss_tot = np.sum((y_test_original - np.mean(y_test_original)) ** 2)
        r2 = 1 - (ss_res / ss_tot)

    return {
        'name': optimizer_name,
        'model': model,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'final_train_loss': train_losses[-1],
        'final_test_loss': test_losses[-1],
        'r2_score': r2,
        'weights': model.linear.weight.item(),
        'bias': model.linear.bias.item()
    }


# 定义要比较的优化器
optimizers = [
    (torch.optim.SGD, 'SGD'),
    (torch.optim.Adam, 'Adam'),
    (torch.optim.Adagrad, 'Adagrad'),
    (torch.optim.Adamax, 'Adamax'),
    (torch.optim.ASGD, 'ASGD'),
    (torch.optim.LBFGS, 'LBFGS'),
    (torch.optim.RMSprop, 'RMSprop'),
    (torch.optim.Rprop, 'Rprop')
]

# 训练所有优化器
results = []
print("\n" + "=" * 80)
print("开始训练不同优化器（使用清洗后数据）...")
print("=" * 80)

for optimizer_class, optimizer_name in optimizers:
    print(f"训练 {optimizer_name}...")
    try:
        result = train_model(optimizer_class, optimizer_name, lr=0.01, epochs=1000)
        results.append(result)
        print(f"{optimizer_name} 训练完成 - 最终训练损失: {result['final_train_loss']:.6f}, "
              f"测试损失: {result['final_test_loss']:.6f}, R2: {result['r2_score']:.4f}")
    except Exception as e:
        print(f"{optimizer_name} 训练失败: {e}")

print("\n" + "=" * 80)
print("优化器对比结果（基于清洗后数据）:")
print("=" * 80)

# 按测试损失排序
results.sort(key=lambda x: x['final_test_loss'])

for i, result in enumerate(results, 1):
    print(f"{i}. {result['name']:8} | "
          f"训练损失: {result['final_train_loss']:.6f} | "
          f"测试损失: {result['final_test_loss']:.6f} | "
          f"R2: {result['r2_score']:.4f} | "
          f"权重: {result['weights']:.4f} | "
          f"偏置: {result['bias']:.4f}")

# 绘制训练损失曲线
plt.figure(figsize=(15, 12))

# 训练损失
plt.subplot(2, 2, 1)
for result in results:
    plt.plot(result['train_losses'][:100], label=result['name'])
plt.title('Training Loss - First 100 Epochs (Cleaned Data)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.yscale('log')
plt.grid(True, alpha=0.3)

# 测试损失
plt.subplot(2, 2, 2)
for result in results:
    plt.plot(result['test_losses'][:100], label=result['name'])
plt.title('Test Loss - First 100 Epochs (Cleaned Data)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.yscale('log')
plt.grid(True, alpha=0.3)

# 完整训练损失
plt.subplot(2, 2, 3)
for result in results:
    plt.plot(result['train_losses'], label=result['name'])
plt.title('Full Training Loss (Cleaned Data)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.yscale('log')
plt.grid(True, alpha=0.3)

# 性能对比条形图
plt.subplot(2, 2, 4)
names = [result['name'] for result in results]
test_losses = [result['final_test_loss'] for result in results]
bars = plt.bar(names, test_losses, color=plt.cm.Set3(np.linspace(0, 1, len(names))))
plt.title('Final Test Loss Comparison (Cleaned Data)')
plt.xlabel('Optimizer')
plt.ylabel('Test Loss')
plt.xticks(rotation=45)

# 在条形图上添加数值
for bar, loss in zip(bars, test_losses):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
             f'{loss:.4f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()

# 选择最佳模型进行预测
best_result = results[0]
print(f"\n最佳优化器: {best_result['name']}")
print(f"最佳模型参数: w = {best_result['weights']:.4f}, b = {best_result['bias']:.4f}")
print(f"最佳R2分数: {best_result['r2_score']:.4f}")

# 使用最佳模型进行预测示例
best_model = best_result['model']
x_sample = torch.Tensor(scaler_x.transform(np.array([[4.0]])))
y_pred_sample = best_model(x_sample)
y_pred_original = scaler_y.inverse_transform(y_pred_sample.detach().numpy())
print(f"输入 x=4.0 的预测结果: {y_pred_original[0][0]:.4f}")

# 数据清洗效果和模型拟合可视化
plt.figure(figsize=(15, 5))

# 数据清洗前后对比
plt.subplot(1, 3, 1)
plt.scatter(data['x'], data['y'], alpha=0.6, color='red', label='原始数据', s=30)
plt.scatter(cleaned_data['x'], cleaned_data['y'], alpha=0.6, color='blue', label='清洗后数据', s=30)
plt.title('数据清洗前后对比')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True, alpha=0.3)

# 最佳模型拟合效果
plt.subplot(1, 3, 2)
x_range = np.linspace(cleaned_data['x'].min(), cleaned_data['x'].max(), 100).reshape(-1, 1)
x_range_scaled = scaler_x.transform(x_range)
x_range_tensor = torch.Tensor(x_range_scaled)

best_model.eval()
with torch.no_grad():
    y_range_pred = best_model(x_range_tensor)
    y_range_original = scaler_y.inverse_transform(y_range_pred.numpy())

plt.scatter(cleaned_data['x'], cleaned_data['y'], alpha=0.6, color='blue', label='清洗后数据', s=30)
plt.plot(x_range, y_range_original, 'r-', linewidth=2, label=f'最佳模型 ({best_result["name"]})')
plt.title('最佳模型拟合效果')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True, alpha=0.3)

# 优化器性能对比（R2分数）
plt.subplot(1, 3, 3)
names = [result['name'] for result in results]
r2_scores = [result['r2_score'] for result in results]
bars = plt.bar(names, r2_scores, color=plt.cm.Set3(np.linspace(0, 1, len(names))))
plt.title('优化器R2分数对比 (Cleaned Data)')
plt.xlabel('Optimizer')
plt.ylabel('R2 Score')
plt.xticks(rotation=45)

# 在条形图上添加数值
for bar, r2 in zip(bars, r2_scores):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
             f'{r2:.4f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()

# 输出清洗效果总结
print("\n" + "=" * 80)
print("数据清洗效果总结:")
print("=" * 80)
print(f"原始数据行数: {data.shape[0]}")
print(f"清洗后数据行数: {cleaned_data.shape[0]}")
print(f"数据保留比例: {cleaned_data.shape[0] / data.shape[0] * 100:.2f}%")
print(f"使用的优化器数量: {len(results)}")
print(f"最佳优化器: {best_result['name']}")
print(f"在清洗数据上的最佳R2分数: {best_result['r2_score']:.4f}")