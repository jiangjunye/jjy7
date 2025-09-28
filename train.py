# create_data.py
import pandas as pd
import numpy as np

# 生成示例数据
np.random.seed(42)
n_samples = 100
w_true = 2.5
b_true = 1.0

x = np.linspace(0, 10, n_samples)
y = w_true * x + b_true + np.random.normal(0, 1, n_samples)

# 创建DataFrame并保存
df = pd.DataFrame({'x': x, 'y': y})
df.to_csv('data/train.csv', index=False)
print("数据已生成并保存到 data/train.csv")