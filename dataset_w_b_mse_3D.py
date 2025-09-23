import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

data = pd.read_csv('train.csv', encoding='gbk')

data_clean = data.dropna(axis=0, how='any')

x_data = data_clean['x'].values
y_data = data_clean['y'].values

def forward(x):
    return x * w + b

def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2

w_range = np.arange(0.0, 4.1, 0.1)
b_range = np.arange(-2.0, 2.1, 0.1)

W, B = np.meshgrid(w_range, b_range)
mse_values = np.zeros_like(W)

for i in range(len(w_range)):
    for j in range(len(b_range)):
        w = w_range[i]
        b = b_range[j]
        l_sum = 0
        for x_val, y_val in zip(x_data, y_data):
            l_sum += loss(x_val, y_val)
        mse_values[j, i] = l_sum / len(x_data)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(W, B, mse_values, cmap='viridis')

ax.set_xlabel('w')
ax.set_ylabel('b')
ax.set_zlabel('MSE')
ax.set_title('w, b, MSE 3D')

plt.tight_layout()
plt.show()