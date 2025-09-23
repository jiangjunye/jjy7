import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('train.csv', encoding='gbk')

data_clean = data.dropna(axis=0, how='any')

x_data = data_clean['x'].values
y_data = data_clean['y'].values

def forward(x):
    return x * w + b

def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2

w_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
b_range = np.arange(-2.0, 2.1, 0.1)

plt.figure(figsize=(15, 10))

for idx, w_fixed in enumerate(w_values):
    plt.subplot(2, 3, idx + 1)

    mse_values = []

    for b in b_range:
        w = w_fixed
        l_sum = 0
        for x_val, y_val in zip(x_data, y_data):
            l_sum += loss(x_val, y_val)
        mse_values.append(l_sum / len(x_data))

    plt.plot(b_range, mse_values, 'b-', linewidth=2)
    plt.xlabel('b')
    plt.ylabel('MSE')
    plt.title(f'w = {w_fixed}')

plt.tight_layout()
plt.show()