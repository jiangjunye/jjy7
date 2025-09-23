import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('train.csv', encoding='gbk')
data_clean = data.dropna(axis=0, how='any')

x_data = data_clean['x'].values
y_data = data_clean['y'].values

mask = ~np.isnan(x_data) & ~np.isnan(y_data)
x_data = x_data[mask]
y_data = y_data[mask]
def forward(x):
    return x * w + b

def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2

b_values = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0]
w_range = np.arange(0.0, 4.1, 0.1)

plt.figure(figsize=(15, 10))

for idx, b_fixed in enumerate(b_values):
    plt.subplot(2, 3, idx + 1)

    mse_values = []

    for w_val in w_range:
        w = w_val
        b = b_fixed
        l_sum = 0
        for x_val, y_val in zip(x_data, y_data):
            l_sum += loss(x_val, y_val)
        mse_values.append(l_sum / len(x_data))

    plt.plot(w_range, mse_values, 'g-', linewidth=2)
    plt.xlabel('b')
    plt.ylabel('MSE')
    plt.title(f'b = {b_fixed}')

plt.tight_layout()
plt.show()