import numpy as np
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

def forward(x, w, b):
    return x * w + b

def loss(x, y, w, b):
    y_pred = forward(x, w, b)
    return (y_pred - y) ** 2

w_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

plt.figure(figsize=(15, 10))

for i, w in enumerate(w_values):
    b_list = []
    mse_list = []

    for b in np.arange(-2.0, 2.1, 0.1):
        l_sum = 0
        for x_val, y_val in zip(x_data, y_data):
            loss_val = loss(x_val, y_val, w, b)
            l_sum += loss_val

        mse = l_sum / 3
        b_list.append(b)
        mse_list.append(mse)

    plt.subplot(2, 3, i + 1)
    plt.plot(b_list, mse_list)
    plt.xlabel('b')
    plt.ylabel('MSE')
    plt.title(f'w = {w}')

plt.tight_layout()
plt.show()