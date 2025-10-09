import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv(r'E:\yj\train.csv')
data_cleaned = data.dropna()
x_data = data_cleaned['x'].values.astype(int)
y_data = data_cleaned['y'].values

def forward(x,w,b):
    return x * w + b

def loss(x,y,w,b):
    y_pred = forward(x,w,b)
    return (y_pred - y) * (y_pred - y)

def mse(x_data, y_data, w, b):
    l_sum = 0
    for x, y in zip(x_data, y_data):
        l_sum += loss(x, y, w, b)
    return l_sum / len(x_data)

w_list = []
b_list = []
mse_list_w = []
mse_list_b = []
b_fixed = 0
for w in np.arange(-1.0,3.1,0.1):
    mse_val = mse(x_data, y_data, w, b_fixed)
    w_list.append(w)
    mse_list_w.append(mse_val)
    print(f'w={w:.1f}, MSE={mse_val:.2f}')

w_fixed = 1
for b in np.arange(-20.0, 20.1, 0.5):
    mse_val = mse(x_data, y_data, w_fixed, b)
    b_list.append(b)
    mse_list_b.append(mse_val)
    print(f'b={b:.1f}, MSE={mse_val:.2f}')

plt.subplot(1,2,1)
plt.plot(w_list,mse_list_w)
plt.ylabel('Loss')
plt.xlabel('w')
plt.title('b=0,w-loss')
plt.subplot(1,2,2)
plt.plot(b_list,mse_list_b)
plt.ylabel('loss')
plt.xlabel('b')
plt.title('w=1,b-loss')
plt.tight_layout()
plt.show()