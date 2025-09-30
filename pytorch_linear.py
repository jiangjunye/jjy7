import torch
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv(r'E:\yj\train.csv')
data_cleaned = data.dropna()
x_data = data_cleaned['x'].values.astype(int)
y_data = data_cleaned['y'].values
w = torch.tensor([1.0], requires_grad=True)
b = torch.tensor([4.0], requires_grad=True)

w_history = []
loss_history = []
def forward(x):
    return x * w + b

def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2

print("predict (before training)", 4, forward(4).item())
for epoch in range(100):
    for x, y in zip(x_data, y_data):
        l = loss(x, y)
        l.backward()
        print('\tgrad:', x, y, w.grad.item())
        w.data = w.data - 0.0000001 * w.grad.data
        w.grad.data.zero_()
    loss_history.append(l.item())
    w_history.append(w.item())
    print("progress:", epoch, l.item())
print("predict (after training)", 4, forward(4).item())
plt.plot(w_history,loss_history)
plt.xlabel('w')
plt.ylabel('Loss')
plt.title('w-loss')
plt.show()