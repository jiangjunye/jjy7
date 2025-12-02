import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

data = pd.read_csv('train.csv')
data = data.dropna(subset=['y'])

x_data = torch.Tensor(data[['x']].values)
y_data = torch.Tensor(data[['y']].values)

class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.Linear = torch.nn.Linear(1, 1)
        torch.nn.init.normal_(self.Linear.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.Linear.bias, mean=0.0, std=1.0)

    def forward(self, x):
        return self.Linear(x)

model = LinearModel()

criterion = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adamax(model.parameters(), lr=0.001)

w_history = []
b_history = []
loss_history = []

for epoch in range(5000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)

    w_history.append(model.Linear.weight.item())
    b_history.append(model.Linear.bias.item())
    loss_history.append(loss.item())

    if epoch % 1000 == 0:
        print(f'第{epoch}轮, 损失: {loss.item():.4f}')

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f'\n最终参数: w = {model.Linear.weight.item():.6f}, b = {model.Linear.bias.item():.6f}')

with torch.no_grad():
    y_pred_all = model(x_data)
    mse_final = criterion(y_pred_all, y_data)

plt.figure(figsize=(10, 6))

epochs = range(len(w_history))
plt.plot(epochs, w_history, 'r-', label='权重 w', alpha=0.7, linewidth=1.5)
plt.plot(epochs, b_history, 'g-', label='偏置 b', alpha=0.7, linewidth=1.5)
plt.xlabel('训练轮次')
plt.ylabel('参数值')
plt.title('参数w和b随训练轮次的变化')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('Adamax_w_b_comparison.png', dpi=150, bbox_inches='tight')

plt.figure(figsize=(10, 8))

y_true = y_data.numpy().flatten()
y_pred = y_pred_all.numpy().flatten()

min_val = min(y_true.min(), y_pred.min())
max_val = max(y_true.max(), y_pred.max())
perfect_line = np.linspace(min_val, max_val, 100)

plt.plot(perfect_line, perfect_line, 'r--', alpha=0.8, linewidth=2, label='完美预测线 (y=x)')
plt.scatter(y_true, y_pred, alpha=0.6, color='blue', s=50, label='预测点')

plt.tight_layout()
plt.savefig('Adamax_model_effect.png', dpi=150, bbox_inches='tight')