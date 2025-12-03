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

criterion = torch.nn.MSELoss(reduction='mean')


def train_with_params(epochs, lr):
    model = LinearModel()
    optimizer = torch.optim.Adamax(model.parameters(), lr=lr)

    loss_history = []

    for epoch in range(epochs):
        y_pred = model(x_data)
        loss = criterion(y_pred, y_data)
        loss_history.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss_history


epochs_list = [1000, 3000, 5000, 10000]
learning_rates = [0.0001, 0.001, 0.01, 0.1]

results = {}

for epochs in epochs_list:
    for lr in learning_rates:
        key = f"epochs_{epochs}_lr_{lr}"
        print(f"训练: epochs={epochs}, lr={lr}")
        results[key] = train_with_params(epochs, lr)

fixed_lr = 0.001
plt.figure(figsize=(12, 6))

for epochs in epochs_list:
    key = f"epochs_{epochs}_lr_{fixed_lr}"
    loss_history = results[key]
    plt.plot(range(len(loss_history)), loss_history,
             label=f'Epochs={epochs}', linewidth=2)

plt.xlabel('训练轮次')
plt.ylabel('损失值')
plt.title(f'固定学习率 LR={fixed_lr} - 不同训练轮次对比')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')
plt.tight_layout()
plt.savefig('fixed_lr_epochs_comparison.png', dpi=150, bbox_inches='tight')

fixed_epochs = 5000
plt.figure(figsize=(12, 6))

for lr in learning_rates:
    key = f"epochs_{fixed_epochs}_lr_{lr}"
    loss_history = results[key]
    plt.plot(range(len(loss_history)), loss_history,
             label=f'LR={lr}', linewidth=2)

plt.xlabel('训练轮次')
plt.ylabel('损失值')
plt.title(f'固定训练轮次 Epochs={fixed_epochs} - 不同学习率对比')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')
plt.tight_layout()
plt.savefig('fixed_epochs_lr_comparison.png', dpi=150, bbox_inches='tight')