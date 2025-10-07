import torch
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv(r'E:\yj\train.csv')
data_cleaned = data.dropna()
x_data = torch.Tensor(data_cleaned['x'].values).reshape(-1, 1)
y_data = torch.Tensor(data_cleaned['y'].values).reshape(-1, 1)
class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)
        with torch.no_grad():
            self.linear.weight.data.fill_(1.0)
            self.linear.bias.data.fill_(1.0)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

model = LinearModel()
loss_history_adam = []
criterion = torch.nn.MSELoss(size_average=False)
optimizer_adam = torch.optim.Adam(model.parameters(), lr=0.001)
for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())

    optimizer_adam.zero_grad()
    loss.backward()
    optimizer_adam.step()
    loss_history_adam.append(loss.item())

print('w = ', model.linear.weight.item())
print('b = ', model.linear.bias.item())
x_test = torch.Tensor([[4.0]])
y_test = model(x_test)
print('y_pred = ', y_test.data)
with torch.no_grad():
    y_pred_all = model(x_data)
    ss_res = torch.sum((y_data - y_pred_all) ** 2)
    ss_tot = torch.sum((y_data - torch.mean(y_data)) ** 2)
    r_squared = 1 - ss_res / ss_tot
    print(f'\n模型R²分数: {r_squared.item():.4f}')

plt.plot(loss_history_adam, label='Adam')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Adam')
plt.legend()
plt.show()