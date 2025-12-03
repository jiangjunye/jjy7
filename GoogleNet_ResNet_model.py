import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


# 数据加载
def load_mnist(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    data_path = r'E:\yj\data'

    train_dataset = datasets.MNIST(data_path, train=True, transform=transform)
    test_dataset = datasets.MNIST(data_path, train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# GoogleNet实现
class InceptionA(nn.Module):
    def __init__(self, in_channels):
        super(InceptionA, self).__init__()
        self.branch1x1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch5x5_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch5x5_2 = nn.Conv2d(16, 24, kernel_size=5, padding=2)
        self.branch3x3_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch3x3_2 = nn.Conv2d(16, 24, kernel_size=3, padding=1)
        self.branch3x3_3 = nn.Conv2d(24, 24, kernel_size=3, padding=1)
        self.branch_pool = nn.Conv2d(in_channels, 24, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch5x5 = self.branch5x5_2(self.branch5x5_1(x))
        branch3x3 = self.branch3x3_3(self.branch3x3_2(self.branch3x3_1(x)))
        branch_pool = self.branch_pool(F.avg_pool2d(x, kernel_size=3, stride=1, padding=1))
        return torch.cat([branch1x1, branch5x5, branch3x3, branch_pool], dim=1)


class GoogleNet(nn.Module):
    def __init__(self):
        super(GoogleNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(88, 20, kernel_size=5)
        self.incep1 = InceptionA(in_channels=10)
        self.incep2 = InceptionA(in_channels=20)
        self.mp = nn.MaxPool2d(2)
        self.fc = nn.Linear(1408, 10)

    def forward(self, x):
        in_size = x.size(0)
        x = self.mp(F.relu(self.conv1(x)))
        x = self.incep1(x)
        x = self.mp(F.relu(self.conv2(x)))
        x = self.incep2(x)
        x = x.view(in_size, -1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


# ResNet实现
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = self.conv2(y)
        return F.relu(x + y)


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.mp = nn.MaxPool2d(2)
        self.rblock1 = ResidualBlock(16)
        self.rblock2 = ResidualBlock(32)
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        in_size = x.size(0)
        x = self.mp(F.relu(self.conv1(x)))
        x = self.rblock1(x)
        x = self.mp(F.relu(self.conv2(x)))
        x = self.rblock2(x)
        x = x.view(in_size, -1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


# 训练与测试函数
def train_epoch(model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)

        if batch_idx % 300 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

    avg_loss = train_loss / len(train_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print(f'\n测试集: 平均损失: {test_loss:.4f}, '
          f'准确率: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')

    return test_loss, accuracy


def train_model(model, model_name, device, train_loader, test_loader, epochs=10):
    print(f"\n开始训练{model_name}...")
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    train_losses, train_accs, test_losses, test_accs = [], [], [], []

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}:")
        print("-" * 50)
        train_loss, train_acc = train_epoch(model, device, train_loader, optimizer, epoch)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_loss, test_acc = test(model, device, test_loader)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        scheduler.step()

    return {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_losses': test_losses,
        'test_accs': test_accs,
        'final_test_acc': test_accs[-1]
    }


# 可视化函数
def plot_training_history_separate(googlenet_history, resnet_history):
    epochs = range(1, len(googlenet_history['train_losses']) + 1)

    # 图1: 训练损失对比
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, googlenet_history['train_losses'], 'b-', label='GoogleNet', marker='o')
    plt.plot(epochs, resnet_history['train_losses'], 'r-', label='ResNet', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Train Loss Comparison')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('train_loss_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 图2: 训练准确率对比
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, googlenet_history['train_accs'], 'b-', label='GoogleNet', marker='o')
    plt.plot(epochs, resnet_history['train_accs'], 'r-', label='ResNet', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Training Accuracy (%)')
    plt.title('Train Accuracy Comparison')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('train_accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 图3: 测试准确率对比
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, googlenet_history['test_accs'], 'b-', label='GoogleNet', marker='o')
    plt.plot(epochs, resnet_history['test_accs'], 'r-', label='ResNet', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Test Accuracy Comparison')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('test_accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 图4: 最终性能对比条形图
    plt.figure(figsize=(8, 6))
    models = ['GoogleNet', 'ResNet']
    final_accs = [googlenet_history['final_test_acc'], resnet_history['final_test_acc']]
    bars = plt.bar(models, final_accs, color=['blue', 'red'])
    plt.ylabel('Final Test Accuracy (%)')
    plt.title('Final Accuracy Comparison')
    plt.ylim([95, 100])

    for bar, acc in zip(bars, final_accs):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                 f'{acc:.2f}%', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('final_accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def print_model_summary(models):
    print("\n" + "=" * 60)
    print("模型性能摘要")
    print("=" * 60)

    for name, history in models.items():
        print(f"\n{name}:")
        print(f"  最终测试准确率: {history['final_test_acc']:.2f}%")
        print(
            f"  最高测试准确率: {max(history['test_accs']):.2f}% (Epoch {history['test_accs'].index(max(history['test_accs'])) + 1})")

    print("\n" + "=" * 60)


# 主程序
def main():
    print("加载MNIST数据集...")
    train_loader, test_loader = load_mnist(batch_size=64)
    googlenet = GoogleNet().to(device)
    resnet = ResNet().to(device)
    epochs = 10

    googlenet_history = train_model(googlenet, "GoogleNet", device, train_loader, test_loader, epochs)
    resnet_history = train_model(resnet, "ResNet", device, train_loader, test_loader, epochs)
    print("训练完成!")

    plot_training_history_separate(googlenet_history, resnet_history)

    models = {'GoogleNet': googlenet_history, 'ResNet': resnet_history}
    print_model_summary(models)

    print("\n性能对比总结:")
    print("-" * 40)
    if googlenet_history['final_test_acc'] > resnet_history['final_test_acc']:
        print("GoogleNet在MNIST上表现略优于ResNet")
    elif resnet_history['final_test_acc'] > googlenet_history['final_test_acc']:
        print("ResNet在MNIST上表现略优于GoogleNet")
    else:
        print("两个模型在MNIST上表现相当")


if __name__ == "__main__":
    main()