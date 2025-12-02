import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1conv=False, strides=1):
        super(Residual, self).__init__()
        self.ReLU = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=num_channels, kernel_size=3, padding=1,
                               stride=strides)
        self.conv2 = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        if use_1conv:
            self.conv3 = nn.Conv2d(in_channels=input_channels, out_channels=num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None

    def forward(self, x):
        y = self.ReLU(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        y = self.ReLU(y + x)
        return y


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.b2 = nn.Sequential(
            Residual(64, 64, use_1conv=False, strides=1),
            Residual(64, 64, use_1conv=False, strides=1)
        )

        self.b3 = nn.Sequential(
            Residual(64, 128, use_1conv=True, strides=2),
            Residual(128, 128, use_1conv=False, strides=1)
        )

        self.b4 = nn.Sequential(
            Residual(128, 256, use_1conv=True, strides=2),
            Residual(256, 256, use_1conv=False, strides=1)
        )

        self.b5 = nn.Sequential(
            Residual(256, 512, use_1conv=True, strides=2),
            Residual(512, 512, use_1conv=False, strides=1)
        )

        self.b6 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        x = self.b6(x)
        return x

class Inception(nn.Module):
    def __init__(self, in_channels, c1, c2, c3, c4):
        super(Inception, self).__init__()
        self.ReLU = nn.ReLU()

        # 单1×1卷积层
        self.p1_1 = nn.Conv2d(in_channels=in_channels, out_channels=c1, kernel_size=1)

        # 1×1卷积层, 3×3的卷积
        self.p2_1 = nn.Conv2d(in_channels=in_channels, out_channels=c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(in_channels=c2[0], out_channels=c2[1], kernel_size=3, padding=1)

        #1×1卷积层, 5×5的卷积
        self.p3_1 = nn.Conv2d(in_channels=in_channels, out_channels=c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(in_channels=c3[0], out_channels=c3[1], kernel_size=5, padding=2)

        # 3×3的最大池化, 1×1的卷积
        self.p4_1 = nn.MaxPool2d(kernel_size=3, padding=1, stride=1)
        self.p4_2 = nn.Conv2d(in_channels=in_channels, out_channels=c4, kernel_size=1)

    def forward(self, x):
        p1 = self.ReLU(self.p1_1(x))
        p2 = self.ReLU(self.p2_2(self.ReLU(self.p2_1(x))))
        p3 = self.ReLU(self.p3_2(self.ReLU(self.p3_1(x))))
        p4 = self.ReLU(self.p4_2(self.p4_1(x)))
        return torch.cat((p1, p2, p3, p4), dim=1)


class GoogLeNet(nn.Module):
    def __init__(self):
        super(GoogLeNet, self).__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.b2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.b3 = nn.Sequential(
            Inception(192, 64, (96, 128), (16, 32), 32),
            Inception(256, 128, (128, 192), (32, 96), 64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.b4 = nn.Sequential(
            Inception(480, 192, (96, 208), (16, 48), 64),
            Inception(512, 160, (112, 224), (24, 64), 64),
            Inception(512, 128, (128, 256), (24, 64), 64),
            Inception(512, 112, (128, 288), (32, 64), 64),
            Inception(528, 256, (160, 320), (32, 128), 128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.b5 = nn.Sequential(
            Inception(832, 256, (160, 320), (32, 128), 128),
            Inception(832, 384, (192, 384), (48, 128), 128),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1024, 10)
        )

        # 模型参数初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        return x


def train_epoch(model, train_loader, criterion, optimizer, device, model_name):
    """训练"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = 100. * correct / total

    return train_loss, train_acc


def test_model(model, test_loader, criterion, device, model_name):
    """测试模型性能"""
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    test_loss /= len(test_loader)
    test_acc = 100. * correct / total

    return test_loss, test_acc


def train_model(model, model_name, train_loader, test_loader, epochs, device):
    """训练整个模型"""
    print(f"\n{'=' * 50}")
    print(f"Training {model_name}")
    print(f"{'=' * 50}")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []

    for epoch in range(1, epochs + 1):
        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, criterion,
                                            optimizer, device, model_name)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # 测试
        test_loss, test_acc = test_model(model, test_loader, criterion,
                                         device, model_name)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        # 输出每个epoch的loss
        print(f'{model_name} - Epoch {epoch}: Train Loss: {train_loss:.4f} | '
              f'Train Acc: {train_acc:.2f}% | Test Loss: {test_loss:.4f} | '
              f'Test Acc: {test_acc:.2f}%')

    return {
        'model': model,
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_losses': test_losses,
        'test_accs': test_accs
    }


def plot_loss_comparison(resnet_results, googlenet_results, epochs, save_path='loss_comparison.png'):
    """Loss曲线"""
    plt.figure(figsize=(10, 6))
    epochs_range = range(1, epochs + 1)

    plt.plot(epochs_range, resnet_results['train_losses'], 'b-',
             linewidth=2, label='ResNet Train Loss')
    plt.plot(epochs_range, resnet_results['test_losses'], 'b--',
             linewidth=2, label='ResNet Test Loss')
    plt.plot(epochs_range, googlenet_results['train_losses'], 'r-',
             linewidth=2, label='GoogLeNet Train Loss')
    plt.plot(epochs_range, googlenet_results['test_losses'], 'r--',
             linewidth=2, label='GoogLeNet Test Loss')

    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Testing Loss Comparison', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')


def plot_accuracy_comparison(resnet_results, googlenet_results, epochs, save_path='accuracy_comparison.png'):
    """绘制Accuracy曲线"""
    plt.figure(figsize=(10, 6))
    epochs_range = range(1, epochs + 1)

    plt.plot(epochs_range, resnet_results['train_accs'], 'b-',
             linewidth=2, label='ResNet Train Accuracy')
    plt.plot(epochs_range, resnet_results['test_accs'], 'b--',
             linewidth=2, label='ResNet Test Accuracy')
    plt.plot(epochs_range, googlenet_results['train_accs'], 'r-',
             linewidth=2, label='GoogLeNet Train Accuracy')
    plt.plot(epochs_range, googlenet_results['test_accs'], 'r--',
             linewidth=2, label='GoogLeNet Test Accuracy')

    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Training and Testing Accuracy Comparison', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')


def print_comparison_table(resnet_results, googlenet_results, epochs):
    """打印详细比较表格"""
    print(f"{'=' * 50}")
    print(f"{'Epoch':<6} {'ResNet Train':<15} {'ResNet Test':<15} {'GoogLeNet Train':<18} {'GoogLeNet Test':<15}")
    print(f"{'':<6} {'Loss/Acc':<15} {'Loss/Acc':<15} {'Loss/Acc':<18} {'Loss/Acc':<15}")
    print(f"{'-' * 80}")

    for epoch in range(epochs):
        print(f"{epoch + 1:<6} "
              f"{resnet_results['train_losses'][epoch]:.4f}/{resnet_results['train_accs'][epoch]:.1f}%  "
              f"{resnet_results['test_losses'][epoch]:.4f}/{resnet_results['test_accs'][epoch]:.1f}%  "
              f"{googlenet_results['train_losses'][epoch]:.4f}/{googlenet_results['train_accs'][epoch]:.1f}%  "
              f"{googlenet_results['test_losses'][epoch]:.4f}/{googlenet_results['test_accs'][epoch]:.1f}%")

    print(f"{'-' * 80}")


def main():
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 加载MNIST数据集
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 训练参数
    epochs = 10

    # 训练ResNet
    print("\nInitializing ResNet model...")
    resnet = ResNet()
    resnet_results = train_model(resnet, "ResNet", train_loader, test_loader, epochs, device)

    # 训练GoogLeNet
    print("\nInitializing GoogLeNet model...")
    googlenet = GoogLeNet()
    googlenet_results = train_model(googlenet, "GoogLeNet", train_loader, test_loader, epochs, device)

    # 性能比较
    print("\nModel Performance Comparison")
    print(f"{'=' * 50}")

    print(f"\nResNet Final Performance:")
    print(f"  Train Loss: {resnet_results['train_losses'][-1]:.4f}, "
          f"Train Acc: {resnet_results['train_accs'][-1]:.2f}%")
    print(f"  Test Loss: {resnet_results['test_losses'][-1]:.4f}, "
          f"Test Acc: {resnet_results['test_accs'][-1]:.2f}%")

    print(f"\nGoogLeNet Final Performance:")
    print(f"  Train Loss: {googlenet_results['train_losses'][-1]:.4f}, "
          f"Train Acc: {googlenet_results['train_accs'][-1]:.2f}%")
    print(f"  Test Loss: {googlenet_results['test_losses'][-1]:.4f}, "
          f"Test Acc: {googlenet_results['test_accs'][-1]:.2f}%")

    # 可视化
    plot_loss_comparison(resnet_results, googlenet_results, epochs)
    plot_accuracy_comparison(resnet_results, googlenet_results, epochs)

    # 打印详细比较表格
    print_comparison_table(resnet_results, googlenet_results, epochs)

    print("\nVisualization files:")
    print("  - loss_comparison.png")
    print("  - accuracy_comparison.png")


if __name__ == "__main__":
    main()