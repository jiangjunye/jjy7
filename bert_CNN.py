import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import time
from tqdm import tqdm


# ==================== GoogleNet 模型 ====================
class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, ch1x1, kernel_size=1),
            nn.BatchNorm2d(ch1x1),
            nn.ReLU(inplace=True)
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3red, kernel_size=1),
            nn.BatchNorm2d(ch3x3red),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch3x3red, ch3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch3x3),
            nn.ReLU(inplace=True)
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, ch5x5red, kernel_size=1),
            nn.BatchNorm2d(ch5x5red),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch5x5red, ch5x5, kernel_size=5, padding=2),
            nn.BatchNorm2d(ch5x5),
            nn.ReLU(inplace=True)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
            nn.BatchNorm2d(pool_proj),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)


class GoogleNet(nn.Module):
    def __init__(self, num_classes=10):
        super(GoogleNet, self).__init__()

        self.pre_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.inception3a = Inception(64, 16, 16, 32, 4, 8, 8)
        self.inception3b = Inception(64, 24, 24, 48, 6, 12, 12)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception4a = Inception(96, 32, 32, 64, 8, 16, 16)
        self.inception4b = Inception(128, 48, 48, 96, 12, 24, 24)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(192, num_classes)

    def forward(self, x):
        x = self.pre_layers(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool(x)
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())


# ==================== ResNet 模型 ====================
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()

        self.in_channels = 64

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels

        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


# ==================== 修复的ModelTrainer类 ====================
class ModelTrainer:
    def __init__(self, model, model_name, device):
        self.model = model
        self.model_name = model_name
        self.device = device
        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []
        self.training_time = 0
        self.all_preds = []
        self.all_targets = []

    def train(self, train_loader, val_loader, num_epochs=10, lr=0.001):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

        print(f"\n{'=' * 60}")
        print(f"Training {self.model_name}")
        print(f"Number of parameters: {self.model.get_num_params():,}")
        print(f"{'=' * 60}")

        start_time = time.time()

        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')
            for batch_idx, (data, target) in enumerate(pbar):
                data, target = data.to(self.device), target.to(self.device)

                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

                pbar.set_postfix({
                    'Loss': f'{running_loss / (batch_idx + 1):.4f}',
                    'Acc': f'{100. * correct / total:.2f}%'
                })

            train_loss = running_loss / len(train_loader)
            train_acc = 100. * correct / total
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)

            # Validation phase - 修复这里：只接收前两个返回值
            val_loss, val_acc = self.evaluate(val_loader, criterion)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)

            scheduler.step()

            print(f'Epoch {epoch + 1}/{num_epochs}:')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print(f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
            print('-' * 50)

        self.training_time = time.time() - start_time
        print(f"Training completed in {self.training_time:.2f} seconds")

    def evaluate(self, loader, criterion=None):
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)

                if criterion is not None:
                    val_loss += criterion(output, target).item()

                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

        # 保存预测结果
        self.all_preds = all_preds
        self.all_targets = all_targets

        if criterion is not None:
            val_loss = val_loss / len(loader)
        else:
            val_loss = 0

        val_acc = 100. * correct / total

        return val_loss, val_acc


# ==================== 可视化函数 ====================
def plot_training_history(googlenet_trainer, resnet_trainer):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    epochs = range(1, len(googlenet_trainer.train_losses) + 1)

    # 1. Loss曲线
    axes[0, 0].plot(epochs, googlenet_trainer.train_losses, 'b-', linewidth=2, label='GoogleNet Train')
    axes[0, 0].plot(epochs, googlenet_trainer.val_losses, 'b--', linewidth=2, label='GoogleNet Val')
    axes[0, 0].plot(epochs, resnet_trainer.train_losses, 'r-', linewidth=2, label='ResNet Train')
    axes[0, 0].plot(epochs, resnet_trainer.val_losses, 'r--', linewidth=2, label='ResNet Val')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Accuracy曲线
    axes[0, 1].plot(epochs, googlenet_trainer.train_accs, 'b-', linewidth=2, label='GoogleNet Train')
    axes[0, 1].plot(epochs, googlenet_trainer.val_accs, 'b--', linewidth=2, label='GoogleNet Val')
    axes[0, 1].plot(epochs, resnet_trainer.train_accs, 'r-', linewidth=2, label='ResNet Train')
    axes[0, 1].plot(epochs, resnet_trainer.val_accs, 'r--', linewidth=2, label='ResNet Val')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. 最终准确率对比
    models = ['GoogleNet', 'ResNet']
    final_accs = [googlenet_trainer.val_accs[-1], resnet_trainer.val_accs[-1]]
    final_losses = [googlenet_trainer.val_losses[-1], resnet_trainer.val_losses[-1]]

    bars1 = axes[0, 2].bar(models, final_accs, color=['blue', 'red'], alpha=0.7)
    axes[0, 2].set_ylabel('Final Validation Accuracy (%)')
    axes[0, 2].set_title('Final Accuracy Comparison')
    axes[0, 2].grid(True, alpha=0.3)

    for bar, acc in zip(bars1, final_accs):
        axes[0, 2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                        f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')

    # 4. 训练时间对比
    training_times = [googlenet_trainer.training_time, resnet_trainer.training_time]
    bars2 = axes[1, 0].bar(models, training_times, color=['blue', 'red'], alpha=0.7)
    axes[1, 0].set_ylabel('Training Time (seconds)')
    axes[1, 0].set_title('Training Efficiency')
    axes[1, 0].grid(True, alpha=0.3)

    for bar, time_val in zip(bars2, training_times):
        axes[1, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                        f'{time_val:.1f}s', ha='center', va='bottom')

    # 5. 参数量对比
    param_counts = [googlenet_trainer.model.get_num_params(), resnet_trainer.model.get_num_params()]
    bars3 = axes[1, 1].bar(models, [p / 1e6 for p in param_counts], color=['blue', 'red'], alpha=0.7)
    axes[1, 1].set_ylabel('Parameters (Millions)')
    axes[1, 1].set_title('Model Complexity')
    axes[1, 1].grid(True, alpha=0.3)

    for bar, params in zip(bars3, param_counts):
        axes[1, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                        f'{params / 1e6:.2f}M', ha='center', va='bottom')

    # 6. 性能对比总结
    axes[1, 2].axis('off')
    axes[1, 2].text(0.1, 0.9, 'Performance Summary:', fontsize=14, fontweight='bold')

    summary_text = f"""
GoogleNet:
  Final Accuracy: {final_accs[0]:.2f}%
  Final Loss: {final_losses[0]:.4f}
  Parameters: {param_counts[0]:,}
  Training Time: {training_times[0]:.1f}s

ResNet-18:
  Final Accuracy: {final_accs[1]:.2f}%
  Final Loss: {final_losses[1]:.4f}
  Parameters: {param_counts[1]:,}
  Training Time: {training_times[1]:.1f}s

Winner:
  Accuracy: {'GoogleNet' if final_accs[0] > final_accs[1] else 'ResNet'}
  Efficiency: {'GoogleNet' if training_times[0] < training_times[1] else 'ResNet'}
  Complexity: {'GoogleNet' if param_counts[0] < param_counts[1] else 'ResNet'}
    """

    axes[1, 2].text(0.1, 0.7, summary_text, fontsize=11,
                    verticalalignment='top', family='monospace')

    plt.tight_layout()
    plt.savefig('mnist_comparison_results.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_confusion_matrices(googlenet_trainer, resnet_trainer, val_loader):
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    # 获取GoogleNet的预测结果
    _, _, googlenet_preds, googlenet_targets = googlenet_trainer.evaluate_with_predictions(val_loader)

    # 获取ResNet的预测结果
    _, _, resnet_preds, resnet_targets = resnet_trainer.evaluate_with_predictions(val_loader)

    # 创建混淆矩阵
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    cm_googlenet = confusion_matrix(googlenet_targets, googlenet_preds)
    sns.heatmap(cm_googlenet, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title(f'GoogleNet Confusion Matrix\nAccuracy: {googlenet_trainer.val_accs[-1]:.2f}%')
    axes[0].set_xlabel('Predicted Label')
    axes[0].set_ylabel('True Label')

    cm_resnet = confusion_matrix(resnet_targets, resnet_preds)
    sns.heatmap(cm_resnet, annot=True, fmt='d', cmap='Reds', ax=axes[1])
    axes[1].set_title(f'ResNet-18 Confusion Matrix\nAccuracy: {resnet_trainer.val_accs[-1]:.2f}%')
    axes[1].set_xlabel('Predicted Label')
    axes[1].set_ylabel('True Label')

    plt.tight_layout()
    plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.show()


# ==================== 主程序 ====================
def main():
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 加载数据集
    print("\nLoading MNIST dataset...")
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    val_dataset = datasets.MNIST('./data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)

    # 初始化模型
    googlenet = GoogleNet().to(device)
    resnet = ResNet18().to(device)

    # 创建训练器
    googlenet_trainer = ModelTrainer(googlenet, "GoogleNet", device)
    resnet_trainer = ModelTrainer(resnet, "ResNet-18", device)

    # 训练GoogleNet
    googlenet_trainer.train(train_loader, val_loader, num_epochs=1, lr=0.001)

    # 训练ResNet
    resnet_trainer.train(train_loader, val_loader, num_epochs=1, lr=0.001)

    # 可视化结果
    print("\nGenerating visualizations...")
    plot_training_history(googlenet_trainer, resnet_trainer)

    # 保存模型
    print("\nSaving models...")
    torch.save(googlenet.state_dict(), 'googlenet_mnist.pth')
    torch.save(resnet.state_dict(), 'resnet_mnist.pth')

    # 打印对比结果
    print("\n" + "=" * 60)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<20} {'GoogleNet':<15} {'ResNet-18':<15} {'Winner':<10}")
    print("-" * 60)
    print(f"{'Final Accuracy':<20} {googlenet_trainer.val_accs[-1]:<15.2f}% "
          f"{resnet_trainer.val_accs[-1]:<15.2f}% "
          f"{'GoogleNet' if googlenet_trainer.val_accs[-1] > resnet_trainer.val_accs[-1] else 'ResNet'}")
    print(f"{'Final Loss':<20} {googlenet_trainer.val_losses[-1]:<15.4f} "
          f"{resnet_trainer.val_losses[-1]:<15.4f} "
          f"{'GoogleNet' if googlenet_trainer.val_losses[-1] < resnet_trainer.val_losses[-1] else 'ResNet'}")
    print(f"{'Training Time':<20} {googlenet_trainer.training_time:<15.1f}s "
          f"{resnet_trainer.training_time:<15.1f}s "
          f"{'GoogleNet' if googlenet_trainer.training_time < resnet_trainer.training_time else 'ResNet'}")
    print(f"{'Parameters':<20} {googlenet.get_num_params():<15,} "
          f"{resnet.get_num_params():<15,} "
          f"{'GoogleNet' if googlenet.get_num_params() < resnet.get_num_params() else 'ResNet'}")
    print("=" * 60)

    print("\nTraining completed successfully!")
    print("Results saved to 'mnist_comparison_results.png'")


if __name__ == "__main__":
    main()