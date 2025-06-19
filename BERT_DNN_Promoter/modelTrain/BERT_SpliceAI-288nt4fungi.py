#!/usr/bin/env python
# coding: utf-8

import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.init as init
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torchmetrics
import os

# 设置基本配置
base_dir = "/data/home/fbchou/deepLearning/BERT_DNN_Promoter/dataset"

# 需要处理的数据集列表
datasets = ["Arabidopsis_tata"]

# 初始化空数组，用于存储合并的数据
X_all_promoter = []
y_all_promoter = []
X_all_non_promoter = []
y_all_non_promoter = []

# 遍历每个数据集并加载数据
for dataset in tqdm(datasets, desc="Loading datasets"):
    promoter_file = os.path.join(base_dir, dataset, "promoter_dataset.h5")
    non_promoter_file = os.path.join(base_dir, dataset, "non_promoter_dataset.h5")

    # 检查文件是否存在
    if not os.path.exists(promoter_file) or not os.path.exists(non_promoter_file):
        print(f"Warning: Files for {dataset} not found, skipping.")
        continue

    try:
        # 加载促进子数据
        with h5py.File(promoter_file, 'r') as hf:
            X_promoter = hf['X_data'][:]
            y_promoter = hf['y_labels'][:]
            print(f"  y_promoter unique values: {np.unique(y_promoter)}")

        # 加载非促进子数据
        with h5py.File(non_promoter_file, 'r') as hf:
            X_non_promoter = hf['X_data'][:]
            y_non_promoter = hf['y_labels'][:]
            print(f"  y_non_promoter unique values: {np.unique(y_non_promoter)}")

        # 添加到总数据列表
        X_all_promoter.append(X_promoter)
        y_all_promoter.append(y_promoter)
        X_all_non_promoter.append(X_non_promoter)
        y_all_non_promoter.append(y_non_promoter)

        # 打印当前数据集的形状信息
        print(f"Dataset: {dataset}")
        print(f"  X_promoter shape: {X_promoter.shape}")
        print(f"  X_non_promoter shape: {X_non_promoter.shape}")

    except Exception as e:
        print(f"Error loading {dataset}: {str(e)}")

# 检查是否有成功加载的数据
if len(X_all_promoter) == 0 or len(X_all_non_promoter) == 0:
    raise ValueError("No valid datasets were loaded.")

# 合并所有数据集
X_promoter_combined = np.concatenate(X_all_promoter, axis=0)
y_promoter_combined = np.concatenate(y_all_promoter, axis=0)
X_non_promoter_combined = np.concatenate(X_all_non_promoter, axis=0)
y_non_promoter_combined = np.concatenate(y_all_non_promoter, axis=0)
# 合并所有数据集后
print(f"Combined y_promoter unique values: {np.unique(y_promoter_combined)}")
print(f"Combined y_non_promoter unique values: {np.unique(y_non_promoter_combined)}")

# 合并正负样本数据
X_trn = np.concatenate([X_promoter_combined, X_non_promoter_combined], axis=0)
y_trn = np.concatenate([y_promoter_combined, y_non_promoter_combined], axis=0)

# 打印合并后的数据集的形状信息
print(f"Combined X_promoter shape: {X_promoter_combined.shape}")
print(f"Combined X_non_promoter shape: {X_non_promoter_combined.shape}")
print(f"Total combined data shape: {X_trn.shape}")
print(f"Total combined labels shape: {y_trn.shape}")
print(f"Unique labels: {np.unique(y_trn)}")

# 划分训练集、验证集和测试集
X_train, X_temp, y_train, y_temp = train_test_split(X_trn, y_trn, test_size=0.25, random_state=523)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=523)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Validation set: {X_val.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")


# 自定义Dataset类
class BERTDataset(Dataset):
    def __init__(self, X_data, y_data):
        # X_data形状为 [n_samples, seq_len, 768]
        self.X_data = torch.tensor(X_data, dtype=torch.float32)
        self.y_data = torch.tensor(y_data, dtype=torch.float32).view(-1, 1)  # 确保y_data是二维的[n_samples, 1]

    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, idx):
        return self.X_data[idx], self.y_data[idx]


# 创建数据集
train_dataset = BERTDataset(X_train, y_train)
val_dataset = BERTDataset(X_val, y_val)
test_dataset = BERTDataset(X_test, y_test)

# 创建数据加载器
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# 打印一个样本的形状以验证
sample_x, sample_y = next(iter(train_loader))
print(f"Batch shape: {sample_x.shape}, Label shape: {sample_y.shape}")


# 定义残差块
class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size, padding=(kernel_size - 1) // 2 * dilation, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size, padding=(kernel_size - 1) // 2 * dilation, dilation=dilation)
        )

    def forward(self, x):
        return x + self.conv_block(x)


# 定义SpliceAI模型
class BERTSpliceModel(nn.Module):
    def __init__(self, input_channels=768, num_classes=1):
        super(BERTSpliceModel, self).__init__()

        # 初始卷积层，从768通道转换到32通道
        self.input_conv = nn.Conv1d(input_channels, 32, kernel_size=1)

        # 第一组残差块：4个RB(32, 5, 1)
        self.rb_group1 = nn.ModuleList()
        for i in range(4):
            self.rb_group1.append(ResidualBlock(channels=32, kernel_size=5, dilation=1))

        # 跳跃连接卷积层1
        self.module1_skip = nn.Conv1d(32, 32, kernel_size=1)

        # 第二组残差块：4个RB(32, 5, 2)
        self.rb_group2 = nn.ModuleList()
        for i in range(4):
            self.rb_group2.append(ResidualBlock(channels=32, kernel_size=5, dilation=2))

        # 跳跃连接卷积层2
        self.module2_skip = nn.Conv1d(32, 32, kernel_size=1)

        # 第三组残差块：4个RB(32, 9, 3)
        self.rb_group3 = nn.ModuleList()
        for i in range(4):
            self.rb_group3.append(ResidualBlock(channels=32, kernel_size=9, dilation=3))

        # 跳跃连接卷积层3
        self.module3_skip = nn.Conv1d(32, 32, kernel_size=1)

        # RB后的一个卷积层
        self.afterRB_conv = nn.Conv1d(32, 32, kernel_size=1)

        # 最终卷积层
        self.final_conv = nn.Conv1d(32, num_classes, kernel_size=1)

        # 添加sigmoid激活函数 (方案1使用)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 输入形状转换：[batch_size, sequence_length, input_channels]
        # -> [batch_size, input_channels, sequence_length]
        x = x.transpose(1, 2)

        # 初始卷积
        x = self.input_conv(x)

        # 保存skip1连接
        skip1 = self.module1_skip(x)

        # 第一组残差块
        for rb in self.rb_group1:
            x = rb(x)

        # 保存skip2连接
        skip2 = self.module2_skip(x)

        # 第二组残差块
        for rb in self.rb_group2:
            x = rb(x)

        # x 右拐经过第三个conv后的值 记为 skip3
        skip3 = self.module3_skip(x)

        # x 向下直行经过第三组 rb 层
        for rb in self.rb_group3:
            x = rb(x)

        # x 向下直行经过 "RB后的一个卷积层"
        x = self.afterRB_conv(x)

        # 积累的 3 个skip值在此汇合，得到 result
        result = x + skip1 + skip2 + skip3

        # 最终卷积
        x = self.final_conv(x)

        # 转换回原始形状：[batch_size, sequence_length, num_classes]
        x = x.transpose(1, 2)

        # 添加全局平均池化，将sequence_length维度压缩
        x = torch.mean(x, dim=1, keepdim=True)  # [batch_size, 1, 1]
        x = x.squeeze(2)  # [batch_size, 1]

        # 应用sigmoid激活函数 (方案1使用)
        x = self.sigmoid(x)

        return x


# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=100, device='cuda',
                early_stopping_patience=5):
    # 初始化指标
    f1_metric = torchmetrics.F1Score(task="binary").to(device)
    auroc_metric = torchmetrics.AUROC(task="binary").to(device)
    conf_matrix_metric = torchmetrics.ConfusionMatrix(task="binary", num_classes=2).to(device)

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} Train", ncols=100)

        for inputs, labels in train_progress:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # 计算准确率
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            train_progress.set_postfix(loss=running_loss / len(train_progress),
                                       acc=correct / total)

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            val_progress = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} Validation", ncols=100)

            for inputs, labels in val_progress:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                # 计算准确率
                predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # 保存预测和标签用于计算指标
                all_preds.append(outputs)
                all_labels.append(labels)

                val_progress.set_postfix(loss=val_loss / len(val_progress),
                                         acc=correct / total)

        val_loss = val_loss / len(val_loader)
        val_acc = correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # 计算F1和AUROC
        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # 确保标签是整数类型
        all_labels_int = all_labels.int()

        # 二值化预测值
        preds_binary = (all_preds > 0.5).int()

        # 计算指标
        f1 = f1_metric(preds_binary, all_labels_int)
        auroc = auroc_metric(all_preds, all_labels_int)
        val_conf_matrix = conf_matrix_metric(preds_binary, all_labels_int)

        # 打印当前epoch的结果
        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Train Loss: {train_loss:.4f}, "
              f"Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val Acc: {val_acc:.4f}, "
              f"F1: {f1:.4f}, "
              f"AUROC: {auroc:.4f}, "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")

        print(f"Validation Confusion Matrix:\n{val_conf_matrix}")

        # 更新学习率
        scheduler.step()

        # 早停机制
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), 'best_BERT_SpliceAI_Arab_TATA_model.pth')
            print("Validation loss improved. Model saved.")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= early_stopping_patience:
            print("Early stopping activated. Training stopped.")
            break

    # 保存训练历史
    history = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_acc': train_accs,
        'val_acc': val_accs
    }

    return history


# 评估函数
def evaluate_model(model, test_loader, criterion, device='cuda'):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    f1_metric = torchmetrics.F1Score(task="binary").to(device)
    auroc_metric = torchmetrics.AUROC(task="binary").to(device)
    conf_matrix_metric = torchmetrics.ConfusionMatrix(task="binary", num_classes=2).to(device)

    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            # 计算准确率
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.append(predicted)
            all_probs.append(outputs)
            all_labels.append(labels)

    test_loss = test_loss / len(test_loader)
    test_acc = correct / total

    # 计算各种指标
    all_preds = torch.cat(all_preds, dim=0)
    all_probs = torch.cat(all_probs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # 确保标签是整数类型
    all_labels_int = all_labels.int()

    # 二值化预测值
    all_preds_binary = (all_preds > 0.5).int()

    # 计算指标
    f1 = f1_metric(all_preds_binary, all_labels_int)
    auroc = auroc_metric(all_probs, all_labels_int)
    conf_matrix = conf_matrix_metric(all_preds_binary, all_labels_int)

    # 计算精确率和召回率
    precision_metric = torchmetrics.Precision(task="binary").to(device)
    recall_metric = torchmetrics.Recall(task="binary").to(device)

    precision = precision_metric(all_preds_binary, all_labels_int)
    recall = recall_metric(all_preds_binary, all_labels_int)

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test F1: {f1:.4f}")
    print(f"Test AUROC: {auroc:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Confusion Matrix:\n{conf_matrix}")

    # 返回测试指标
    return {
        'loss': test_loss,
        'accuracy': test_acc,
        'f1': f1.item(),
        'auroc': auroc.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'confusion_matrix': conf_matrix.cpu().numpy()
    }


# 主函数
if __name__ == "__main__":
    # 设置设备（如果有GPU则使用GPU，否则使用CPU）
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 创建模型实例
    model = BERTSpliceModel().to(device)  # 将模型移动到指定设备（GPU/CPU）

    # 计算模型参数数量，打印参数统计信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # 损失函数和优化器
    criterion = nn.BCELoss()  # 使用二元交叉熵损失
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-6)  # 使用Adam优化器，带L2正则化

    # 学习率调度器 (CosineAnnealingLR是为了动态调整学习率)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    # 训练模型
    print("Starting model training...")
    history = train_model(
        model=model,  # 模型实例
        train_loader=train_loader,  # 训练集DataLoader
        val_loader=val_loader,  # 验证集DataLoader
        criterion=criterion,  # 损失函数
        optimizer=optimizer,  # 优化器
        scheduler=scheduler,  # 学习率调度器
        num_epochs=100,  # 最大训练轮数
        device=device,  # 设备（GPU或CPU）
        early_stopping_patience=10  # 早停机制的耐心轮数
    )

    # 加载训练过程中最好的模型（以最低验证损失为准）
    model.load_state_dict(torch.load('best_BERT_SpliceAI_Arab_TATA_model.pth'))

    # 在测试集上评估模型
    print("Evaluating model on test set...")
    test_metrics = evaluate_model(model, test_loader, criterion, device)

    # 保存测试指标到文件
    import json
    with open('test_metrics.json', 'w') as f:
        # 将numpy数组转换为列表，避免JSON不支持numpy类型
        metrics_json = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in test_metrics.items()}
        json.dump(metrics_json, f, indent=4)

    print("Testing completed! Results saved.")

    # 保存模型用于推理，使用TorchScript优化的模型
    model_scripted = torch.jit.script(model)  # 将模型脚本化
    model_scripted.save('bert_splice_model.pt')  # 保存为TorchScript模型
    print("Saved scripted model for deployment.")

