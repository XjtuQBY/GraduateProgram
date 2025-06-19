#!/usr/bin/env python
# coding: utf-8

import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm  # 导入 tqdm 库
import torchmetrics  # 导入 torchmetrics

# 加载 HDF5 文件
promoter_file = "/data/home/fbchou/deepLearning/BERT_DNN_Promoter/dataset/3382/promoter_dataset.h5"
non_promoter_file = "/data/home/fbchou/deepLearning/BERT_DNN_Promoter/dataset/3382/non_promoter_dataset.h5"

with h5py.File(promoter_file, 'r') as hf:
    X_enhancer = hf['X_data'][:]  # 正样本特征
    y_enhancer = hf['y_labels'][:]  # 正样本标签

with h5py.File(non_promoter_file, 'r') as hf:
    X_non_enhancer = hf['X_data'][:]  # 负样本特征
    y_non_enhancer = hf['y_labels'][:]  # 负样本标签

# 合并正负样本数据
X_trn = np.concatenate([X_enhancer, X_non_enhancer], axis=0)
y_trn = np.concatenate([y_enhancer, y_non_enhancer], axis=0)

# 划分训练集、验证集和测试集
X_train, X_temp, y_train, y_temp = train_test_split(X_trn, y_trn, test_size=0.25, random_state=523)  # 75% 训练集，25% 剩余数据
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=523)  # 剩余的 50% 用作测试和验证


# PyTorch Dataset 和 DataLoader
class CustomDataset(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = torch.tensor(X_data.reshape(-1, 1, 768, 81), dtype=torch.float32)
        self.y_data = torch.tensor(y_data, dtype=torch.long)

    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, idx):
        return self.X_data[idx], self.y_data[idx]

# 创建数据集
train_dataset = CustomDataset(X_train, y_train)
val_dataset = CustomDataset(X_val, y_val)
test_dataset = CustomDataset(X_test, y_test)

# 数据加载器
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 打印数据集的一些关键信息
print(f"Training Dataset Size: {len(train_dataset)} samples")
print(f"Validation Dataset Size: {len(val_dataset)} samples")
print(f"Test Dataset Size: {len(test_dataset)} samples")

# 打印训练集的第一个样本的形状和标签
sample_data, sample_label = train_dataset[0]
print(f"Shape of one sample in training dataset: {sample_data.shape}")
print(f"Label of one sample in training dataset: {sample_label}")

#---------------------------------------------------------------------------------------------------------------------

# 设置device为GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 构建 CNN 模型
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        # 2D卷积层
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 自适应池化层
        self.adaptive_pool = nn.AdaptiveAvgPool2d((16, 16))

        # 全连接层
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return torch.softmax(x, dim=1)

# 初始化 torchmetrics 指标
output_size = 2  # 二分类任务
f1_metric = torchmetrics.classification.F1Score(num_classes=output_size, average=None, task="binary").to(device)
conf_matrix_metric = torchmetrics.ConfusionMatrix(num_classes=output_size, task="binary").to(device)

# 训练模型
num_epochs = 50
train_losses = []
val_losses = []

# 早停机制相关参数
early_stopping_patience = 8
best_val_loss = float('inf')
epochs_without_improvement = 0

model = CNNModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 创建余弦退火学习率调度器
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=early_stopping_patience)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # 使用 tqdm 来显示训练进度
    train_progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} Train", ncols=100)

    for inputs, labels in train_progress:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        train_progress.set_postfix(loss=running_loss / len(train_progress), accuracy=correct / total)

    train_loss = running_loss / len(train_loader)
    train_acc = correct / total
    train_losses.append(train_loss)

    scheduler.step()  # 更新学习率
    current_lr = scheduler.get_last_lr()[0]

    # 评估模型在验证集上的性能
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    f1_metric.reset()
    conf_matrix_metric.reset()

    with torch.no_grad():
        val_progress = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} Validation", ncols=100)

        for inputs, labels in val_progress:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 更新 F1 分数和混淆矩阵
            f1_metric.update(predicted, labels)
            conf_matrix_metric.update(predicted, labels)

            val_progress.set_postfix(loss=val_loss / len(val_progress), accuracy=correct / total)

    val_loss = val_loss / len(val_loader)
    val_acc = correct / total
    val_losses.append(val_loss)

    # 计算 F1 分数和混淆矩阵
    f1_score = f1_metric.compute()
    val_conf_matrix = conf_matrix_metric.compute()

    print(
        f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, "
        f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}, Learning Rate: {current_lr:.6f}")

    # 打印 F1 分数和混淆矩阵
    print(f"F1 Score: {f1_score:.4f}")
    print("Confusion Matrix:\n", val_conf_matrix)

    # 早停机制
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_without_improvement = 0
        torch.save(model.state_dict(), 'best_model.pth')
        print("Validation loss improved. Model saved.")
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= early_stopping_patience:
        print("Early stopping activated. Training stopped.")
        break

# 测试集评估
model.eval()
test_loss = 0.0
correct = 0
total = 0

f1_metric.reset()
conf_matrix_metric.reset()

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # 更新 F1 分数和混淆矩阵
        f1_metric.update(predicted, labels)
        conf_matrix_metric.update(predicted, labels)

test_loss = test_loss / len(test_loader)
test_acc = correct / total

# 计算 F1 分数和混淆矩阵
f1_score = f1_metric.compute()
test_conf_matrix = conf_matrix_metric.compute()

# 输出测试集的准确率和损失
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
print(f"Test F1 Score: {f1_score:.4f}")
print("Test Confusion Matrix:\n", test_conf_matrix)
