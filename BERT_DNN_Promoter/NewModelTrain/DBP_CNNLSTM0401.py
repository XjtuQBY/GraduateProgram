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


# 定义iProLModel模型
class iProLModel(nn.Module):
    def __init__(self, seq_len=246):
        super(iProLModel, self).__init__()

        # CNN层和BatchNorm层（根据提供的模型结构，但调整了输入参数）
        # CNN层1
        self.conv1 = nn.Conv1d(in_channels=768, out_channels=256, kernel_size=8, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(256)
        self.maxpool1 = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.dropout1 = nn.Dropout(p=0.5)

        # CNN层2
        self.conv2 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=8, stride=2, padding=3)
        self.bn2 = nn.BatchNorm1d(128)
        self.maxpool2 = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.dropout2 = nn.Dropout(p=0.5)

        # CNN层3
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.maxpool3 = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)

        # 计算经过三层卷积和池化后的序列长度
        # 初始长度为seq_len
        # 第一层卷积：seq_len -> (seq_len + 2*padding - kernel_size) / stride + 1 = (seq_len + 2*3 - 8) / 2 + 1 = (seq_len - 2) / 2 + 1
        # 第一层池化：len1 -> len1 + 2*padding - kernel_size + 1 = len1 + 2*1 - 3 + 1 = len1
        length_after_conv1 = (seq_len - 2) // 2 + 1
        length_after_pool1 = length_after_conv1

        # 第二层卷积：length_after_pool1 -> (length_after_pool1 + 2*padding - kernel_size) / stride + 1
        length_after_conv2 = (length_after_pool1 + 2 * 3 - 8) // 2 + 1
        length_after_pool2 = length_after_conv2

        # 第三层卷积：length_after_pool2 -> (length_after_pool2 + 2*padding - kernel_size) / stride + 1
        length_after_conv3 = (length_after_pool2 + 2 * 1 - 3) // 2 + 1
        self.final_seq_len = length_after_conv3

        # BiLSTM层
        self.bilstm = nn.LSTM(input_size=64, hidden_size=64,num_layers=2, bidirectional=True, batch_first=True,dropout=0.3)

        # 全连接层
        self.fc1 = nn.Linear(128, 128)  # 64是因为BiLSTM的输出是32*2=64（双向）
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)

        # 激活函数
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 输入x的形状为 [batch_size, seq_len, 768]
        # 需要转置为 [batch_size, 768, seq_len] 以适应PyTorch的Conv1d
        x = x.transpose(1, 2)  # 形状变为 [batch_size, 768, seq_len]

        # CNN层1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.dropout1(x)

        # CNN层2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool2(x)
        x = self.dropout2(x)

        # CNN层3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.maxpool3(x)

        # 转置回 [batch_size, seq_len, features] 以适应LSTM
        x = x.transpose(1, 2)

        # BiLSTM层
        x, _ = self.bilstm(x)

        # 获取最后一个时间步的输出
        x = x[:, -1, :]

        # 全连接层
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.sigmoid(self.fc4(x))

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
            torch.save(model.state_dict(), 'iProL_Arab_TATA.pth')
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

    # 获取样本形状
    sample_x, _ = next(iter(train_loader))
    seq_len = sample_x.shape[1]
    print(f"Sequence length: {seq_len}")

    # 创建模型实例
    model = iProLModel(seq_len=seq_len).to(device)  # 将模型移动到指定设备（GPU/CPU）

    # 计算模型参数数量，打印参数统计信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # 损失函数和优化器
    criterion = nn.BCELoss()  # 使用二元交叉熵损失
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # 使用Adam优化器，带L2正则化

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
    model.load_state_dict(torch.load('iProL_Arab_TATA.pth'))

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
    model_scripted.save('iprol_model.pt')  # 保存为TorchScript模型
    print("Saved scripted model for deployment.")