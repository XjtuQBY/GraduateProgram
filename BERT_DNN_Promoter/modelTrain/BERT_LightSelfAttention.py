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
import math

# 加载 HDF5 文件
promoter_file = "/data/home/fbchou/deepLearning/BERT_DNN_Promoter/dataset/Arabidopsis_tata/promoter_dataset.h5"
non_promoter_file = "/data/home/fbchou/deepLearning/BERT_DNN_Promoter/dataset/Arabidopsis_tata/non_promoter_dataset.h5"

with h5py.File(promoter_file, 'r') as hf:
    X_enhancer = hf['X_data'][:]  # 正样本特征 - 形状为 [样本数, 81, 768]
    y_enhancer = hf['y_labels'][:]  # 正样本标签

with h5py.File(non_promoter_file, 'r') as hf:
    X_non_enhancer = hf['X_data'][:]  # 负样本特征 - 形状为 [样本数, 81, 768]
    y_non_enhancer = hf['y_labels'][:]  # 负样本标签

# 合并正负样本数据
X_trn = np.concatenate([X_enhancer, X_non_enhancer], axis=0)
y_trn = np.concatenate([y_enhancer, y_non_enhancer], axis=0)

# 打印数据集的形状信息
print(f"X_enhancer shape: {X_enhancer.shape}")
print(f"X_non_enhancer shape: {X_non_enhancer.shape}")
print(f"Combined data shape: {X_trn.shape}")
print(f"Combined labels shape: {y_trn.shape}")
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


# 轻量级自注意力模块
class LightSelfAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(LightSelfAttention, self).__init__()
        self.query = nn.Conv1d(channels, channels // reduction, kernel_size=1)
        self.key = nn.Conv1d(channels, channels // reduction, kernel_size=1)
        self.value = nn.Conv1d(channels, channels, kernel_size=1)
        # 使用极小的初始gamma值
        self.gamma = nn.Parameter(torch.zeros(1) * 0.001)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, C, length = x.size()

        # 计算注意力权重
        proj_query = self.query(x).permute(0, 2, 1)
        proj_key = self.key(x)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)

        # 应用注意力权重
        proj_value = self.value(x)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))

        # 使用极小的gamma进行残差连接
        out = self.gamma * out + x

        return out


# 定义残差块 - 保持原样，不包含注意力
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
        self.dropout = nn.Dropout(0.1)  # 添加dropout

    def forward(self, x):
        return x + self.dropout(self.conv_block(x))


# 定义简化版SpliceAI模型
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

        # 添加轻量级自注意力模块 - 只在最终特征融合后应用
        self.light_attention = LightSelfAttention(32)

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

        # 在特征融合后，应用轻量级自注意力 - 这是添加注意力的关键位置
        result = self.light_attention(result)

        # 最终卷积
        x = self.final_conv(result)

        # 转换回原始形状：[batch_size, sequence_length, num_classes]
        x = x.transpose(1, 2)

        # 添加全局平均池化，将sequence_length维度压缩
        x = torch.mean(x, dim=1, keepdim=True)  # [batch_size, 1, 1]
        x = x.squeeze(2)  # [batch_size, 1]

        # 应用sigmoid激活函数 (方案1使用)
        x = self.sigmoid(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                # 使用更小的标准差初始化卷积层权重
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=100, device='cuda',
                early_stopping_patience=5):
    # 初始化指标 - 修复指标类型问题
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
            torch.save(model.state_dict(), 'best_BERT_LSA_model.pth')
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
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 创建模型
    model = BERTSpliceModel().to(device)
    # print("Model Architecture:")
    # print(model)

    # 计算模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # 损失函数和优化器 (方案1 - 使用BCELoss + sigmoid)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-6)

    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    # 训练模型
    print("Starting model training...")
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=100,
        device=device,
        early_stopping_patience=10
    )

    # 加载最佳模型
    model.load_state_dict(torch.load('best_BERT_LSA_model.pth'))

    # 在测试集上评估模型
    print("Evaluating model on test set...")
    test_metrics = evaluate_model(model, test_loader, criterion, device)

    # 保存测试指标
    import json

    with open('test_metrics.json', 'w') as f:
        # 将numpy数组转换为列表
        metrics_json = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in test_metrics.items()}
        json.dump(metrics_json, f, indent=4)

    print("Testing completed! Results saved.")

    # 保存模型用于推理
    model_scripted = torch.jit.script(model)
    model_scripted.save('bert_splice_model.pt')
    print("Saved scripted model for deployment.")