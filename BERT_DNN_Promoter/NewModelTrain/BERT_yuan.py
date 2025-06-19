#!/usr/bin/env python
# coding: utf-8

import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.init as init
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torchmetrics
import os
import math
import json

# 设置基本配置
base_dir = "/data/home/fbchou/deepLearning/BERT_DNN_Promoter/dataset"

# 需要处理的数据集列表
#datasets = ["Mouse_non_tata"]
datasets = ["C_JEJUNI_1", "C_JEJUNI_2", "C_JEJUNI_3", "C_JEJUNI_4", "C_JEJUNI_5",
            "CPNEUMONIAE", "ECOLI_1", "ECOLI_2", "HPYLORI_1", "HPYLORI_2",
            "LINTERROGANS", "SCOELICOLOR", "SONEIDENSIS", "SPYOGENE", "STYPHIRMURIUM"]


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
X_train, X_temp, y_train, y_temp = train_test_split(X_trn, y_trn, test_size=0.25, random_state=416)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=416)

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
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# 打印一个样本的形状以验证
sample_x, sample_y = next(iter(train_loader))
print(f"Batch shape: {sample_x.shape}, Label shape: {sample_y.shape}")


# 定义多头自注意力机制模块
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim必须能被num_heads整除"

        # 定义查询、键、值的线性映射
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, x):
        # x形状: [batch_size, seq_len, embed_dim]
        batch_size, seq_len, embed_dim = x.shape

        # 计算查询、键、值
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 重塑为多头形式
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 应用注意力权重
        out = torch.matmul(attn_weights, v)

        # 重塑回原始形状
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        out = self.out_proj(out)

        return out


# 定义位置编码模块
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x形状: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return x


# 特征聚合模块
class FeatureAggregationModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        # 注意力加权池化
        self.attention_pool = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // 4, 1),
            nn.BatchNorm1d(in_channels // 4),
            nn.ReLU(),
            nn.Conv1d(in_channels // 4, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [batch_size, channels, seq_len]
        # 最大池化
        max_feat = self.global_max_pool(x).squeeze(-1)  # [batch_size, channels]

        # 平均池化
        avg_feat = self.global_avg_pool(x).squeeze(-1)  # [batch_size, channels]

        # 注意力加权池化
        attn_weights = self.attention_pool(x)  # [batch_size, 1, seq_len]
        attn_feat = torch.sum(x * attn_weights, dim=2)  # [batch_size, channels]

        # 拼接所有特征
        return torch.cat([max_feat, avg_feat, attn_feat], dim=1)  # [batch_size, channels*3]


# 完整的增强型DNABERT启动子模型
class EnhancedDNABERTPromoterModel(nn.Module):
    def __init__(self, input_channels=768, num_classes=1):
        super(EnhancedDNABERTPromoterModel, self).__init__()

        # 核心参数
        hidden_dim = 256
        kernel_sizes = [3, 5, 7]  # 多尺度卷积核
        num_filters = 128  # 每种卷积核的滤波器数量
        dropout_rate = 0.4  # 防止过拟合
        num_heads = 4  # 注意力头数量

        # 降维卷积 - 使用1x1卷积降低通道维度
        self.dim_reduction = nn.Sequential(
            nn.Conv1d(input_channels, hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )

        # 位置编码
        self.pos_encoding = PositionalEncoding(hidden_dim)

        # 自注意力层
        self.self_attention = MultiHeadSelfAttention(hidden_dim, num_heads=num_heads)

        # 多尺度CNN模块 - 并行捕获不同长度的模式
        self.conv_modules = nn.ModuleList()
        for kernel_size in kernel_sizes:
            padding = (kernel_size - 1) // 2
            module = nn.Sequential(
                nn.Conv1d(hidden_dim, num_filters, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm1d(num_filters),
                nn.ReLU(),
                nn.Conv1d(num_filters, num_filters, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm1d(num_filters),
                nn.ReLU()
            )
            self.conv_modules.append(module)

        # 合并后的通道数
        combined_channels = num_filters * len(kernel_sizes)

        # 特征聚合模块
        self.feature_aggregation = FeatureAggregationModule(combined_channels)

        # 分类器 - 使用聚合后的特征
        self.classifier = nn.Sequential(
            nn.Linear(combined_channels * 3, 256),  # *3 是因为有3种池化方法
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes),
            nn.Sigmoid()
        )

        # 初始化权重
        self._initialize_weights()

    def forward(self, x):
        # x形状: [batch_size, seq_len, input_channels]
        batch_size, seq_len, _ = x.shape

        # 转换为卷积所需的形状: [batch_size, input_channels, seq_len]
        x = x.transpose(1, 2)  # [batch_size, input_channels, seq_len]

        # 降维
        x = self.dim_reduction(x)  # [batch_size, hidden_dim, seq_len]

        # 为自注意力机制转置
        x_attn = x.transpose(1, 2)  # [batch_size, seq_len, hidden_dim]

        # 应用位置编码
        x_attn = self.pos_encoding(x_attn)  # [batch_size, seq_len, hidden_dim]

        # 应用自注意力
        x_attn = self.self_attention(x_attn)  # [batch_size, seq_len, hidden_dim]

        # 转回卷积格式
        x = x_attn.transpose(1, 2)  # [batch_size, hidden_dim, seq_len]

        # 通过多尺度卷积模块
        conv_outputs = []
        for conv_module in self.conv_modules:
            conv_outputs.append(conv_module(x))

        # 合并不同卷积核的输出
        combined = torch.cat(conv_outputs, dim=1)  # [batch_size, combined_channels, seq_len]

        # 应用特征聚合
        aggregated_features = self.feature_aggregation(combined)  # [batch_size, combined_channels*3]

        # 分类
        output = self.classifier(aggregated_features)

        return output

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)


# Focal Dice Loss实现
class FocalDiceLoss(nn.Module):
    def __init__(self, alpha=0.7, gamma=2.0, beta=0.3):
        super(FocalDiceLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta  # Dice损失权重
        self.eps = 1e-6  # 避免数值不稳定

    def forward(self, inputs, targets):
        # 将输入和目标调整为1D
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1)

        # Focal Loss部分
        # 计算 BCE loss
        BCE_loss = F.binary_cross_entropy(inputs_flat, targets_flat, reduction='none')

        # 计算 pt，即预测正确类别的概率
        pt = torch.where(targets_flat == 1, inputs_flat, 1 - inputs_flat)

        # 应用 alpha 权重
        alpha_weight = torch.where(targets_flat == 1, self.alpha, 1 - self.alpha)

        # 计算调制因子 (1-pt)^gamma
        modulating_factor = torch.pow(1.0 - pt, self.gamma)

        # 计算最终的 Focal Loss
        focal_loss = alpha_weight * modulating_factor * BCE_loss
        focal_loss = focal_loss.mean()

        # Dice Loss部分
        intersection = torch.sum(inputs_flat * targets_flat)
        dice_loss = 1 - (2. * intersection + self.eps) / (torch.sum(inputs_flat) + torch.sum(targets_flat) + self.eps)

        # 组合损失
        return focal_loss * (1 - self.beta) + dice_loss * self.beta


# 寻找最佳阈值的函数
def find_optimal_threshold(model, val_loader, device):
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Finding optimal threshold"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            all_probs.append(outputs)
            all_labels.append(labels)

    all_probs = torch.cat(all_probs, dim=0).cpu().numpy().flatten()
    all_labels = torch.cat(all_labels, dim=0).cpu().numpy().flatten()

    # 初始化最佳F1和阈值
    best_f1 = 0
    best_threshold = 0.5

    # 尝试不同阈值
    thresholds = np.arange(0.2, 0.7, 0.01)
    for threshold in tqdm(thresholds, desc="Testing thresholds"):
        preds = (all_probs > threshold).astype(int)

        # 计算混淆矩阵元素
        tp = np.sum((preds == 1) & (all_labels == 1))
        fp = np.sum((preds == 1) & (all_labels == 0))
        fn = np.sum((preds == 0) & (all_labels == 1))

        # 计算F1分数
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    print(f"最佳阈值: {best_threshold:.3f}, 最佳F1: {best_f1:.4f}")
    return best_threshold, best_f1


# 修改后的训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=100, device='cuda',
                early_stopping_patience=5, find_threshold=True):
    # 初始化指标
    f1_metric = torchmetrics.F1Score(task="binary").to(device)
    auroc_metric = torchmetrics.AUROC(task="binary").to(device)
    conf_matrix_metric = torchmetrics.ConfusionMatrix(task="binary", num_classes=2).to(device)
    precision_metric = torchmetrics.Precision(task="binary").to(device)
    recall_metric = torchmetrics.Recall(task="binary").to(device)

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    val_f1s = []
    best_val_f1 = 0  # 使用F1分数进行早停
    epochs_without_improvement = 0
    best_threshold = 0.5

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
        all_probs = []  # 收集原始概率输出而不是二值化的预测
        all_labels = []

        with torch.no_grad():
            val_progress = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} Validation", ncols=100)

            for inputs, labels in val_progress:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                # 计算准确率 (使用0.5阈值)
                predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # 保存原始预测概率和标签用于计算指标
                all_probs.append(outputs)
                all_labels.append(labels)

                val_progress.set_postfix(loss=val_loss / len(val_progress),
                                         acc=correct / total)

        val_loss = val_loss / len(val_loader)
        val_acc = correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # 计算各种指标
        all_probs_tensor = torch.cat(all_probs, dim=0)
        all_labels_tensor = torch.cat(all_labels, dim=0)
        all_labels_int = all_labels_tensor.int()

        # 重置所有指标，确保计算准确性
        f1_metric.reset()
        auroc_metric.reset()
        conf_matrix_metric.reset()
        precision_metric.reset()
        recall_metric.reset()

        # 使用原始概率输出计算F1、AUROC、精确率和召回率
        val_f1 = f1_metric(all_probs_tensor, all_labels_int)
        val_auroc = auroc_metric(all_probs_tensor, all_labels_int)
        val_precision = precision_metric(all_probs_tensor, all_labels_int)
        val_recall = recall_metric(all_probs_tensor, all_labels_int)

        # 使用二值化预测计算混淆矩阵
        preds_binary = (all_probs_tensor > 0.5).int()
        val_conf_matrix = conf_matrix_metric(preds_binary, all_labels_int)

        # 保存F1分数
        val_f1s.append(val_f1.item())

        # 如果启用阈值优化，每5个epoch寻找一次最佳阈值
        if find_threshold and (epoch % 5 == 0 or epoch == num_epochs - 1):
            best_threshold, best_epoch_f1 = find_optimal_threshold(model, val_loader, device)

            # 如果优化的F1更好，更新
            if best_epoch_f1 > val_f1:
                print(f"使用优化阈值后F1提升: {val_f1:.4f} -> {best_epoch_f1:.4f}")
                val_f1 = torch.tensor(best_epoch_f1, device=device)

            # 重新计算混淆矩阵
            preds_binary = (all_probs_tensor > best_threshold).int()
            val_conf_matrix = conf_matrix_metric(preds_binary, all_labels_int)

        # 打印当前epoch的结果
        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Train Loss: {train_loss:.4f}, "
              f"Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val Acc: {val_acc:.4f}, "
              f"F1: {val_f1:.4f}, "
              f"AUROC: {val_auroc:.4f}, "
              f"Precision: {val_precision:.4f}, "
              f"Recall: {val_recall:.4f}")

        print(f"Validation Confusion Matrix (threshold={best_threshold:.3f}):\n{val_conf_matrix}")

        # 更新学习率
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_f1)
            print(f"Current LR: {optimizer.param_groups[0]['lr']:.6f}")
        else:
            scheduler.step()
            print(f"Current LR: {scheduler.get_last_lr()[0]:.6f}")

        # 早停机制 - 使用F1分数
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_without_improvement = 0
            # 保存模型时同时保存最佳阈值
            torch.save({
                'model_state_dict': model.state_dict(),
                'best_threshold': best_threshold,
                'epoch': epoch,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1.item()
            }, 'DNABERT_Promoter_model_YuanHe.pth')
            print(f"验证F1分数提高到 {best_val_f1:.4f}。模型已保存。最佳阈值: {best_threshold:.3f}")
        else:
            epochs_without_improvement += 1
            print(f"F1分数已连续 {epochs_without_improvement} 个epoch没有提高。当前最佳F1: {best_val_f1:.4f}")

        if epochs_without_improvement >= early_stopping_patience:
            print("早停机制激活。训练停止。")
            break

    # 保存训练历史
    history = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_acc': train_accs,
        'val_acc': val_accs,
        'val_f1': val_f1s,
        'best_threshold': best_threshold
    }

    return history, best_threshold


# 修改后的评估函数
# 接上面的evaluate_model函数
def evaluate_model(model, test_loader, criterion, device='cuda', threshold=0.5):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    # 初始化指标
    f1_metric = torchmetrics.F1Score(task="binary").to(device)
    auroc_metric = torchmetrics.AUROC(task="binary").to(device)
    conf_matrix_metric = torchmetrics.ConfusionMatrix(task="binary", num_classes=2).to(device)
    precision_metric = torchmetrics.Precision(task="binary").to(device)
    recall_metric = torchmetrics.Recall(task="binary").to(device)

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            # 计算准确率 (使用指定阈值)
            predicted = (outputs > threshold).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_probs.append(outputs)
            all_labels.append(labels)

    test_loss = test_loss / len(test_loader)
    test_acc = correct / total

    # 计算各种指标
    all_probs = torch.cat(all_probs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_labels_int = all_labels.int()

    # 重置所有指标
    f1_metric.reset()
    auroc_metric.reset()
    conf_matrix_metric.reset()
    precision_metric.reset()
    recall_metric.reset()

    # 使用原始概率输出计算AUROC
    auroc = auroc_metric(all_probs, all_labels_int)
    # 使用原始概率计算度量
    f1 = f1_metric(all_probs, all_labels_int)
    precision = precision_metric(all_probs, all_labels_int)
    recall = recall_metric(all_probs, all_labels_int)

    # 使用指定阈值的二值化预测计算其他指标
    all_preds_binary = (all_probs > threshold).int()
    conf_matrix = conf_matrix_metric(all_preds_binary, all_labels_int)

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test F1: {f1:.4f}")
    print(f"Test AUROC: {auroc:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Confusion Matrix (threshold={threshold:.3f}):\n{conf_matrix}")

    # 返回测试指标
    return {
        'loss': test_loss,
        'accuracy': test_acc,
        'f1': f1.item(),
        'auroc': auroc.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'confusion_matrix': conf_matrix.cpu().numpy(),
        'threshold': threshold
    }


# 主函数
if __name__ == "__main__":
    # 设置设备（如果有GPU则使用GPU，否则使用CPU）
    device = torch.device("cuda:1")
    print(f"Using device: {device}")

    # 创建模型实例
    model = EnhancedDNABERTPromoterModel().to(device)  # 将模型移动到指定设备（GPU/CPU）

    # 计算模型参数数量，打印参数统计信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # 损失函数 - 使用组合的FocalDiceLoss
    criterion = FocalDiceLoss(alpha=0.80, gamma=3.0, beta=0.6)
    #criterion = FocalDiceLoss(alpha=0.5, gamma=1.0, beta=0.2)

    # 打印损失函数信息
    print(f"Using FocalDiceLoss with alpha={criterion.alpha}, gamma={criterion.gamma}, beta={criterion.beta}")

    # 优化器，带权重衰减
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # 使用Adam优化器，带L2正则化

    # 学习率调度器 - 使用ReduceLROnPlateau代替CosineAnnealingLR
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )

    # 训练模型
    print("Starting model training...")
    history, best_threshold = train_model(
        model=model,  # 模型实例
        train_loader=train_loader,  # 训练集DataLoader
        val_loader=val_loader,  # 验证集DataLoader
        criterion=criterion,  # 损失函数
        optimizer=optimizer,  # 优化器
        scheduler=scheduler,  # 学习率调度器
        num_epochs=100,  # 最大训练轮数
        device=device,  # 设备（GPU或CPU）
        early_stopping_patience=10,  # 早停机制的耐心轮数
        find_threshold=True  # 启用阈值优化
    )

    # 加载训练过程中最好的模型
    print("Loading best model...")
    checkpoint = torch.load('DNABERT_Promoter_model_YuanHe.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    best_threshold = checkpoint['best_threshold']
    print(f"Loaded model from epoch {checkpoint['epoch']} with validation F1: {checkpoint['val_f1']:.4f}")
    print(f"Using best threshold: {best_threshold:.3f}")

    # 在测试集上评估模型
    print("Evaluating model on test set...")
    test_metrics = evaluate_model(model, test_loader, criterion, device, threshold=best_threshold)

    # 保存测试指标到文件
    print("Saving test metrics...")
    with open('test_metrics.json', 'w') as f:
        # 将numpy数组转换为列表，避免JSON不支持numpy类型
        metrics_json = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in test_metrics.items()}
        json.dump(metrics_json, f, indent=4)

    # 保存训练历史
    print("Saving training history...")
    with open('training_history.json', 'w') as f:
        # 将tensor转换为列表
        history_json = {k: [float(val) for val in v] if isinstance(v, list) else float(v)
                        for k, v in history.items()}
        json.dump(history_json, f, indent=4)

    print("Testing completed! Results saved.")

    # 保存模型用于推理，使用TorchScript优化的模型
    print("Saving optimized model for deployment...")
    model_scripted = torch.jit.script(model)  # 将模型脚本化
    model_scripted.save('enhanced_bert_promoter_model.pt')  # 保存为TorchScript模型
    print("Saved scripted model for deployment.")

    print("\nCompleted all tasks!")