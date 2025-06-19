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
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# 打印一个样本的形状以验证
sample_x, sample_y = next(iter(train_loader))
print(f"Batch shape: {sample_x.shape}, Label shape: {sample_y.shape}")


# 定义注意力机制模块
class AttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(AttentionModule, self).__init__()
        self.query = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.key = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.value = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.scale = torch.sqrt(torch.FloatTensor([in_channels]))
        
    def forward(self, x):
        # 确保scale在正确的设备上
        self.scale = self.scale.to(x.device)
        
        # x形状: [batch_size, channels, seq_len]
        batch_size, channels, seq_len = x.shape
        
        # 计算查询、键、值投影
        q = self.query(x)  # [batch_size, channels, seq_len]
        k = self.key(x)    # [batch_size, channels, seq_len]
        v = self.value(x)  # [batch_size, channels, seq_len]
        
        # 变形用于矩阵乘法
        q = q.permute(0, 2, 1)  # [batch_size, seq_len, channels]
        k = k.permute(0, 2, 1)  # [batch_size, seq_len, channels]
        v = v.permute(0, 2, 1)  # [batch_size, seq_len, channels]
        
        # 计算注意力分数
        attention = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # [batch_size, seq_len, seq_len]
        attention = F.softmax(attention, dim=-1)
        
        # 将注意力应用于值
        output = torch.matmul(attention, v)  # [batch_size, seq_len, channels]
        output = output.permute(0, 2, 1)  # [batch_size, channels, seq_len]
        
        return output + x  # 残差连接

# 定义残差块 (您原始代码中已有)
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

# 修改后的BERTSpliceModel
class BERTSpliceModel(nn.Module):
    def __init__(self, input_channels=768, num_classes=1):
        super(BERTSpliceModel, self).__init__()

        # 基本通道数
        base_channels = 64

        # 初始卷积层
        self.input_conv = nn.Conv1d(input_channels, base_channels, kernel_size=1)
        
        # 添加注意力模块
        self.attention = AttentionModule(base_channels)

        # 第一组残差块
        self.rb_group1 = nn.ModuleList()
        for i in range(4):
            self.rb_group1.append(ResidualBlock(channels=base_channels, kernel_size=5, dilation=1))

        # 跳跃连接卷积层1
        self.module1_skip = nn.Conv1d(base_channels, base_channels, kernel_size=1)

        # 第二组残差块
        self.rb_group2 = nn.ModuleList()
        for i in range(4):
            self.rb_group2.append(ResidualBlock(channels=base_channels, kernel_size=5, dilation=2))

        # 跳跃连接卷积层2
        self.module2_skip = nn.Conv1d(base_channels, base_channels, kernel_size=1)

        # 第三组残差块
        self.rb_group3 = nn.ModuleList()
        for i in range(4):
            self.rb_group3.append(ResidualBlock(channels=base_channels, kernel_size=9, dilation=3))

        # 跳跃连接卷积层3
        self.module3_skip = nn.Conv1d(base_channels, base_channels, kernel_size=1)

        # RB后的一个卷积层
        self.afterRB_conv = nn.Conv1d(base_channels, base_channels, kernel_size=1)

        # 最终卷积层
        self.final_conv = nn.Conv1d(base_channels, base_channels, kernel_size=1)

        # 全连接层
        self.fc1 = nn.Linear(base_channels, 128)
        self.dropout1 = nn.Dropout(0.5)
        self.fc_out = nn.Linear(128, num_classes)

        # 激活函数
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 输入形状转换
        x = x.transpose(1, 2)

        # 初始卷积
        x = self.input_conv(x)
        
        # 应用注意力模块
        x = self.attention(x)

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

        # 保存skip3连接
        skip3 = self.module3_skip(x)

        # 第三组残差块
        for rb in self.rb_group3:
            x = rb(x)

        # RB后的一个卷积层
        x = self.afterRB_conv(x)

        # 组合所有skip连接
        x = x + skip1 + skip2 + skip3

        # 最终卷积层
        x = self.final_conv(x)

        # 全局最大池化
        x = torch.max(x, dim=2)[0]  # [batch_size, base_channels]

        # 全连接层
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc_out(x)

        # sigmoid激活
        x = self.sigmoid(x)

        return x

# Focal Loss实现
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.66, gamma=2.0, reduction='mean'):
        """
        实现 Focal Loss

        参数:
            alpha (float): 类别权重因子，用于平衡正负样本。默认为0.66（基于你的数据集）
            gamma (float): 调制因子，用于降低易分样本的权重。默认为2.0
            reduction (str): 'mean' 或 'sum'，指定损失的归约方式
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = 1e-6  # 添加一个小值以避免数值不稳定

    def forward(self, inputs, targets):
        """
        计算 Focal Loss

        参数:
            inputs (torch.Tensor): 模型的预测概率，已经过 sigmoid，形状为 [batch_size, 1]
            targets (torch.Tensor): 实际标签，形状为 [batch_size, 1]

        返回:
            loss (torch.Tensor): 计算得到的 focal loss
        """
        # 确保输入和目标的形状一致
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # Focal Loss 计算
        # BCE_loss = -[y*log(p) + (1-y)*log(1-p)]
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')

        # 计算 pt，即预测正确类别的概率
        pt = torch.where(targets == 1, inputs, 1 - inputs)

        # 应用 alpha 权重
        alpha_weight = torch.where(targets == 1, self.alpha, 1 - self.alpha)

        # 计算调制因子 (1-pt)^gamma
        modulating_factor = torch.pow(1.0 - pt, self.gamma)

        # 计算最终的 Focal Loss
        focal_loss = alpha_weight * modulating_factor * BCE_loss

        # 根据指定的 reduction 方法归约损失
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# 修改后的训练函数，使用F1分数作为早停指标
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=100, device='cuda',
                early_stopping_patience=5):
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

                # 计算准确率
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
        all_probs = torch.cat(all_probs, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        all_labels_int = all_labels.int()

        # 重置所有指标，确保计算准确性
        f1_metric.reset()
        auroc_metric.reset()
        conf_matrix_metric.reset()
        precision_metric.reset()
        recall_metric.reset()

        # 使用原始概率输出计算F1、AUROC、精确率和召回率
        val_f1 = f1_metric(all_probs, all_labels_int)
        val_auroc = auroc_metric(all_probs, all_labels_int)
        val_precision = precision_metric(all_probs, all_labels_int)
        val_recall = recall_metric(all_probs, all_labels_int)

        # 使用二值化预测计算混淆矩阵
        preds_binary = (all_probs > 0.5).int()
        val_conf_matrix = conf_matrix_metric(preds_binary, all_labels_int)

        # 保存F1分数
        val_f1s.append(val_f1.item())

        # 打印当前epoch的结果
        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Train Loss: {train_loss:.4f}, "
              f"Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val Acc: {val_acc:.4f}, "
              f"F1: {val_f1:.4f}, "
              f"AUROC: {val_auroc:.4f}, "
              f"Precision: {val_precision:.4f}, "
              f"Recall: {val_recall:.4f}, "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")

        print(f"Validation Confusion Matrix:\n{val_conf_matrix}")

        # 更新学习率
        scheduler.step()

        # 早停机制 - 使用F1分数
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_without_improvement = 0
            torch.save(model.state_dict(), 'best_BERT_SpliceAI_Channel64LR0001AddFC-SA_model.pth')
            print(f"验证F1分数提高到 {best_val_f1:.4f}。模型已保存。")
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
        'val_f1': val_f1s
    }

    return history


# 修改后的评估函数
def evaluate_model(model, test_loader, criterion, device='cuda'):
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

            # 计算准确率
            predicted = (outputs > 0.5).float()
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

    # 使用原始概率输出计算F1、AUROC、精确率和召回率
    f1 = f1_metric(all_probs, all_labels_int)
    auroc = auroc_metric(all_probs, all_labels_int)
    precision = precision_metric(all_probs, all_labels_int)
    recall = recall_metric(all_probs, all_labels_int)

    # 使用二值化预测计算混淆矩阵
    all_preds_binary = (all_probs > 0.5).int()
    conf_matrix = conf_matrix_metric(all_preds_binary, all_labels_int)

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
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 创建模型实例
    model = BERTSpliceModel().to(device)  # 将模型移动到指定设备（GPU/CPU）

    # 计算模型参数数量，打印参数统计信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # 损失函数和优化器
    # 使用 Focal Loss 代替 BCELoss
    criterion = FocalLoss(alpha=0.66, gamma=2.0)  # 使用指定的 alpha=0.65 和 gamma=3.0

    # 打印损失函数信息
    print(f"Using Focal Loss with alpha={criterion.alpha}, gamma={criterion.gamma}")
    
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

    # 加载训练过程中最好的模型（以最高F1分数为准）
    model.load_state_dict(torch.load('best_BERT_SpliceAI_Channel64LR0001AddFC-SA_model.pth'))

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