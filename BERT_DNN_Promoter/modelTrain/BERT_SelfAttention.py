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

# 加载 HDF5 文件
promoter_file = "/data/home/fbchou/deepLearning/BERT_DNN_Promoter/dataset/3382/promoter_dataset.h5"
non_promoter_file = "/data/home/fbchou/deepLearning/BERT_DNN_Promoter/dataset/3382/non_promoter_dataset.h5"

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


# 定义自注意力模块
class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "Embedding dimension must be divisible by number of heads"

        # 定义线性变换层
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        # 定义输出线性变换层
        self.fc_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()

        # 通过线性变换得到 Q, K, V
        Q = self.query(x)  # [batch_size, seq_len, embed_dim]
        K = self.key(x)    # [batch_size, seq_len, embed_dim]
        V = self.value(x)  # [batch_size, seq_len, embed_dim]

        # 多头注意力机制
        # 将 Q, K, V 分成 num_heads 个头
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]

        # 计算注意力分数
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))  # [batch_size, num_heads, seq_len, seq_len]
        attention_weights = torch.softmax(attention_scores, dim=-1)  # [batch_size, num_heads, seq_len, seq_len]

        # 计算加权和
        out = torch.matmul(attention_weights, V)  # [batch_size, num_heads, seq_len, head_dim]
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)  # [batch_size, seq_len, embed_dim]

        # 通过输出线性变换
        out = self.fc_out(out)  # [batch_size, seq_len, embed_dim]
        return out


# 定义SpliceAI模型
class BERTSpliceModel(nn.Module):
    def __init__(self, input_channels=768, num_classes=1):
        super(BERTSpliceModel, self).__init__()
        self.input_conv = nn.Conv1d(input_channels, 32, kernel_size=1)

        # 自注意力机制
        self.attention = SelfAttention(embed_dim=32, num_heads=4)

        # 全局平均池化
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        # 最终全连接层
        self.fc = nn.Linear(32, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 输入形状转换
        x = x.transpose(1, 2)  # [batch_size, 768, 81]

        # 初始卷积
        x = self.input_conv(x)  # [batch_size, 32, 81]

        # 自注意力机制
        x = x.transpose(1, 2)  # [batch_size, 81, 32]
        x = self.attention(x)  # [batch_size, 81, 32]
        x = x.transpose(1, 2)  # [batch_size, 32, 81]

        # 全局平均池化
        x = self.global_avg_pool(x)  # [batch_size, 32, 1]
        x = x.squeeze(2)  # [batch_size, 32]

        # 最终分类
        x = self.fc(x)  # [batch_size, 1]
        x = self.sigmoid(x)
        return x
    
    """
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)"""


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
            torch.save(model.state_dict(), 'best_BERT_SA_model.pth')
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
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

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
    model.load_state_dict(torch.load('best_BERT_SA_model.pth'))

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