import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchmetrics
import os
import math
import json
from tqdm import tqdm
from sklearn.metrics import auc  # 用于计算AUPRC
import sys

# 需要测试的数据集列表
test_datasets = ["BACILLUS", "CLOSTRIDIUM", "MYCOBACTER", "RHODOBACTER_1", "RHODOBACTER_2"]

# 设置基本配置
base_dir = "/data/home/fbchou/deepLearning/BERT_DNN_Promoter/dataset"


# 自定义Dataset类
class BERTDataset(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = torch.tensor(X_data, dtype=torch.float32)
        self.y_data = torch.tensor(y_data, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, idx):
        return self.X_data[idx], self.y_data[idx]


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


# 评估函数
def evaluate_model(model, test_loader, criterion, species_name, device='cuda', threshold=0.5):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    # 初始化指标
    f1_metric = torchmetrics.F1Score(task="binary").to(device)
    auroc_metric = torchmetrics.AUROC(task="binary").to(device)
    precision_metric = torchmetrics.Precision(task="binary").to(device)
    recall_metric = torchmetrics.Recall(task="binary").to(device)
    conf_matrix_metric = torchmetrics.ConfusionMatrix(task="binary", num_classes=2).to(device)

    # 使用 PrecisionRecallCurve 来计算 Precision-Recall 曲线
    precision_recall_curve_metric = torchmetrics.PrecisionRecallCurve(task="binary").to(device)

    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc=f"评估 {species_name}"):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            # 使用指定阈值计算准确率
            predicted = (outputs > threshold).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.append(predicted)
            all_probs.append(outputs)
            all_labels.append(labels)

    test_loss = test_loss / len(test_loader)
    test_acc = correct / total

    # 合并所有批次的结果
    all_preds = torch.cat(all_preds, dim=0)
    all_probs = torch.cat(all_probs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    all_labels_int = all_labels.int()
    all_preds_binary = (all_probs > threshold).int()

    # 计算指标
    f1 = f1_metric(all_preds_binary, all_labels_int)
    auroc = auroc_metric(all_probs, all_labels_int)
    precision = precision_metric(all_preds_binary, all_labels_int)
    recall = recall_metric(all_preds_binary, all_labels_int)
    conf_matrix = conf_matrix_metric(all_preds_binary, all_labels_int)

    # 计算 Precision-Recall 曲线数据
    precision_values, recall_values, _ = precision_recall_curve_metric(all_probs, all_labels_int)

    # 计算 AUPRC（AUC of the Precision-Recall Curve）
    auprc = auc(recall_values.cpu().numpy(), precision_values.cpu().numpy())

    print(f"\n{species_name} 测试结果:")
    print(f"Loss: {test_loss:.4f}")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUROC: {auroc:.4f}")
    print(f"AUPRC: {auprc:.4f}")  # 输出AUPRC
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"阈值: {threshold:.4f}")
    print(f"Confusion Matrix:\n{conf_matrix}")

    return {
        'dataset': species_name,
        'loss': test_loss,
        'accuracy': test_acc,
        'f1': f1.item(),
        'auroc': auroc.item(),
        'auprc': auprc,  # 保存AUPRC
        'precision': precision.item(),
        'recall': recall.item(),
        'threshold': threshold,
        'confusion_matrix': conf_matrix.cpu().numpy()
    }


# 找最佳阈值的函数
def find_optimal_threshold(model, val_loader, device):
    model.eval()
    all_probs = []
    all_labels = []

    print("收集预测概率和真实标签...")
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="寻找最佳阈值"):
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
    print("测试不同阈值以找到最优值...")
    thresholds = np.arange(0.3, 0.7, 0.01)
    for threshold in tqdm(thresholds, desc="测试阈值"):
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


# 主函数
def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 创建模型实例
    model = EnhancedDNABERTPromoterModel().to(device)

    # 加载训练好的模型
    model_path = 'best_Enhanced_BERT_Promoter_model.pth'

    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        # 获取最佳阈值
        best_threshold = checkpoint.get('best_threshold', 0.5)
        print(f"已加载模型，来自epoch {checkpoint.get('epoch', 'unknown')}")
        print(f"使用最佳阈值: {best_threshold:.4f}")
    except Exception as e:
        print(f"加载模型时出错: {str(e)}")
        best_threshold = 0.5
        print(f"使用默认阈值: {best_threshold}")

    # 使用与训练相同的损失函数
    criterion = FocalDiceLoss(alpha=0.5, gamma=1.0, beta=0.2)
    print(f"使用FocalDiceLoss，alpha={criterion.alpha}, gamma={criterion.gamma}, beta={criterion.beta}")

    # 存储所有测试结果
    all_results = {}

    for species_name in test_datasets:
        try:
            # 加载该物种的数据
            promoter_file = os.path.join(base_dir, species_name, "promoter_dataset.h5")
            non_promoter_file = os.path.join(base_dir, species_name, "non_promoter_dataset.h5")

            # 检查文件是否存在
            if not os.path.exists(promoter_file) or not os.path.exists(non_promoter_file):
                print(f"警告: {species_name} 的文件未找到，跳过此物种。")
                continue

            print(f"\n开始加载 {species_name} 的测试数据...")

            # 加载促进子数据
            with h5py.File(promoter_file, 'r') as hf:
                X_promoter = hf['X_data'][:]
                y_promoter = hf['y_labels'][:]
                print(f"  促进子数据形状: {X_promoter.shape}")
                print(f"  标签唯一值: {np.unique(y_promoter)}")

            # 加载非促进子数据
            with h5py.File(non_promoter_file, 'r') as hf:
                X_non_promoter = hf['X_data'][:]
                y_non_promoter = hf['y_labels'][:]
                print(f"  非促进子数据形状: {X_non_promoter.shape}")
                print(f"  标签唯一值: {np.unique(y_non_promoter)}")

            # 合并数据集
            X_test = np.concatenate([X_promoter, X_non_promoter], axis=0)
            y_test = np.concatenate([y_promoter, y_non_promoter], axis=0)

            # 随机打乱数据
            indices = np.arange(X_test.shape[0])
            np.random.seed(42)  # 设置随机种子以确保可重复性
            np.random.shuffle(indices)
            X_test = X_test[indices]
            y_test = y_test[indices]

            # 创建数据集和数据加载器
            test_dataset = BERTDataset(X_test, y_test)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

            print(f"开始评估 {species_name} 物种的性能...")
            # 使用最佳阈值评估模型
            species_metrics = evaluate_model(model, test_loader, criterion, species_name, device,
                                             threshold=best_threshold)

            # 存储结果
            all_results[species_name] = species_metrics

        except Exception as e:
            print(f"处理 {species_name} 时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    # 保存所有物种的测试结果
    all_results_file = "enhanced_dnabert_all_species_metrics.json"
    with open(all_results_file, 'w') as f:
        # 将numpy数组转换为列表
        all_metrics_json = {species: {k: v.tolist() if isinstance(v, np.ndarray) else v
                                      for k, v in metrics.items()}
                            for species, metrics in all_results.items()}
        json.dump(all_metrics_json, f, indent=4)

    print(f"\n所有物种的测试已完成! 综合结果已保存到 {all_results_file}")

    # 打印性能比较表格
    print("\n各物种性能指标比较:")
    print("=" * 90)
    print(f"{'物种名称':<15} {'准确率':<8} {'F1分数':<8} {'AUROC':<8} {'AUPRC':<8} {'精确率':<8} {'召回率':<8}")
    print("-" * 90)

    for species, metrics in all_results.items():
        print(
            f"{species:<15} {metrics['accuracy']:<8.4f} {metrics['f1']:<8.4f} {metrics['auroc']:<8.4f} {metrics['auprc']:<8.4f} {metrics['precision']:<8.4f} {metrics['recall']:<8.4f}")

    print("=" * 90)


if __name__ == "__main__":
    main()