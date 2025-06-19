import numpy as np
import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.init as init
from tqdm import tqdm
import torchmetrics
import os
import json
from sklearn.metrics import auc  # 用于计算AUPRC

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

        self.input_conv = nn.Conv1d(input_channels, 32, kernel_size=1)
        self.rb_group1 = nn.ModuleList([ResidualBlock(channels=32, kernel_size=5, dilation=1) for _ in range(4)])
        self.module1_skip = nn.Conv1d(32, 32, kernel_size=1)
        self.rb_group2 = nn.ModuleList([ResidualBlock(channels=32, kernel_size=5, dilation=2) for _ in range(4)])
        self.module2_skip = nn.Conv1d(32, 32, kernel_size=1)
        self.rb_group3 = nn.ModuleList([ResidualBlock(channels=32, kernel_size=9, dilation=3) for _ in range(4)])
        self.module3_skip = nn.Conv1d(32, 32, kernel_size=1)
        self.afterRB_conv = nn.Conv1d(32, 32, kernel_size=1)
        self.final_conv = nn.Conv1d(32, num_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.input_conv(x)

        skip1 = self.module1_skip(x)
        for rb in self.rb_group1:
            x = rb(x)

        skip2 = self.module2_skip(x)
        for rb in self.rb_group2:
            x = rb(x)

        skip3 = self.module3_skip(x)
        for rb in self.rb_group3:
            x = rb(x)

        x = self.afterRB_conv(x)
        result = x + skip1 + skip2 + skip3

        x = self.final_conv(x)
        x = x.transpose(1, 2)
        x = torch.mean(x, dim=1, keepdim=True)
        x = x.squeeze(2)
        x = self.sigmoid(x)

        return x


# 评估函数
def evaluate_model(model, test_loader, criterion, species_name, device='cuda'):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

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
        for inputs, labels in tqdm(test_loader, desc=f"Evaluating {species_name}"):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.append(predicted)
            all_probs.append(outputs)
            all_labels.append(labels)

    test_loss = test_loss / len(test_loader)
    test_acc = correct / total

    all_preds = torch.cat(all_preds, dim=0)
    all_probs = torch.cat(all_probs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    all_labels_int = all_labels.int()
    all_preds_binary = (all_preds > 0.5).int()

    # 计算其他指标
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
        'confusion_matrix': conf_matrix.cpu().numpy()
    }


# 主函数
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = BERTSpliceModel().to(device)
    model_path = 'best_BERT_SpliceAI_model.pth'
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Loaded model from {model_path}")

    criterion = torch.nn.BCELoss()

    all_results = {}

    for species_name in test_datasets:
        try:
            promoter_file = os.path.join(base_dir, species_name, "promoter_dataset.h5")
            non_promoter_file = os.path.join(base_dir, species_name, "non_promoter_dataset.h5")

            if not os.path.exists(promoter_file) or not os.path.exists(non_promoter_file):
                print(f"警告: {species_name} 的文件未找到，跳过此物种。")
                continue

            print(f"\n开始加载 {species_name} 的测试数据...")
            with h5py.File(promoter_file, 'r') as hf:
                X_promoter = hf['X_data'][:]
                y_promoter = hf['y_labels'][:]

            with h5py.File(non_promoter_file, 'r') as hf:
                X_non_promoter = hf['X_data'][:]
                y_non_promoter = hf['y_labels'][:]

            X_test = np.concatenate([X_promoter, X_non_promoter], axis=0)
            y_test = np.concatenate([y_promoter, y_non_promoter], axis=0)

            indices = np.arange(X_test.shape[0])
            np.random.seed(42)
            np.random.shuffle(indices)
            X_test = X_test[indices]
            y_test = y_test[indices]

            test_dataset = BERTDataset(X_test, y_test)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

            print(f"开始评估 {species_name} 物种的性能...")
            species_metrics = evaluate_model(model, test_loader, criterion, species_name, device)

            all_results[species_name] = species_metrics

        except Exception as e:
            print(f"处理 {species_name} 时出错: {str(e)}")
            continue

    all_results_file = "all_species_test_metrics.json"
    with open(all_results_file, 'w') as f:
        all_metrics_json = {species: {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in metrics.items()} for species, metrics in all_results.items()}
        json.dump(all_metrics_json, f, indent=4)

    print(f"\n所有物种的测试已完成! 综合结果已保存到 {all_results_file}")

    # 打印性能比较表格
    print("\n各物种性能指标比较:")
    print("=" * 80)
    print(f"{'物种名称':<12} {'准确率':<8} {'F1分数':<8} {'AUROC':<8} {'AUPRC':<8} {'精确率':<8} {'召回率':<8}")
    print("-" * 80)

    for species, metrics in all_results.items():
        print(
            f"{species:<15} {metrics['accuracy']:<10.4f} {metrics['f1']:<10.4f} {metrics['auroc']:<10.4f} {metrics['auprc']:<10.4f} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f}")

    print("=" * 80)


if __name__ == "__main__":
    main()
