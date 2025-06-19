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
import random
import time
from collections import defaultdict

# 设置基本配置
base_dir = "/data/home/fbchou/deepLearning/BERT_DNN_Promoter/dataset"

# 需要处理的数据集列表
datasets = ["Arabidopsis_non_tata", "Arabidopsis_tata", "human_non_tata", "Mouse_non_tata", "Mouse_tata"]

# 设置随机种子以确保可重复性
SEED = 523
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# 定义一个类来收集和处理文件路径
class DatasetInfo:
    def __init__(self, base_dir, datasets):
        self.base_dir = base_dir
        self.datasets = datasets
        self.promoter_files = []
        self.non_promoter_files = []
        self.promoter_counts = []
        self.non_promoter_counts = []
        self.collect_file_info()

    def collect_file_info(self):
        """收集所有文件路径和样本计数"""
        for dataset in tqdm(self.datasets, desc="Collecting dataset info"):
            promoter_file = os.path.join(self.base_dir, dataset, "promoter_dataset.h5")
            non_promoter_file = os.path.join(self.base_dir, dataset, "non_promoter_dataset.h5")

            # 检查文件是否存在
            if not os.path.exists(promoter_file) or not os.path.exists(non_promoter_file):
                print(f"Warning: Files for {dataset} not found, skipping.")
                continue

            # 打开文件获取形状信息，但不加载数据
            try:
                with h5py.File(promoter_file, 'r') as hf:
                    promoter_count = hf['X_data'].shape[0]
                    # 获取单个样本的形状，用于后续内存估算
                    sample_shape = hf['X_data'].shape[1:]
                with h5py.File(non_promoter_file, 'r') as hf:
                    non_promoter_count = hf['X_data'].shape[0]

                self.promoter_files.append(promoter_file)
                self.non_promoter_files.append(non_promoter_file)
                self.promoter_counts.append(promoter_count)
                self.non_promoter_counts.append(non_promoter_count)

                print(f"Dataset: {dataset}")
                print(f"  X_promoter shape: ({promoter_count}, {', '.join(map(str, sample_shape))})")
                print(f"  X_non_promoter shape: ({non_promoter_count}, {', '.join(map(str, sample_shape))})")

            except Exception as e:
                print(f"Error accessing {dataset}: {str(e)}")

        # 计算单个样本大小（以字节为单位）
        if self.promoter_files:
            with h5py.File(self.promoter_files[0], 'r') as hf:
                # 获取单个样本
                sample = hf['X_data'][0]
                # 计算字节大小 (float32通常为4字节/元素)
                self.sample_bytes = sample.nbytes
                # 记录样本形状
                self.sample_shape = sample.shape
        else:
            print("Warning: No valid files found to determine sample size")
            self.sample_bytes = 0
            self.sample_shape = None

        # 总样本数量
        self.total_promoter = sum(self.promoter_counts)
        self.total_non_promoter = sum(self.non_promoter_counts)
        self.total_samples = self.total_promoter + self.total_non_promoter

        print(f"Total promoter samples: {self.total_promoter}")
        print(f"Total non-promoter samples: {self.total_non_promoter}")
        print(f"Total samples: {self.total_samples}")
        print(f"Sample size: {self.sample_bytes / 1024:.2f} KB per sample")

        if self.sample_shape:
            print(f"Sample shape: {self.sample_shape}")


# 分块加载数据集类
class ChunkedH5Dataset(Dataset):
    def __init__(self, file_mapping, indices_map, chunk_size=1000, max_chunks_in_memory=10):
        """
        初始化分块加载数据集

        Args:
            file_mapping: 字典，文件ID到文件信息的映射
            indices_map: 列表，包含(file_id, local_index)元组
            chunk_size: 每个块包含的样本数量
            max_chunks_in_memory: 内存中保留的最大块数
        """
        self.file_mapping = file_mapping
        self.indices_map = indices_map
        self.chunk_size = chunk_size
        self.max_chunks_in_memory = max_chunks_in_memory

        # 块缓存: {(file_id, chunk_id): {data: tensor, last_used: timestamp}}
        self.chunk_cache = {}

        # 打开的文件句柄
        self.open_files = {}

        # 缓存统计
        self.cache_hits = 0
        self.cache_misses = 0

        # 计算并打印估计的缓存内存使用量
        self._estimate_memory_usage()

    def _estimate_memory_usage(self):
        """估计缓存会使用多少内存"""
        # 尝试找出样本大小
        sample_size = 0
        for file_id in self.file_mapping:
            try:
                if not os.path.exists(self.file_mapping[file_id]['path']):
                    continue
                with h5py.File(self.file_mapping[file_id]['path'], 'r') as f:
                    # 获取单个样本的字节数
                    sample = f['X_data'][0]
                    sample_size = sample.nbytes
                    break
            except:
                continue

        if sample_size > 0:
            chunk_mem = (sample_size * self.chunk_size) / (1024 * 1024)  # MB
            total_mem = chunk_mem * self.max_chunks_in_memory
            print(f"Estimated memory usage per chunk: {chunk_mem:.2f} MB")
            print(f"Estimated total cache memory: {total_mem:.2f} MB")

    def __len__(self):
        return len(self.indices_map)

    def __getitem__(self, idx):
        # 获取文件ID和本地索引
        file_id, local_idx = self.indices_map[idx]
        file_info = self.file_mapping[file_id]

        # 计算块ID
        chunk_id = local_idx // self.chunk_size
        chunk_key = (file_id, chunk_id)

        # 从缓存获取或加载数据块
        if chunk_key in self.chunk_cache:
            # 缓存命中
            self.cache_hits += 1
            chunk_data = self.chunk_cache[chunk_key]['data']
            # 更新最后使用时间
            self.chunk_cache[chunk_key]['last_used'] = time.time()
        else:
            # 缓存未命中，需要从文件加载
            self.cache_misses += 1

            # 确保文件已打开
            if file_id not in self.open_files:
                try:
                    self.open_files[file_id] = h5py.File(file_info['path'], 'r')
                except Exception as e:
                    print(f"Error opening file {file_info['path']}: {e}")
                    raise

            h5_file = self.open_files[file_id]

            # 计算块的起始和结束索引
            start_idx = chunk_id * self.chunk_size
            end_idx = min(start_idx + self.chunk_size, h5_file['X_data'].shape[0])

            try:
                # 加载数据块
                chunk_data = torch.tensor(h5_file['X_data'][start_idx:end_idx], dtype=torch.float32)

                # 缓存数据块
                self.chunk_cache[chunk_key] = {
                    'data': chunk_data,
                    'last_used': time.time()
                }

                # 如果缓存过大，移除最久未使用的块
                if len(self.chunk_cache) > self.max_chunks_in_memory:
                    self._evict_oldest_chunk()

            except Exception as e:
                print(f"Error loading chunk {chunk_key} from file {file_info['path']}: {e}")
                print(f"Chunk range: {start_idx} to {end_idx}")
                raise

        # 计算样本在块内的索引
        within_chunk_idx = local_idx % self.chunk_size

        # 确保索引有效
        if within_chunk_idx >= chunk_data.size(0):
            print(f"Warning: Index {within_chunk_idx} out of bounds for chunk of size {chunk_data.size(0)}")
            within_chunk_idx = chunk_data.size(0) - 1

        # 获取样本数据
        x_sample = chunk_data[within_chunk_idx]
        y_sample = torch.tensor(file_info['label'], dtype=torch.float32)

        return x_sample, y_sample

    def _evict_oldest_chunk(self):
        """移除最久未使用的数据块"""
        if not self.chunk_cache:
            return

        oldest_key = min(self.chunk_cache.keys(),
                         key=lambda k: self.chunk_cache[k]['last_used'])
        del self.chunk_cache[oldest_key]

    def get_cache_stats(self):
        """返回缓存命中率统计信息"""
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0
        return {
            'hits': self.cache_hits,
            'misses': self.cache_misses,
            'hit_rate': hit_rate,
            'chunks_in_memory': len(self.chunk_cache)
        }

    def __del__(self):
        """确保所有文件被关闭"""
        for f in self.open_files.values():
            try:
                f.close()
            except:
                pass


# 数据集分割管理类
class DataSplitter:
    def __init__(self, dataset_info, test_size=0.25, val_size=0.5, random_state=523):
        self.dataset_info = dataset_info
        self.test_size = test_size
        self.val_size = val_size  # val_size是applied on test portion
        self.random_state = random_state
        self.file_mapping = {}
        self.setup_file_mapping()

    def setup_file_mapping(self):
        """设置文件映射字典：文件ID -> 文件信息"""
        file_id = 0

        # 处理促进子文件
        for i, filepath in enumerate(self.dataset_info.promoter_files):
            self.file_mapping[file_id] = {
                'path': filepath,
                'label': 1.0,  # 促进子标签为1
                'count': self.dataset_info.promoter_counts[i]
            }
            file_id += 1

        # 处理非促进子文件
        for i, filepath in enumerate(self.dataset_info.non_promoter_files):
            self.file_mapping[file_id] = {
                'path': filepath,
                'label': 0.0,  # 非促进子标签为0
                'count': self.dataset_info.non_promoter_counts[i]
            }
            file_id += 1

    def create_indices_map(self):
        """创建完整的索引映射列表"""
        indices_map = []

        # 为每个文件创建本地索引
        for file_id, file_info in self.file_mapping.items():
            for local_idx in range(file_info['count']):
                indices_map.append((file_id, local_idx))

        return indices_map

    def split_data(self, chunk_size=1000, max_chunks_in_memory=10):
        """
        分割数据为训练、验证和测试集

        Args:
            chunk_size: 每个块包含的样本数量
            max_chunks_in_memory: 内存中保留的最大块数
        """
        indices_map = self.create_indices_map()

        # 随机打乱索引
        random.seed(self.random_state)
        random.shuffle(indices_map)

        # 计算分割点
        test_split = int(len(indices_map) * (1 - self.test_size))
        val_split = test_split + int(len(indices_map[test_split:]) * (1 - self.val_size))

        # 分割索引
        train_indices = indices_map[:test_split]
        val_indices = indices_map[test_split:val_split]
        test_indices = indices_map[val_split:]

        # 创建分块加载数据集
        print(f"Creating datasets with chunk_size={chunk_size}, max_chunks_in_memory={max_chunks_in_memory}")

        train_dataset = ChunkedH5Dataset(
            self.file_mapping,
            train_indices,
            chunk_size=chunk_size,
            max_chunks_in_memory=max_chunks_in_memory
        )

        val_dataset = ChunkedH5Dataset(
            self.file_mapping,
            val_indices,
            chunk_size=chunk_size,
            max_chunks_in_memory=max_chunks_in_memory
        )

        test_dataset = ChunkedH5Dataset(
            self.file_mapping,
            test_indices,
            chunk_size=chunk_size,
            max_chunks_in_memory=max_chunks_in_memory
        )

        # 打印数据集大小
        print(f"Training set: {len(train_dataset)} samples")
        print(f"Validation set: {len(val_dataset)} samples")
        print(f"Test set: {len(test_dataset)} samples")

        return train_dataset, val_dataset, test_dataset


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

        # 添加sigmoid激活函数
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

        # 应用sigmoid激活函数
        x = self.sigmoid(x)

        return x


# 批次大小限制
def worker_init_fn(worker_id):
    torch.set_num_threads(1)  # 限制每个工作线程使用的核心数


# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=100, device='cuda',
                early_stopping_patience=5, print_cache_stats_every=5):
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

    # 获取数据集对象来打印缓存统计
    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset

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
            outputs = outputs.squeeze(1)  # 从 [batch_size, 1] 变为 [batch_size]
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
                outputs = outputs.squeeze(1)  # 从 [batch_size, 1] 变为 [batch_size]
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

        # 打印缓存统计
        if epoch % print_cache_stats_every == 0:
            train_stats = train_dataset.get_cache_stats()
            val_stats = val_dataset.get_cache_stats()
            print(f"Train cache stats: hit rate={train_stats['hit_rate']:.2f}, "
                  f"chunks in memory={train_stats['chunks_in_memory']}")
            print(f"Val cache stats: hit rate={val_stats['hit_rate']:.2f}, "
                  f"chunks in memory={val_stats['chunks_in_memory']}")

        # 更新学习率
        scheduler.step()

        # 早停机制
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), 'best_BERT_SpliceAI_251bp_model.pth')
            print("Validation loss improved. Model saved.")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= early_stopping_patience:
            print("Early stopping activated. Training stopped.")
            break

        # 强制清理一些内存
        torch.cuda.empty_cache()

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
            outputs = outputs.squeeze(1)  # 从 [batch_size, 1] 变为 [batch_size]
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
    import argparse

    # 命令行参数
    parser = argparse.ArgumentParser(description="Promoter Prediction using BERT embeddings")
    parser.add_argument('--chunk_size', type=int, default=5000, help='Number of samples per chunk')
    parser.add_argument('--max_chunks', type=int, default=8, help='Maximum number of chunks to keep in memory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Maximum number of epochs')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--device', type=str, default='cuda:2', help='Device to use (cuda, cuda:0, cuda:1, etc)')
    parser.add_argument('--workers', type=int, default=4, help='Number of dataloader workers')
    parser.add_argument('--prefetch', type=int, default=2, help='Prefetch factor for dataloader')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for optimizer')

    args = parser.parse_args()

    # 设置设备
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.device.startswith('cuda:') and torch.cuda.is_available():
        device = torch.device(args.device)
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # 收集数据集信息
    dataset_info = DatasetInfo(base_dir, datasets)

    # 确保有有效的数据集
    if len(dataset_info.promoter_files) == 0 or len(dataset_info.non_promoter_files) == 0:
        raise ValueError("No valid datasets were found.")

    # 分割数据集，使用命令行参数指定的块大小和内存中保留的块数
    data_splitter = DataSplitter(dataset_info)
    train_dataset, val_dataset, test_dataset = data_splitter.split_data(
        chunk_size=args.chunk_size,
        max_chunks_in_memory=args.max_chunks
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        worker_init_fn=worker_init_fn,
        prefetch_factor=args.prefetch,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        worker_init_fn=worker_init_fn,
        prefetch_factor=args.prefetch,
        persistent_workers=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        worker_init_fn=worker_init_fn,
        prefetch_factor=args.prefetch,
        persistent_workers=True
    )

    # 检查批次形状
    print("Checking batch shapes...")
    for inputs, labels in train_loader:
        print(f"Batch shape: {inputs.shape}, Label shape: {labels.shape}")
        break

    # 创建模型实例
    model = BERTSpliceModel().to(device)

    # 计算模型参数数量，打印参数统计信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # 损失函数和优化器
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

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
        num_epochs=args.epochs,
        device=device,
        early_stopping_patience=args.patience,
        print_cache_stats_every=5  # 每5个epoch打印一次缓存统计
    )

    # 加载训练过程中最好的模型
    model.load_state_dict(torch.load('best_BERT_SpliceAI_251bp_model.pth'))

    # 打印测试集上的缓存统计
    test_stats = test_dataset.get_cache_stats()
    print(f"Test cache stats: hit rate={test_stats['hit_rate']:.2f}, "
          f"chunks in memory={test_stats['chunks_in_memory']}")

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
    model_scripted = torch.jit.script(model)
    model_scripted.save('bert_splice_model.pt')
    print("Saved scripted model for deployment.")