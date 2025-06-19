import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.init as init
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
import torchmetrics
from tqdm import tqdm
from distributed_dataloader_small import prepare_distributed_dataloaders

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

class CNNRBModel(nn.Module):
    def __init__(self, input_channels=4, num_classes=2):
        super(CNNRBModel, self).__init__()

        # 初始卷积层，从4通道转换到32通道
        self.input_conv = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=1),
            nn.Conv1d(32, 32, kernel_size=1)
        )

        # 第一组残差块：6个RB(32, 11, 1)
        self.rb_group1 = nn.ModuleList()
        for i in range(6):
            self.rb_group1.append(ResidualBlock(channels=32, kernel_size=11, dilation=1))

        # 中间卷积层
        self.middle_conv = nn.Conv1d(32, 32, kernel_size=1)

        # 第二组残差块：6个RB(32, 11, 4)
        self.rb_group2 = nn.ModuleList()
        for i in range(6):
            self.rb_group2.append(ResidualBlock(channels=32, kernel_size=11, dilation=4))

        # 第二个中间卷积层
        self.middle_conv2 = nn.Conv1d(32, 32, kernel_size=1)

        # 第三组残差块：6个RB(32, 21, 10)
        self.rb_group3 = nn.ModuleList()
        for i in range(6):
            self.rb_group3.append(ResidualBlock(channels=32, kernel_size=21, dilation=10))

        # 输出层
        self.output_conv = nn.Sequential(nn.Conv1d(32, 32, kernel_size=1),
                                         nn.Conv1d(32, num_classes, kernel_size=1))

        self._initialize_weights()

    def forward(self, x):
        # 输入形状转换：[batch_size, sequence_length, input_channels]
        # -> [batch_size, input_channels, sequence_length]
        x = x.permute(0, 2, 1)

        # 初始卷积
        x = self.input_conv(x)

        # 保存用于跳跃连接的初始特征
        skip_connections = []

        # 第一组残差块
        for rb in self.rb_group1:
            x = rb(x)
        skip_connections.append(x)

        # 中间卷积 + 跳跃连接
        x = self.middle_conv(x) + skip_connections[0]

        # 第二组残差块
        for rb in self.rb_group2:
            x = rb(x)
        skip_connections.append(x)

        # 中间卷积 + 跳跃连接
        x = self.middle_conv2(x) + skip_connections[1]

        # 第三组残差块
        for rb in self.rb_group3:
            x = rb(x)

        # 输出卷积
        x = self.output_conv(x)

        # 转换回原始形状：[batch_size, sequence_length, num_classes]
        x = x.permute(0, 2, 1)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

def setup(rank, world_size):
    # 初始化分布式训练环境
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12376'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    # 清理分布式训练环境
    dist.destroy_process_group()

def train_and_validate(rank, world_size):
    setup(rank, world_size)

    # 数据准备
    data_path = "/data/home/fbchou/RandomSampledBacteria_Process2h5_Small/"
    train_dataset, val_dataset, test_dataset = prepare_distributed_dataloaders(data_path, rank, world_size)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)

    train_loader = DataLoader(train_dataset, batch_size=32 // world_size, sampler=train_sampler, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32 // world_size, sampler=val_sampler)
    test_loader = DataLoader(test_dataset, batch_size=32 // world_size, sampler=test_sampler)

    # 模型初始化
    model = CNNRBModel(input_channels=4, num_classes=2).to(rank)
    model = DistributedDataParallel(model, device_ids=[rank])

    # 训练配置
    learning_rate = 0.001 * world_size
    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=6, eta_min=0)

    num_epochs = 25
    output_size = 2

    # 早停设置
    patience = 5
    counter = 0
    best_f1_score = 0.0

    # 初始化 F1 分数和混淆矩阵计算器
    f1_metric = torchmetrics.classification.F1Score(num_classes=output_size, average=None, task="binary").to(rank)
    conf_matrix_metric = torchmetrics.ConfusionMatrix(num_classes=output_size, task="binary").to(rank)

    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)

        # 训练
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", disable=rank != 0):
            inputs, labels = inputs.to(rank), labels.to(rank)

            optimizer.zero_grad()
            outputs = model(inputs)

            labels = labels.view(-1)
            outputs = outputs.reshape(-1, output_size)

            loss = criterion(outputs, labels).mean()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        # 验证
        model.eval()
        val_loss = 0.0
        f1_metric.reset()
        conf_matrix_metric.reset()
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(rank), labels.to(rank)
                outputs = model(inputs)
                labels = labels.view(-1)
                outputs = outputs.reshape(-1, output_size)
                loss = criterion(outputs, labels).mean()
                val_loss += loss.item()
                _, val_preds = torch.max(outputs, 1)
                f1_metric.update(val_preds, labels)
                conf_matrix_metric.update(val_preds, labels)

        # 汇总验证损失
        avg_val_loss_tensor = torch.tensor(val_loss / len(val_loader)).to(rank)
        torch.distributed.all_reduce(avg_val_loss_tensor, op=torch.distributed.ReduceOp.SUM)
        avg_val_loss = avg_val_loss_tensor.item() / world_size

        # 汇总 F1 分数和混淆矩阵
        f1_score_tensor = f1_metric.compute().to(rank)
        torch.distributed.all_reduce(f1_score_tensor, op=torch.distributed.ReduceOp.SUM)
        f1_score = f1_score_tensor / world_size

        conf_matrix_tensor = conf_matrix_metric.compute().to(rank)
        torch.distributed.all_reduce(conf_matrix_tensor, op=torch.distributed.ReduceOp.SUM)
        val_conf_matrix = conf_matrix_tensor

        # 仅在 rank 0 上打印
        if rank == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(
                f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, F1 Score: {f1_score}, Current Learning Rate: {current_lr:.8f}')
            print(f'Validation Confusion Matrix for Epoch {epoch + 1}:')
            print(val_conf_matrix)

            # 早停机制
            avg_f1_score = f1_score.mean().item()
            if avg_f1_score > best_f1_score:
                best_f1_score = avg_f1_score
                counter = 0
                torch.save(model.state_dict(), 'best_CNNRB_6Small_model.pth')
                print(f'save best model in Epoch {epoch + 1}')
            else:
                counter += 1
                if counter >= patience:
                    print("Early stopping")
                    break

    # 测试（仅在 rank 0 上进行）
    if rank == 0:
        model.load_state_dict(torch.load('best_CNNRB_6Small_model.pth'))
        model.eval()
        test_loss = 0
        f1_metric.reset()
        conf_matrix_metric.reset()

        all_test_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(rank), labels.to(rank)

                outputs = model(inputs)
                labels = labels.view(-1)
                outputs = outputs.reshape(-1, output_size)

                # 计算损失
                loss = criterion(outputs, labels).mean()
                test_loss += loss.item()

                # 获取每个碱基的预测标签
                _, test_preds = torch.max(outputs, 1)

                # 累积所有批次的预测和标签
                all_test_preds.append(test_preds)
                all_labels.append(labels)

                # 更新 F1 分数和混淆矩阵计算器
                f1_metric.update(test_preds, labels)
                conf_matrix_metric.update(test_preds, labels)

        # 合并所有批次的数据
        all_test_preds = torch.cat(all_test_preds)
        all_labels = torch.cat(all_labels)

        # 测试阶段的平均损失
        avg_test_loss = test_loss / len(test_loader)

        # 测试集准确率
        test_accuracy = torchmetrics.functional.accuracy(all_test_preds, all_labels, num_classes=output_size, task="binary")

        # 计算 F1 分数和混淆矩阵
        test_f1_score = f1_metric.compute()
        test_conf_matrix = conf_matrix_metric.compute()

        # 输出测试结果
        print(f'Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Test F1 Score: {test_f1_score}')
        print("Test Confusion Matrix:")
        print(test_conf_matrix)

    cleanup()

# 启动分布式训练
if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(train_and_validate, args=(world_size,), nprocs=world_size)

