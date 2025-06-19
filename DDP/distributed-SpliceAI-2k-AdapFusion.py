import torch
import torch.nn as nn
import os
#from dataloaderSmall_NW4 import train_loader, val_loader, test_loader
from dataloaderBig_seed import train_loader, val_loader, test_loader
from tqdm import tqdm
import torchmetrics
import torch.nn.init as init
import torch.distributed as dist
import h5py
import numpy as np

# 设置device为GPU
device = torch.device("cuda:0")

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


class AdaptiveFeatureFusion(nn.Module):
    def __init__(self, channels):
        super(AdaptiveFeatureFusion, self).__init__()

        # 每个特征流的注意力模块
        self.attention_main = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels // 4, channels, kernel_size=1),
            nn.Sigmoid()
        )

        self.attention_skip1 = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels // 4, channels, kernel_size=1),
            nn.Sigmoid()
        )

        self.attention_skip2 = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels // 4, channels, kernel_size=1),
            nn.Sigmoid()
        )

        self.attention_skip3 = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels // 4, channels, kernel_size=1),
            nn.Sigmoid()
        )

        # 特征融合后的调整层
        self.adjustment = nn.Sequential(
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=1)
        )

    def forward(self, main_path, skip1, skip2, skip3):
        # 确保所有特征尺寸一致
        seq_len = min(main_path.size(2), skip1.size(2), skip2.size(2), skip3.size(2))
        main_path = main_path[:, :, :seq_len]
        skip1 = skip1[:, :, :seq_len]
        skip2 = skip2[:, :, :seq_len]
        skip3 = skip3[:, :, :seq_len]

        # 计算每个特征流的注意力权重
        w_main = self.attention_main(main_path)
        w_skip1 = self.attention_skip1(skip1)
        w_skip2 = self.attention_skip2(skip2)
        w_skip3 = self.attention_skip3(skip3)

        # 归一化权重，确保总和为1
        weight_sum = w_main + w_skip1 + w_skip2 + w_skip3
        w_main = w_main / weight_sum
        w_skip1 = w_skip1 / weight_sum
        w_skip2 = w_skip2 / weight_sum
        w_skip3 = w_skip3 / weight_sum

        # 加权融合
        fused = (main_path * w_main +
                 skip1 * w_skip1 +
                 skip2 * w_skip2 +
                 skip3 * w_skip3)

        # 最终调整
        return self.adjustment(fused)


class SpliceAI2kAdaptiveFusionModel(nn.Module):
    def __init__(self, input_channels=4, num_classes=1):
        super(SpliceAI2kAdaptiveFusionModel, self).__init__()

        # 初始卷积层，从4通道转换到32通道
        self.input_conv = nn.Conv1d(input_channels, 32, kernel_size=1)

        # 第一组残差块：4个RB(32, 11, 1)
        self.rb_group1 = nn.ModuleList()
        for i in range(4):
            self.rb_group1.append(ResidualBlock(channels=32, kernel_size=11, dilation=1))

        # 跳跃连接卷积层1
        self.module1_skip = nn.Conv1d(32, 32, kernel_size=1)

        # 第二组残差块：4个RB(32, 11, 4)
        self.rb_group2 = nn.ModuleList()
        for i in range(4):
            self.rb_group2.append(ResidualBlock(channels=32, kernel_size=11, dilation=4))

        # 跳跃连接卷积层2
        self.module2_skip = nn.Conv1d(32, 32, kernel_size=1)

        # 第三组残差块：4个RB(32, 21, 10)
        self.rb_group3 = nn.ModuleList()
        for i in range(4):
            self.rb_group3.append(ResidualBlock(channels=32, kernel_size=21, dilation=10))

        # 跳跃连接卷积层3
        self.module3_skip = nn.Conv1d(32, 32, kernel_size=1)

        # RB后的一个卷积层
        self.afterRB_conv = nn.Conv1d(32, 32, kernel_size=1)

        # 添加自适应特征融合模块
        self.adaptive_fusion = AdaptiveFeatureFusion(channels=32)

        # 最后卷积层
        self.last_conv = nn.Conv1d(32, num_classes, kernel_size=1)

    def forward(self, x):
        # 输入形状转换：[batch_size, sequence_length, input_channels]
        # -> [batch_size, input_channels, sequence_length]
        x = x.permute(0, 2, 1)

        # 初始卷积
        x = self.input_conv(x)

        # x 右拐经过第一个conv后的值 记为 skip1
        skip1 = self.module1_skip(x)

        # x 向下直行经过第一组 rb 层
        for rb in self.rb_group1:
            x = rb(x)

        # x 右拐经过第二个conv后的值 记为 skip2
        skip2 = self.module2_skip(x)

        # x 向下直行经过第二组 rb 层
        for rb in self.rb_group2:
            x = rb(x)

        # x 右拐经过第三个conv后的值 记为 skip3
        skip3 = self.module3_skip(x)

        # x 向下直行经过第三组 rb 层
        for rb in self.rb_group3:
            x = rb(x)

        # x 向下直行经过 "RB后的一个卷积层"
        x = self.afterRB_conv(x)

        # 使用自适应融合替代简单相加
        result = self.adaptive_fusion(x, skip1, skip2, skip3)

        # result 向下直行经过：最后的卷积层
        x = self.last_conv(result)

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


# 初始化分布式训练环境
def init_dist():
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12375'
    rank = 0
    world_size = 1
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


if __name__ == "__main__":
    # 创建自适应融合模型
    model = SpliceAI2kAdaptiveFusionModel(input_channels=4, num_classes=2)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)  # 转换 BatchNorm 层为 SyncBatchNorm
    model = nn.DataParallel(model)  # 使用 DataParallel 包装模型
    model = model.to(device)

    learning_rate = 0.001  # 学习率在使用余弦退火时可以设置更高一点
    output_size = 2  # 二分类问题
    num_epochs = 100  # 设置得高一些，反正已经有早停机制保障了，以防万一训练到 25 轮还没完全收敛，就会导致前功尽弃

    # 定义损失函数
    criterion = nn.CrossEntropyLoss(reduction='none')

    # 定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 设置余弦退火动态学习率
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=0)

    # 初始化 F1 分数和混淆矩阵计算器
    f1_metric = torchmetrics.classification.F1Score(num_classes=output_size, average=None, task="binary").to(device)
    conf_matrix_metric = torchmetrics.ConfusionMatrix(num_classes=output_size, task="binary").to(device)
    accuracy_metric = torchmetrics.Accuracy(num_classes=output_size, task="binary").to(device)  # 新增用于计算准确率的指标

    # 添加训练和验证阶段的准确率指标
    train_accuracy_metric = torchmetrics.Accuracy(num_classes=output_size, task="binary").to(device)
    val_accuracy_metric = torchmetrics.Accuracy(num_classes=output_size, task="binary").to(device)

    # 早停设置
    early_stop = False
    patience = 5
    counter = 0
    best_f1_score = 0.0

    # 开始训练循环
    init_dist()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        # 重置训练阶段的准确率指标
        train_accuracy_metric.reset()

        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            labels = labels.view(-1)
            outputs = outputs.reshape(-1, output_size)

            loss = criterion(outputs, labels).mean()

            loss.backward()
            optimizer.step()

            running_loss += loss.item()  # 累积每个batch的损失，得到整个epoch的总损失

            # 计算训练准确率
            _, train_preds = torch.max(outputs, 1)
            train_accuracy_metric.update(train_preds, labels)

        avg_train_loss = running_loss / len(train_loader)
        # 计算训练准确率
        train_accuracy = train_accuracy_metric.compute()

        # 验证
        model.eval()
        val_loss = 0.0
        f1_metric.reset()
        conf_matrix_metric.reset()
        # 重置验证阶段的准确率指标
        val_accuracy_metric.reset()

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                labels = labels.view(-1)
                outputs = outputs.reshape(-1, output_size)

                loss = criterion(outputs, labels).mean()
                val_loss += loss.item()

                _, val_preds = torch.max(outputs, 1)

                f1_metric.update(val_preds, labels)
                conf_matrix_metric.update(val_preds, labels)
                # 计算验证准确率
                val_accuracy_metric.update(val_preds, labels)

        avg_val_loss = val_loss / len(val_loader)

        scheduler.step()

        f1_score = f1_metric.compute()
        val_conf_matrix = conf_matrix_metric.compute()
        # 计算验证准确率
        val_accuracy = val_accuracy_metric.compute()

        current_lr = optimizer.param_groups[0]['lr']
        print(
            f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, F1 Score: {f1_score}, Current Learning Rate: {current_lr:.8f}')
        print(f'Validation Confusion Matrix for Epoch {epoch + 1}:')
        print(val_conf_matrix)

        # 早停机制
        avg_f1_score = f1_metric.compute().mean().item()
        if best_f1_score is None or avg_f1_score > best_f1_score:
            best_f1_score = avg_f1_score
            counter = 0
            torch.save(model.state_dict(), 'best_AdaptiveFusion_model.pth')
            print(f'save best model in Epoch {epoch + 1}')
        else:
            counter += 1
            if counter >= patience:
                early_stop = True
                print("Early stopping")
                break

    # 测试
    model.load_state_dict(torch.load('best_AdaptiveFusion_model.pth'))
    model.eval()
    test_loss = 0
    f1_metric.reset()
    conf_matrix_metric.reset()
    accuracy_metric.reset()  # 重置准确率指标

    with torch.no_grad():
        for inputs, labels in test_loader:  # 增加引入掩码矩阵,新增id和strand信息
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            labels = labels.view(-1)
            outputs = outputs.reshape(-1, output_size)

            # 计算损失
            loss = criterion(outputs, labels).mean()
            test_loss += loss.item()

            # 获取每个碱基的预测标签，torch.max(input, dim)返回每行的：最大值（忽略） 和 对应的索引
            _, test_preds = torch.max(outputs, 1)

            # 更新 F1 分数和混淆矩阵计算器
            f1_metric.update(test_preds, labels)
            conf_matrix_metric.update(test_preds, labels)
            accuracy_metric.update(test_preds, labels)  # 更新准确率指标

    # 测试阶段的平均损失
    avg_test_loss = test_loss / len(test_loader)

    # 计算准确率
    test_accuracy = accuracy_metric.compute()

    # task设置为2分类，预测类别为 output_size = 2
    test_f1_score = f1_metric.compute()

    # 混淆矩阵
    test_conf_matrix = conf_matrix_metric.compute()

    # 输出测试结果
    print(f'Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Test F1 Score: {test_f1_score}')
    print("Test Confusion Matrix:")
    print(test_conf_matrix)