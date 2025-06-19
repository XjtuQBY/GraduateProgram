import torch
import torch.nn as nn
from dataloaderNOmask import train_loader, val_loader, test_loader
from tqdm import tqdm
import torchmetrics
import torch.nn.init as init
import h5py
import numpy as np

# 设置device为GPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:3")

class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size,
                      padding=(kernel_size - 1) // 2 * dilation, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size,
                      padding=(kernel_size - 1) // 2 * dilation, dilation=dilation)
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

        # 第一组残差块：4个RB(32, 11, 1)
        self.rb_group1 = nn.ModuleList([
            ResidualBlock(channels=32, kernel_size=11, dilation=1)
            for _ in range(4)
        ])

        # 中间卷积层
        self.middle_conv = nn.Conv1d(32, 32, kernel_size=1)

        # 第二组残差块：4个RB(32, 11, 4)
        self.rb_group2 = nn.ModuleList([
            ResidualBlock(channels=32, kernel_size=11, dilation=4)
            for _ in range(4)
        ])

        # 第二个中间卷积层
        self.middle_conv2 = nn.Conv1d(32, 32, kernel_size=1)

        # 第三组残差块：4个RB(32, 21, 10)
        self.rb_group3 = nn.ModuleList([
            ResidualBlock(channels=32, kernel_size=21, dilation=10)
            for _ in range(4)
        ])

        # 输出层
        self.output_conv = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=1),
            nn.Conv1d(32, num_classes, kernel_size=1)
        )

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

model = CNNRBModel(input_channels=4, num_classes=2).to(device)
learning_rate = 0.001  # 学习率在使用余弦退火时可以设置更高一点
output_size = 2 #二分类问题
num_epochs = 25

# 定义类别权重
#multiclass_weights = torch.tensor([0.2, 0.2, 1]).to(device)  # label0/1/2 0.2:0.2:1

# 定义损失函数
criterion = nn.CrossEntropyLoss(reduction='none')#criterion = nn.CrossEntropyLoss(weight=multiclass_weights)

# 定义优化器
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 设置余弦退火动态学习率
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=6, eta_min=0)
# Tmax设置为patience数更合适，避免还没达到鞍点就早停了

# 初始化 F1 分数和混淆矩阵计算器
#f1_metric = torchmetrics.classification.F1Score(num_classes=output_size, average=None, task='multiclass').to(device)
#conf_matrix_metric = torchmetrics.ConfusionMatrix(num_classes=output_size, task='multiclass').to(device)
f1_metric = torchmetrics.classification.F1Score(num_classes=output_size, average=None, task="binary").to(device)
conf_matrix_metric = torchmetrics.ConfusionMatrix(num_classes=output_size, task="binary").to(device)

# 早停设置
early_stop = False
patience = 5
counter = 0
best_f1_score = 0.0
'''
# 训练循环
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(train_loader):  
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        # 调试信息
        #print(f'Outputs shape: {outputs.shape}')
        #print(f'Labels shape: {labels.shape}')

        labels = labels.view(-1)
        outputs = outputs.reshape(-1, output_size)

        # 调试信息
        #print(f'Reshaped Outputs shape: {outputs.shape}')
        #print(f'Reshaped Labels shape: {labels.shape}')

        loss = criterion(outputs, labels).mean()

        loss.backward()
        optimizer.step()

        running_loss += loss.item()  # 累积每个batch的损失，得到整个epoch的总损失

    avg_train_loss = running_loss / len(train_loader)

    # 验证
    model.eval()
    val_loss = 0.0
    f1_metric.reset()
    conf_matrix_metric.reset()
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

    avg_val_loss = val_loss / len(val_loader)

    scheduler.step()

    f1_score = f1_metric.compute()
    val_conf_matrix = conf_matrix_metric.compute()

    current_lr = optimizer.param_groups[0]['lr']
    print(
        f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, F1 Score: {f1_score}, Current Learning Rate: {current_lr:.8f}')
    print(f'Validation Confusion Matrix for Epoch {epoch + 1}:')
    print(val_conf_matrix)

    # 早停机制
    avg_f1_score = f1_metric.compute().mean().item()
    if best_f1_score is None or avg_f1_score > best_f1_score:
        best_f1_score = avg_f1_score
        counter = 0
        torch.save(model.state_dict(), 'best_CNNRB_model1.0.pth')
        print(f'save best model in Epoch {epoch + 1}')
    else:
        counter += 1
        if counter >= patience:
            early_stop = True
            print("Early stopping")
            break


def save_predictions_with_context(predictions, ids, strands, output_path):
    """保存包含上下文信息的预测结果到 HDF5 文件"""
    with h5py.File(output_path, 'w') as hf:
        for idx, pred in enumerate(predictions):  # 遍历所有预测结果
            id_str = ids[idx]  # 获取当前样本的ID
            strand = strands[idx]  # 获取当前样本的strand（正链或负链）

            # 在HDF5文件中创建一个分组，以id和strand为组名
            group = hf.create_group(f'{id_str}/{strand}')
            group.create_dataset('predictions', data=pred) # 在分组下创建一个dataset保存预测结果
'''
# 测试
model.load_state_dict(torch.load('best_CNNRB_model1.0.pth'))
model.eval()
test_loss = 0
f1_metric.reset()
conf_matrix_metric.reset()

all_predictions = []
all_ids = []
all_strands = [] #还没预测

with torch.no_grad():
    for inputs, labels in test_loader:  # 增加引入掩码矩阵,新增id和strand信息
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        labels = labels.view(-1)
        outputs = outputs.reshape(-1, output_size)

        loss = criterion(outputs, labels).mean()
        test_loss += loss.item()

        # 获取每个碱基的预测标签，torch.max(input, dim)返回每行的：最大值（忽略） 和 对应的索引
        _, test_preds = torch.max(outputs, 1)

        # 更新 F1 分数和混淆矩阵计算器
        f1_metric.update(test_preds, labels)
        conf_matrix_metric.update(test_preds, labels)

# 所有预测结果保存为h5文件
#save_predictions_with_context(all_predictions, all_ids, all_strands, 'deep_learning_predictions.h5')

avg_test_loss = test_loss / len(test_loader)
test_accuracy = torchmetrics.functional.accuracy(test_preds, labels.view(-1), num_classes=output_size, task="binary")
# task设置为2分类，预测类别为 output_size = 2
test_f1_score = f1_metric.compute()
test_conf_matrix = conf_matrix_metric.compute()
# 计算准确率的另一种方法
correct_predictions = (test_preds == labels).sum().item()  # 预测正确的数量
total_predictions = labels.numel()  # 总预测数量
manual_accuracy = correct_predictions / total_predictions

print(f'Manual Test Accuracy: {manual_accuracy:.4f}')

print(f'Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Test F1 Score: {test_f1_score}')
print("Test Confusion Matrix:")
print(test_conf_matrix)