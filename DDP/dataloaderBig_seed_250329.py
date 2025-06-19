import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import glob
import os
import random
import json
from datetime import datetime

# 设置随机种子以确保可重复性
RANDOM_SEED = 523
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# 定义自定义数据集
class GenomeDataset(Dataset):
    def __init__(self, h5_files, transform=None):
        self.h5_files = h5_files
        self.transform = transform
        self.data = load_data_from_h5(h5_files)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence, label = self.data[idx]

        # 将sequence的小写转换为大写
        sequence = sequence.upper()

        # 转为one-hot编码
        encoded_sequence = []
        for base in sequence:
            if base == 'A':
                encoded_sequence.append([1, 0, 0, 0])
            elif base == 'G':
                encoded_sequence.append([0, 1, 0, 0])
            elif base == 'T':
                encoded_sequence.append([0, 0, 1, 0])
            elif base == 'C':
                encoded_sequence.append([0, 0, 0, 1])
            else:
                encoded_sequence.append([0.25, 0.25, 0.25, 0.25])  # 补的N

        sequence = torch.tensor(encoded_sequence, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        return sequence, label


def load_data_from_h5(h5_files):
    data = []
    for h5_file in h5_files:
        print(f"Loading file: {h5_file}")  # 添加调试信息
        with h5py.File(h5_file, 'r') as f:
            for id_group in f.keys():
                for strand in ['forward', 'reverse']:
                    grp = f[f'{id_group}/{strand}']
                    label_keys = []
                    sequence_keys = []
                    for key in grp.keys():
                        if key.startswith('label_'):
                            label_keys.append(key)
                        elif key.startswith('sequence_'):
                            sequence_keys.append(key)

                    for label_key, sequence_key in zip(label_keys, sequence_keys):
                        label = grp[label_key][()]
                        # 字节序列转为字符串形式
                        byte_sequence = grp[sequence_key][()]
                        str_sequence = ''
                        for byte in byte_sequence:
                            str_sequence += byte.decode('utf-8')
                        sequence = str_sequence
                        # 检查序列长度是否为预期的2000
                        if len(sequence) != 2000:
                            print(f"Warning: Sequence length {len(sequence)} in file {h5_file} at group {id_group}/{strand} is not 2000.")
                        data.append((sequence, label))
    return data

# 获取数据文件路径
data_path = "/data/home/fbchou/RandomSampledBacteria_Process2h5_filtered/"
all_files = glob.glob(os.path.join(data_path, "*.h5"))

# 打乱文件路径列表
random.shuffle(all_files)

# 分割数据集
train_files = all_files[:int(0.8 * len(all_files))]  # 80% 作为训练集
val_files = all_files[int(0.8 * len(all_files)):int(0.9 * len(all_files))]  # 10% 作为验证集
test_files = all_files[int(0.9 * len(all_files)):]  # 10% 作为测试集

# 创建数据集划分信息
dataset_split = {
    "random_seed": RANDOM_SEED,
    "total_species": len(all_files),
    "split_ratio": "80-10-10",
    "datetime": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    "training_set": [os.path.basename(f).replace('.h5', '') for f in train_files],
    "validation_set": [os.path.basename(f).replace('.h5', '') for f in val_files],
    "test_set": [os.path.basename(f).replace('.h5', '') for f in test_files]
}

# 保存数据集划分信息
split_info_path = os.path.join("/data/home/fbchou/deepLearning/DDP/", "dataset_split_info.json")
with open(split_info_path, 'w') as f:
    json.dump(dataset_split, f, indent=4)

print(f"数据集划分信息已保存到: {split_info_path}")
print(f"训练集大小: {len(train_files)} 物种")
print(f"验证集大小: {len(val_files)} 物种")
print(f"测试集大小: {len(test_files)} 物种")

# 创建数据集和数据加载器
train_dataset = GenomeDataset(train_files)
val_dataset = GenomeDataset(val_files)
test_dataset = GenomeDataset(test_files)

batch_size = 512

train_loader = DataLoader(train_dataset, batch_size= batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# 打印数据形状以验证
for sequences, labels in train_loader:
    print(f'Sequences shape: {sequences.shape}')
    print(f'Labels shape: {labels.shape}')
    break
