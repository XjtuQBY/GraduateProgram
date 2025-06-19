import h5py
import torch
from torch.utils.data import Dataset, DataLoader


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
                encoded_sequence.append([0, 0, 0, 0])  # 补的N

        sequence = torch.tensor(encoded_sequence, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        return sequence, label


def load_data_from_h5(h5_files):
    data = []
    for h5_file in h5_files:
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
                        data.append((sequence, label))
    return data


# 数据文件
train_files = [
'output1.h5','output2.h5','output3.h5','output4.h5','output5.h5','output6.h5','output7.h5','output8.h5','output9.h5','output10.h5',
'output11.h5','output12.h5','output13.h5','output14.h5','output15.h5','output16.h5','output17.h5','output18.h5','output19.h5','output20.h5',
'output21.h5','output22.h5','output23.h5','output24.h5','output25.h5','output26.h5','output27.h5','output28.h5','output29.h5','output30.h5',
'output31.h5','output32.h5','output33.h5','output34.h5','output35.h5','output36.h5','output37.h5','output38.h5','output39.h5','output40.h5',
'output41.h5','output42.h5','output43.h5','output44.h5','output45.h5','output46.h5','output47.h5','output48.h5','output49.h5','output50.h5',
'output51.h5','output52.h5','output53.h5','output54.h5','output55.h5','output56.h5','output57.h5','output58.h5','output59.h5','output60.h5',
'output61.h5','output62.h5','output63.h5','output64.h5','output65.h5','output66.h5','output67.h5','output68.h5','output69.h5','output70.h5',
'output71.h5','output72.h5','output73.h5','output74.h5']
val_files = ['output75.h5','output76.h5','output77.h5','output78.h5','output79.h5','output80.h5','output81.h5','output82.h5','output83.h5','output84.h5']
test_files = ['output84.h5']

# 创建数据集和数据加载器
train_dataset = GenomeDataset(train_files)
val_dataset = GenomeDataset(val_files)
test_dataset = GenomeDataset(test_files)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 打印数据形状以验证
for sequences, labels in train_loader:
    print(f'Sequences shape: {sequences.shape}')
    print(f'Labels shape: {labels.shape}')
    break
