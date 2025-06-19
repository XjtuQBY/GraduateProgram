import h5py
import torch
import os
import glob
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler


class GenomeDataset(Dataset):
    def __init__(self, h5_files, rank, world_size, transform=None):
        self.h5_files = h5_files
        self.rank = rank
        self.world_size = world_size
        self.transform = transform

        # 调用 load_data_from_h5 只加载当前进程负责的部分数据
        self.data = self.load_data_from_h5(h5_files)

    def load_data_from_h5(self, h5_files):
        data = []
        # 仅加载属于当前 rank 进程的数据部分
        for idx, h5_file in enumerate(h5_files):
            if idx % self.world_size == self.rank:  # 根据 rank 和 world_size 进行分割
                print(f"Rank {self.rank} loading file: {h5_file}")
                with h5py.File(h5_file, 'r') as f:
                    for id_group in f.keys():
                        for strand in ['forward', 'reverse']:
                            grp = f[f'{id_group}/{strand}']
                            label_keys = [key for key in grp.keys() if key.startswith('label_')]
                            sequence_keys = [key for key in grp.keys() if key.startswith('sequence_')]

                            for label_key, sequence_key in zip(label_keys, sequence_keys):
                                label = grp[label_key][()]
                                byte_sequence = grp[sequence_key][()]
                                str_sequence = ''.join([byte.decode('utf-8') for byte in byte_sequence])

                                if len(str_sequence) != 2000:
                                    print(f"Warning: Sequence length {len(str_sequence)} in file {h5_file} is not 2000.")

                                data.append((str_sequence, label))
        return data

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

# prepare_distributed_dataloaders 需要修改以传递 rank 和 world_size
def prepare_distributed_dataloaders(data_path, rank, world_size, batch_size=32, num_workers=4):
    all_files = glob.glob(os.path.join(data_path, "*.h5"))

    train_files = all_files[:int(0.8 * len(all_files))]
    val_files = all_files[int(0.8 * len(all_files)):int(0.9 * len(all_files))]
    test_files = all_files[int(0.9 * len(all_files)):]

    # 创建数据集时传递 rank 和 world_size 参数
    train_dataset = GenomeDataset(train_files, rank, world_size)
    val_dataset = GenomeDataset(val_files, rank, world_size)
    test_dataset = GenomeDataset(test_files, rank, world_size)

    return train_dataset, val_dataset, test_dataset
