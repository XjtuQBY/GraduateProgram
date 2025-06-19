import os
import re


def extract_seq(file_path, dir_path='seq', seq_length=510):
    """
    从FASTA文件中提取序列并保存到指定目录
    不在碱基间添加空格，使用纯数字命名文件。

    Args:
        file_path: FASTA文件路径
        dir_path: 输出目录路径
        seq_length: 序列长度
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    nseq = 0  # 计数序列数量
    nsmp = 0  # 计数样本数量
    data = re.split(
        r'(^>.*)', ''.join(open(file_path).readlines()), flags=re.M)

    seq_count = 1  # 每个文件夹独立计数，从1开始

    for i in range(2, len(data), 2):
        fasta = data[i].replace('\n', '').replace('\x1a', '')
        seq = [fasta[j:j + seq_length] for j in range(0, len(fasta), seq_length)]
        nsmp += len(seq)

        # 使用数字命名文件
        with open(f"{dir_path}/{seq_count}.seq", "w") as ffas:
            ffas.write('\n'.join(seq))

        nseq += 1
        seq_count += 1  # 每个序列编号增加

    return nseq, nsmp


def process_fasta_batch(batch_data, base_output_dir, seq_length=510):
    """
    批量处理多组正负例文件

    Args:
        batch_data: 包含多组数据的列表，每组数据为(正例文件路径, 负例文件路径, 输出目录标识符)
        base_output_dir: 基础输出目录
        seq_length: 序列长度
    """
    total_pos_seq = 0
    total_pos_smp = 0
    total_neg_seq = 0
    total_neg_smp = 0

    for pos_file, neg_file, folder_id in batch_data:
        print(f"\n处理组 {folder_id}:")
        print(f"正例文件: {pos_file}")
        print(f"负例文件: {neg_file}")

        # 创建输出目录
        pos_output_dir = os.path.join(base_output_dir, folder_id, "promoter")
        neg_output_dir = os.path.join(base_output_dir, folder_id, "non_promoter")

        # 处理正例
        pos_nseq, pos_nsmp = extract_seq(pos_file, pos_output_dir, seq_length)
        total_pos_seq += pos_nseq
        total_pos_smp += pos_nsmp
        print(f"正例 - 序列数: {pos_nseq}, 样本数: {pos_nsmp}")

        # 处理负例
        neg_nseq, neg_nsmp = extract_seq(neg_file, neg_output_dir, seq_length)
        total_neg_seq += neg_nseq
        total_neg_smp += neg_nsmp
        print(f"负例 - 序列数: {neg_nseq}, 样本数: {neg_nsmp}")

    print("\n总计:")
    print(f"正例 - 总序列数: {total_pos_seq}, 总样本数: {total_pos_smp}")
    print(f"负例 - 总序列数: {total_neg_seq}, 总样本数: {total_neg_smp}")


if __name__ == "__main__":
    # 设置工作目录为脚本所在目录
    full_path = os.path.realpath(__file__)
    os.chdir(os.path.dirname(full_path))
    print(f"Change CWD to: {os.path.dirname(full_path)}")

    # 在这里直接配置批处理任务
    # 格式: (正例文件路径, 负例文件路径, 输出目录标识符)
    batch_data = [
        # 训练
        (
            "/data/home/fbchou/deepLearning/BERT_DNN_Promoter/data/origin/251bp/Arabidopsis_non_tata.fa",
            "/data/home/fbchou/deepLearning/BERT_DNN_Promoter/data/origin/251bp/Arabidopsis_non_tata_nonprom.fa",
            'Arabidopsis_non_tata'
        ),
        (
            "/data/home/fbchou/deepLearning/BERT_DNN_Promoter/data/origin/251bp/Arabidopsis_tata.fa",
            "/data/home/fbchou/deepLearning/BERT_DNN_Promoter/data/origin/251bp/Arabidopsis_nonprom.fa",
            'Arabidopsis_tata'
        ),
        (
            "/data/home/fbchou/deepLearning/BERT_DNN_Promoter/data/origin/251bp/human_non_tata.fa",
            "/data/home/fbchou/deepLearning/BERT_DNN_Promoter/data/origin/251bp/human_nonprom.fa",
            'human_non_tata'
        ),
        (
            "/data/home/fbchou/deepLearning/BERT_DNN_Promoter/data/origin/251bp/Mouse_non_tata.fa",
            "/data/home/fbchou/deepLearning/BERT_DNN_Promoter/data/origin/251bp/Mouse_non_tata_nonprom.fa",
            'Mouse_non_tata'
        ),
        (
            "/data/home/fbchou/deepLearning/BERT_DNN_Promoter/data/origin/251bp/Mouse_tata.fa",
            "/data/home/fbchou/deepLearning/BERT_DNN_Promoter/data/origin/251bp/Mouse_nonprom.fa",
            'Mouse_tata'
        )
        # 添加更多任务...
    ]

    # 设置基础输出目录
    base_output_dir = "/data/home/fbchou/deepLearning/BERT_DNN_Promoter/data/singleSeq/"

    # 设置序列长度
    seq_length = 510

    # 执行批处理
    process_fasta_batch(batch_data, base_output_dir, seq_length)
