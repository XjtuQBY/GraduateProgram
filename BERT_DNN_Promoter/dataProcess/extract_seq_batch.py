import os
import re


def clean_filename(filename):
    """
    清理文件名中的非法字符，替换为下划线。
    """
    # 定义非法字符的正则表达式
    illegal_chars = r'[/\\:*?"<>|$$$$()\s]'
    # 将所有非法字符替换为下划线
    return re.sub(illegal_chars, '_', filename)


def extract_seq(file_path, dir_path='seq', seq_length=510):
    """
    从FASTA文件中提取序列并保存到指定目录
    不在碱基间添加空格

    Args:
        file_path: FASTA文件路径
        dir_path: 输出目录路径
        seq_length: 序列长度
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    nseq = 0
    nsmp = 0
    data = re.split(
        r'(^>.*)', ''.join(open(file_path).readlines()), flags=re.M)
    for i in range(2, len(data), 2):
        fid = data[i - 1][1:].split('|')[0]

        # 清理文件名中的非法字符
        fid = clean_filename(fid)

        nseq = nseq + 1
        fasta = data[i].replace('\n', '').replace('\x1a', '')

        # 不添加空格，直接分段
        seq = [fasta[j:j + seq_length] for j in range(0, len(fasta), seq_length)]

        nsmp = nsmp + len(seq)
        ffas = open(f"{dir_path}/{fid}.seq", "w")
        ffas.write('\n'.join(seq))
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
            "/data/home/fbchou/deepLearning/BERT_DNN_Promoter/data/origin/C_JEJUNI_1_positive.fasta",
            "/data/home/fbchou/deepLearning/BERT_DNN_Promoter/data/origin/C_JEJUNI_1_negative.fasta",
            'C_JEJUNI_1'
        ),
        (
            "/data/home/fbchou/deepLearning/BERT_DNN_Promoter/data/origin/C_JEJUNI_2_positive.fasta",
            "/data/home/fbchou/deepLearning/BERT_DNN_Promoter/data/origin/C_JEJUNI_2_negative.fasta",
            'C_JEJUNI_2'
        ),
        (
            "/data/home/fbchou/deepLearning/BERT_DNN_Promoter/data/origin/C_JEJUNI_3_positive.fasta",
            "/data/home/fbchou/deepLearning/BERT_DNN_Promoter/data/origin/C_JEJUNI_3_negative.fasta",
            'C_JEJUNI_3'
        ),
        (
            "/data/home/fbchou/deepLearning/BERT_DNN_Promoter/data/origin/C_JEJUNI_4_positive.fasta",
            "/data/home/fbchou/deepLearning/BERT_DNN_Promoter/data/origin/C_JEJUNI_4_negative.fasta",
            'C_JEJUNI_4'
        ),
        (
            "/data/home/fbchou/deepLearning/BERT_DNN_Promoter/data/origin/C_JEJUNI_5_positive.fasta",
            "/data/home/fbchou/deepLearning/BERT_DNN_Promoter/data/origin/C_JEJUNI_5_negative.fasta",
            'C_JEJUNI_5'
        ),
        (
            "/data/home/fbchou/deepLearning/BERT_DNN_Promoter/data/origin/CPNEUMONIAE_positive.fasta",
            "/data/home/fbchou/deepLearning/BERT_DNN_Promoter/data/origin/CPNEUMONIAE_negative.fasta",
            'CPNEUMONIAE'
        ),
        (
            "/data/home/fbchou/deepLearning/BERT_DNN_Promoter/data/origin/ECOLI_1_positive.fasta",
            "/data/home/fbchou/deepLearning/BERT_DNN_Promoter/data/origin/ECOLI_1_negative.fasta",
            'ECOLI_1'
        ),
        (
            "/data/home/fbchou/deepLearning/BERT_DNN_Promoter/data/origin/ECOLI_2_positive.fasta",
            "/data/home/fbchou/deepLearning/BERT_DNN_Promoter/data/origin/ECOLI_2_negative.fasta",
            'ECOLI_2'
        ),
        (
            "/data/home/fbchou/deepLearning/BERT_DNN_Promoter/data/origin/HPYLORI_1_positive.fasta",
            "/data/home/fbchou/deepLearning/BERT_DNN_Promoter/data/origin/HPYLORI_1_negative.fasta",
            'HPYLORI_1'
        ),
        (
            "/data/home/fbchou/deepLearning/BERT_DNN_Promoter/data/origin/HPYLORI_2_positive.fasta",
            "/data/home/fbchou/deepLearning/BERT_DNN_Promoter/data/origin/HPYLORI_2_negative.fasta",
            'HPYLORI_2'
        ),
        (
            "/data/home/fbchou/deepLearning/BERT_DNN_Promoter/data/origin/LINTERROGANS_positive.fasta",
            "/data/home/fbchou/deepLearning/BERT_DNN_Promoter/data/origin/LINTERROGANS_negative.fasta",
            'LINTERROGANS'
        ),
        (
            "/data/home/fbchou/deepLearning/BERT_DNN_Promoter/data/origin/SCOELICOLOR_positive.fasta",
            "/data/home/fbchou/deepLearning/BERT_DNN_Promoter/data/origin/SCOELICOLOR_negative.fasta",
            'SCOELICOLOR'
        ),
        (
            "/data/home/fbchou/deepLearning/BERT_DNN_Promoter/data/origin/SONEIDENSIS_positive.fasta",
            "/data/home/fbchou/deepLearning/BERT_DNN_Promoter/data/origin/SONEIDENSIS_negative.fasta",
            'SONEIDENSIS'
        ),
        (
            "/data/home/fbchou/deepLearning/BERT_DNN_Promoter/data/origin/SPYOGENE_positive.fasta",
            "/data/home/fbchou/deepLearning/BERT_DNN_Promoter/data/origin/SPYOGENE_negative.fasta",
            'SPYOGENE'
        ),
        (
            "/data/home/fbchou/deepLearning/BERT_DNN_Promoter/data/origin/STYPHIRMURIUM_positive.fasta",
            "/data/home/fbchou/deepLearning/BERT_DNN_Promoter/data/origin/STYPHIRMURIUM_negative.fasta",
            'STYPHIRMURIUM'
        ),
        # 测试
        (
            "/data/home/fbchou/deepLearning/BERT_DNN_Promoter/data/origin/BACILLUS_positive.fasta",
            "/data/home/fbchou/deepLearning/BERT_DNN_Promoter/data/origin/BACILLUS_negative.fasta",
            'BACILLUS'
        ),
        (
            "/data/home/fbchou/deepLearning/BERT_DNN_Promoter/data/origin/CLOSTRIDIUM_positive.fasta",
            "/data/home/fbchou/deepLearning/BERT_DNN_Promoter/data/origin/CLOSTRIDIUM_negative.fasta",
            'CLOSTRIDIUM'
        ),
        (
            "/data/home/fbchou/deepLearning/BERT_DNN_Promoter/data/origin/MYCOBACTER_positive.fasta",
            "/data/home/fbchou/deepLearning/BERT_DNN_Promoter/data/origin/MYCOBACTER_negative.fasta",
            'MYCOBACTER'
        ),
        (
            "/data/home/fbchou/deepLearning/BERT_DNN_Promoter/data/origin/RHODOBACTER_1_positive.fasta",
            "/data/home/fbchou/deepLearning/BERT_DNN_Promoter/data/origin/RHODOBACTER_1_negative.fasta",
            'RHODOBACTER_1'
        ),
        (
            "/data/home/fbchou/deepLearning/BERT_DNN_Promoter/data/origin/RHODOBACTER_2_positive.fasta",
            "/data/home/fbchou/deepLearning/BERT_DNN_Promoter/data/origin/RHODOBACTER_2_negative.fasta",
            'RHODOBACTER_2'
        )
        # 添加更多任务...
    ]

    # 设置基础输出目录
    base_output_dir = "/data/home/fbchou/deepLearning/BERT_DNN_Promoter/data/singleSeq/"

    # 设置序列长度
    seq_length = 510

    # 执行批处理
    process_fasta_batch(batch_data, base_output_dir, seq_length)