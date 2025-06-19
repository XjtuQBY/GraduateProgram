import os
import re
import glob
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from tqdm import tqdm


class InputExample:
    def __init__(self, unique_id, text_a, text_b=None, label=None):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


def read_examples(input_pattern, label=None):
    """读取输入文件，支持通配符模式"""
    examples = []
    unique_id = 0

    # 使用glob处理通配符
    input_files = glob.glob(input_pattern)
    print(f"找到 {len(input_files)} 个文件匹配模式 '{input_pattern}'")

    for input_file in tqdm(input_files, desc="读取文件"):
        with open(input_file, "r") as reader:
            for line in reader:
                line = line.strip()
                if not line:  # 跳过空行
                    continue

                text_a = None
                text_b = None

                # 尝试匹配原始实现的文本分割模式
                m = re.match(r"^(.*) \|\|\| (.*)$", line)
                if m is None:
                    text_a = line
                else:
                    text_a = m.group(1)
                    text_b = m.group(2)

                examples.append(
                    InputExample(
                        unique_id=unique_id,
                        text_a=text_a,
                        text_b=text_b,
                        label=label
                    )
                )
                unique_id += 1

    print(f"共读取 {len(examples)} 个序列样本")
    return examples


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """按照原始实现截断序列对"""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def kmer_tokenize(sequence, k=6):
    """
    将DNA序列转换为k-mer tokens

    参数:
    sequence: DNA序列
    k: k-mer大小，默认为6

    返回:
    list: k-mer tokens列表
    """
    tokens = []
    sequence = sequence.upper()  # 转换为大写

    # 确保序列长度足够生成至少一个k-mer
    if len(sequence) < k:
        return []

    # 生成k-mers
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i + k]
        # 检查k-mer是否只包含ACGT
        if re.match(r'^[ACGT]+$', kmer):
            tokens.append(kmer)

    return tokens


def convert_examples_to_features(examples, max_seq_length, tokenizer, k=6):
    """将示例转换为模型可接受的特征，使用k-mer分词"""
    features = []
    for (ex_index, example) in enumerate(tqdm(examples, desc="转换特征")):
        # 使用k-mer分词替代原始分词
        tokens_a = kmer_tokenize(example.text_a, k)

        tokens_b = None
        if example.text_b:
            tokens_b = kmer_tokenize(example.text_b, k)

        if tokens_b:
            # 按照原始实现，为序列对预留3个特殊标记位置
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # 为单序列预留2个特殊标记位置
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]

        # 按BERT约定添加特殊标记
        tokens = []
        tokens.append("[CLS]")
        tokens.extend(tokens_a)
        tokens.append("[SEP]")

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
            tokens.append("[SEP]")

        # 转换为ID
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # 创建输入掩码
        input_mask = [1] * len(input_ids)

        # 填充到指定长度
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length

        if ex_index < 5:
            print(f"*** 示例 {ex_index} ***")
            print(f"唯一ID: {example.unique_id}")
            print(f"标记: {' '.join(tokens[:10])}... (共{len(tokens)}个)")
            print(f"输入IDs: {' '.join([str(x) for x in input_ids[:10]])}... (共{len(input_ids)}个)")

        features.append({
            "unique_id": example.unique_id,
            "tokens": tokens,
            "input_ids": input_ids,
            "input_mask": input_mask,
            "label": example.label
        })

    return features


def extract_bert_features_numpy(input_pattern, output_prefix, bert_model_path, max_seq_length=510, k=6,
                                do_lower_case=False, label=None, exclude_special_tokens=True):
    """
    从输入序列提取BERT特征并保存为多种格式，只提取最后一层

    参数:
    input_pattern: 输入文件模式，支持通配符
    output_prefix: 输出文件前缀（会添加后缀.npz）
    bert_model_path: BERT模型路径
    max_seq_length: 最大序列长度
    k: k-mer大小
    do_lower_case: 是否将文本转为小写
    label: 数据标签（0表示非启动子，1表示启动子）
    exclude_special_tokens: 是否排除[CLS]和[SEP]等特殊标记
    """
    # 构建输出文件路径
    npz_file = f"{output_prefix}.npz"

    print(f"开始处理: {input_pattern}")
    print(f"输出NPZ文件: {npz_file}")

    print(f"特殊标记处理: {'排除' if exclude_special_tokens else '包含'}")

    # 加载分词器和模型
    print(f"加载DNABERT模型: {bert_model_path}")
    tokenizer = BertTokenizer.from_pretrained(bert_model_path)
    model = BertModel.from_pretrained(bert_model_path)

    # 检查是否有GPU可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    model = model.to(device)
    model.eval()  # 设置为评估模式

    # 读取并处理示例
    examples = read_examples(input_pattern, label)
    if not examples:
        print(f"警告: 没有找到匹配 {input_pattern} 的样本!")
        return 0

    features = convert_examples_to_features(examples, max_seq_length, tokenizer, k)

    # 创建输出目录
    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
    print(f"输出目录: {os.path.dirname(output_prefix)}")

    # 提取特征
    print(f"开始提取特征...")

    # 准备存储所有特征的列表
    all_features = []
    all_ids = []
    all_labels = []

    with torch.no_grad():
        for feature in tqdm(features, desc="提取特征"):
            # 准备输入
            input_ids = torch.tensor([feature["input_ids"]], dtype=torch.long).to(device)
            input_mask = torch.tensor([feature["input_mask"]], dtype=torch.long).to(device)

            # 模型前向传播
            outputs = model(input_ids,
                            attention_mask=input_mask,
                            output_hidden_states=True)

            # 获取隐藏状态 (只使用最后一层)
            hidden_states = outputs.hidden_states[-1]  # 最后一层

            # 收集所有token的特征向量
            seq_features = hidden_states[0].cpu().numpy()

            # 如果排除特殊标记，则只保留真实k-mer的特征
            if exclude_special_tokens:
                # 排除第一个[CLS]和最后一个[SEP]标记
                seq_features = seq_features[1:-1]

            # 添加到列表
            all_features.append(seq_features)
            all_ids.append(feature["unique_id"])
            all_labels.append(feature["label"])

    # 转换为NumPy数组
    all_features = np.array(all_features)
    all_ids = np.array(all_ids)
    all_labels = np.array(all_labels)

    # 打印特征形状
    print(f"特征形状: {all_features.shape}")  # 如果排除特殊标记，应该是 (样本数, seq_length-2, 隐藏状态维度)

    # 保存为.npz格式
    print(f"保存特征为NPZ格式: {npz_file}")
    np.savez(
        npz_file,
        features=all_features,  # shape: (num_samples, actual_seq_length, hidden_size)
        ids=all_ids,  # shape: (num_samples,)
        labels=all_labels  # shape: (num_samples,)
    )

    print(f"特征提取完成! 已处理 {len(all_features)} 个样本")

    return len(all_features)


def process_promoter_data_numpy(promoter_input, non_promoter_input, output_dir, bert_model_path, max_seq_length=510,
                                k=6, exclude_special_tokens=True):
    """
    处理启动子和非启动子数据的完整流程，保存为NumPy格式

    参数:
    promoter_input: 正样本输入文件路径（支持通配符）
    non_promoter_input: 负样本输入文件路径（支持通配符）
    output_dir: 输出目录路径
    bert_model_path: BERT模型路径
    max_seq_length: 最大序列长度
    k: k-mer大小
    exclude_special_tokens: 是否排除[CLS]和[SEP]等特殊标记
    """
    # 确保输出目录存在
    features_dir = output_dir
    if not os.path.exists(features_dir):
        os.makedirs(features_dir)

    # 定义输出文件路径前缀
    promoter_output_prefix = os.path.join(features_dir, "promoter_features")
    non_promoter_output_prefix = os.path.join(features_dir, "non_promoter_features")

    # 处理正样本
    print("\n" + "=" * 50)
    print("处理正样本 (Promoter) ")
    print("=" * 50)
    promoter_count = extract_bert_features_numpy(
        promoter_input,
        promoter_output_prefix,
        bert_model_path,
        max_seq_length,
        k,
        label=1,  # 启动子标签为1
        exclude_special_tokens=exclude_special_tokens
    )

    # 处理负样本
    print("\n" + "=" * 50)
    print("处理负样本 (Non-Promoter) ")
    print("=" * 50)
    non_promoter_count = extract_bert_features_numpy(
        non_promoter_input,
        non_promoter_output_prefix,
        bert_model_path,
        max_seq_length,
        k,
        label=0,  # 非启动子标签为0
        exclude_special_tokens=exclude_special_tokens
    )

    print("\n数据处理完成!")
    print(f"正样本: {promoter_count} 个")
    print(f"负样本: {non_promoter_count} 个")
    print(f"总样本: {promoter_count + non_promoter_count} 个")

    return {
        'promoter_npz': f"{promoter_output_prefix}.npz",
        'non_promoter_npz': f"{non_promoter_output_prefix}.npz",
        'promoter_count': promoter_count,
        'non_promoter_count': non_promoter_count
    }


def main():
    # 设置当前工作目录
    full_path = os.path.realpath(__file__)
    os.chdir(os.path.dirname(full_path))

    # 基本路径
    base_input_path = "/data/home/fbchou/deepLearning/BERT_DNN_Promoter/data/singleSeq"
    base_output_path = "/data/home/fbchou/deepLearning/BERT_DNN_Promoter/data/features"
    bert_model_path = "/data/home/fbchou/deepLearning/BERT_DNN_Promoter/DNA_bert_6"

    # 需要处理的数据集列表
    datasets = ["Mouse_non_tata", "Mouse_tata"]

    # k-mer大小
    k = 6

    # 设置序列长度和对应的token数量
    sequence_length = 251  # 你的实际序列长度，例如40bp
    seq_length = sequence_length - k + 1  # 计算k-mer数量

    # 是否排除特殊标记
    exclude_special_tokens = True

    # 计算需要的max_seq_length
    # 如果我们要排除特殊标记，依然需要在模型输入中包含它们
    max_seq_length = seq_length + 2  # 加上[CLS]和[SEP]标记的长度

    print(f"使用k-mer大小: {k}")
    print(f"输入序列长度: {sequence_length}bp")
    print(f"计算得到的token数量: {seq_length}")
    print(f"设置的max_seq_length: {max_seq_length}")
    print(f"特殊标记处理: {'排除' if exclude_special_tokens else '包含'}")

    # 循环处理每个数据集
    for dataset in datasets:
        print("\n" + "=" * 80)
        print(f"开始处理数据集: {dataset}")
        print("=" * 80)

        # 构建当前数据集的输入输出路径
        promoter_input = f"{base_input_path}/{dataset}/promoter/*.seq"
        non_promoter_input = f"{base_input_path}/{dataset}/non_promoter/*.seq"
        output_dir = f"{base_output_path}/{dataset}"

        print(f"promoter路径: {promoter_input}")
        print(f"non_promoter路径: {non_promoter_input}")
        print(f"输出目录: {output_dir}")

        # 处理数据，使用6-mer，保存为NumPy格式
        print("\n" + "=" * 50)
        print(f"提取{dataset}数据集的BERT特征并保存为中间格式")
        print("=" * 50)

        try:
            results = process_promoter_data_numpy(
                promoter_input,
                non_promoter_input,
                output_dir,
                bert_model_path,
                max_seq_length=max_seq_length,  # 使用计算的长度而不是固定值
                k=k,
                exclude_special_tokens=exclude_special_tokens
            )

            print(f"\n数据集 {dataset} 生成的文件:")
            print(f"- NPZ文件: {results['promoter_npz']}, {results['non_promoter_npz']}")

        except Exception as e:
            print(f"\n处理数据集 {dataset} 时发生错误:")
            print(f"错误信息: {str(e)}")
            print("继续处理下一个数据集...")

    print("\n" + "=" * 50)
    print("所有数据集的BERT特征提取完成!请在process2h5环境中运行npz2h5.py来生成最终的H5文件")
    print("=" * 50)


if __name__ == "__main__":
    main()
