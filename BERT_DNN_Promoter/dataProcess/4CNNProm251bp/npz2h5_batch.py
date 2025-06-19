import numpy as np
import h5py
import os
import time
import argparse


def convert_npz_to_h5(npz_file, h5_file, seq_length=None, no_reshape=False):
    """
    将NPZ格式转换为H5格式，可选择重塑特征

    参数:
    npz_file: 输入NPZ文件路径
    h5_file: 输出H5文件路径
    seq_length: 如果提供，将每seq_length个向量视为一个样本进行重塑
    no_reshape: 如果为True，则不进行重塑，原样保留特征(例如当特征已经不包含特殊标记时)
    """
    print(f"转换 {npz_file} 到 {h5_file}")

    # 加载NPZ文件
    data = np.load(npz_file)
    features = data['features']
    # 根据文件名或其他条件生成标签，要先判断负样本，否则会全是 label 1
    if 'non_promoter' in npz_file:
        labels = np.zeros(features.shape[0], dtype=np.int32)  # 负样本标签为0
    elif 'promoter' in npz_file:
        labels = np.ones(features.shape[0], dtype=np.int32)  # 正样本标签为1
    else:
        raise ValueError("无法识别文件类型")

    print(f"原始数据形状: 特征 {features.shape}, 标签 {labels.shape}")

    # 检查是否需要重塑 (当no_reshape=False且提供了seq_length)
    if seq_length is not None and not no_reshape:
        # 确认我们可以重塑
        original_shape = features.shape
        print(f"原始特征形状: {original_shape}")

        # 检查是否能被seq_length整除
        total_samples = features.shape[0]
        total_tokens = total_samples * features.shape[1]

        if total_tokens % seq_length != 0:
            print(f"警告: 总token数 ({total_tokens}) 不能被seq_length ({seq_length}) 整除")
            # 计算可用的token数
            usable_tokens = (total_tokens // seq_length) * seq_length
            print(f"将使用前 {usable_tokens} 个token，舍弃剩余的 {total_tokens - usable_tokens} 个")

            # 计算需要的完整样本数
            samples_needed = usable_tokens // features.shape[1]
            features = features[:samples_needed]
            labels = labels[:samples_needed]
            total_samples = samples_needed

        # 将3D特征 [样本数, token数, 特征维度] 展平为2D [总token数, 特征维度]
        flattened_features = features.reshape(-1, features.shape[2])
        print(f"展平后特征形状: {flattened_features.shape}")

        # 然后按seq_length重塑为新的样本
        new_sample_count = flattened_features.shape[0] // seq_length
        reshaped_features = flattened_features[:new_sample_count * seq_length].reshape(new_sample_count, seq_length,
                                                                                       features.shape[2])
        print(f"重塑后特征形状: {reshaped_features.shape}")

        # 扩展或压缩标签以匹配新的样本数
        if new_sample_count > total_samples:
            # 需要扩展标签 - 复制现有标签
            repeat_factor = new_sample_count // total_samples
            remainder = new_sample_count % total_samples

            new_labels = np.concatenate([
                np.repeat(labels, repeat_factor),
                labels[:remainder]
            ])
        else:
            # 需要压缩标签 - 每seq_length个样本取平均值并四舍五入
            new_labels = np.zeros(new_sample_count, dtype=np.int32)
            for i in range(new_sample_count):
                start_idx = i * seq_length // original_shape[1]
                end_idx = min((i + 1) * seq_length // original_shape[1], total_samples)
                if start_idx < end_idx:
                    # 取多数票作为新标签
                    new_labels[i] = np.round(np.mean(labels[start_idx:end_idx]))
                else:
                    # 边界情况，使用最后一个标签
                    new_labels[i] = labels[-1]

        print(f"调整后标签形状: {new_labels.shape}")

        # 更新特征和标签为重塑后的版本
        features = reshaped_features
        labels = new_labels.astype(np.int32)
    else:
        if no_reshape:
            print(f"跳过重塑，保留原始特征形状")
        else:
            print(f"未提供seq_length，保留原始特征形状")

    # 创建目录（如果不存在）
    os.makedirs(os.path.dirname(h5_file), exist_ok=True)

    # 创建H5文件
    with h5py.File(h5_file, 'w') as f:
        # 保存特征和标签
        print(f"创建X_data数据集...")
        f.create_dataset('X_data', data=features)
        print(f"创建y_labels数据集...")
        f.create_dataset('y_labels', data=labels, dtype=np.int32)

        # 保存元数据
        f.attrs['num_samples'] = features.shape[0]
        f.attrs['seq_length'] = features.shape[1]
        f.attrs['hidden_size'] = features.shape[2]

    print(f"H5文件创建成功: {h5_file}")
    print(f"最终数据形状: 特征 {features.shape}, 标签 {labels.shape}")

    return {
        'h5_file': h5_file,
        'features_shape': features.shape,
        'labels_shape': labels.shape
    }


def process_npz_files(input_dir, output_dir, sequence_length=40, k=6, exclude_special_tokens=True):
    """
    处理所有NPZ文件并转换为H5格式

    参数:
    input_dir: 包含NPZ文件的目录
    output_dir: 输出H5文件的目录
    sequence_length: 输入序列长度(bp)
    k: k-mer大小
    exclude_special_tokens: 特征中是否已排除特殊标记
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 计算实际的token数量
    token_count = sequence_length - k + 1
    print(f"输入序列长度: {sequence_length}bp")
    print(f"k-mer大小: {k}")
    print(f"计算token数量: {token_count}")
    print(f"特殊标记已排除: {exclude_special_tokens}")

    # 定义文件路径
    promoter_npz = os.path.join(input_dir, "promoter_features.npz")
    non_promoter_npz = os.path.join(input_dir, "non_promoter_features.npz")

    promoter_h5 = os.path.join(output_dir, "promoter_dataset.h5")
    non_promoter_h5 = os.path.join(output_dir, "non_promoter_dataset.h5")

    results = {}

    # 如果特征已经排除了特殊标记，我们应该跳过重塑
    # 因为特征已经是我们想要的形状
    no_reshape = exclude_special_tokens

    # 检查和处理每个文件
    if os.path.exists(promoter_npz):
        print("\n" + "=" * 50)
        print("处理正样本数据...")
        results['promoter'] = convert_npz_to_h5(
            promoter_npz,
            promoter_h5,
            seq_length=token_count,
            no_reshape=no_reshape
        )
    else:
        print(f"警告: 未找到正样本文件 {promoter_npz}")

    if os.path.exists(non_promoter_npz):
        print("\n" + "=" * 50)
        print("处理负样本数据...")
        results['non_promoter'] = convert_npz_to_h5(
            non_promoter_npz,
            non_promoter_h5,
            seq_length=token_count,
            no_reshape=no_reshape
        )
    else:
        print(f"警告: 未找到负样本文件 {non_promoter_npz}")


    return results


def main():
    # 记录开始时间
    import time
    start_time = time.time()

    # 设置基本配置（硬编码，不使用命令行参数）
    base_input_dir = "/data/home/fbchou/deepLearning/BERT_DNN_Promoter/data/features"
    base_output_dir = "/data/home/fbchou/deepLearning/BERT_DNN_Promoter/dataset"

    # 需要处理的数据集列表
    datasets = ["Arabidopsis_non_tata", "Arabidopsis_tata", "human_non_tata", "Mouse_non_tata", "Mouse_tata"]

    # 序列和k-mer配置
    sequence_length = 251  # 序列长度（bp）
    kmer = 6  # k-mer大小
    exclude_special_tokens = True  # 特征中是否已排除特殊标记

    print("\n" + "=" * 50)
    print("开始批量转换NPZ文件到H5格式")
    print("=" * 50)
    print(f"基本输入目录: {base_input_dir}")
    print(f"基本输出目录: {base_output_dir}")
    print(f"要处理的数据集: {datasets}")
    print(f"序列长度: {sequence_length}bp")
    print(f"k-mer大小: {kmer}")
    print(f"特殊标记已排除: {exclude_special_tokens}")

    # 用于存储所有数据集的结果
    all_results = {}

    # 循环处理每个数据集
    for dataset in datasets:
        print("\n" + "=" * 80)
        print(f"处理数据集: {dataset}")
        print("=" * 80)

        # 构建当前数据集的输入和输出路径
        input_dir = os.path.join(base_input_dir, dataset)
        output_dir = os.path.join(base_output_dir, dataset)

        print(f"输入目录: {input_dir}")
        print(f"输出目录: {output_dir}")

        # 检查输入目录是否存在
        if not os.path.exists(input_dir):
            print(f"警告: 输入目录 {input_dir} 不存在，跳过此数据集")
            continue

        try:
            # 处理当前数据集的NPZ文件
            results = process_npz_files(
                input_dir,
                output_dir,
                sequence_length,
                kmer,
                exclude_special_tokens=exclude_special_tokens
            )

            # 存储结果
            all_results[dataset] = results

            print(f"\n数据集 {dataset} 处理完成!")

        except Exception as e:
            print(f"\n处理数据集 {dataset} 时发生错误:")
            print(f"错误信息: {str(e)}")
            print("继续处理下一个数据集...")
            continue

    # 计算总用时
    elapsed_time = time.time() - start_time

    print("\n" + "=" * 50)
    print("所有数据集转换完成!")
    print(f"总用时: {elapsed_time:.2f} 秒")
    print("=" * 50)

    # 打印所有数据集的结果摘要
    print("\n所有生成的H5文件:")
    for dataset, results in all_results.items():
        print(f"\n数据集: {dataset}")
        for data_type, result in results.items():
            print(f"- {data_type}: {result['h5_file']}")
            print(f"  特征形状: {result['features_shape']}")
            print(f"  标签形状: {result['labels_shape']}")

    print("\n每个H5文件包含:")
    print(f"- X_data: 特征数据")
    print(f"- y_labels: 标签数据 (0表示非启动子，1表示启动子)")

    if exclude_special_tokens:
        print(f"\n注意: 输入特征中已排除[CLS]和[SEP]特殊标记")


if __name__ == "__main__":
    main()