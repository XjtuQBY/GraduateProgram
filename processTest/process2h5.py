import os
import sys
from Bio import SeqIO
from Bio.Seq import Seq
from BCBio import GFF
import h5py
import numpy as np


def process_genome(gff_file, fna_file, output_file, window=2000):
    # 提取信息
    forward_labels, reverse_labels, sequences = extract_info(gff_file, fna_file)

    # 存储到HDF5文件
    store_in_hdf5(forward_labels, reverse_labels, sequences, output_file, window)
    print(f"结果已存储到 {output_file}")


def extract_info(gff_file, fna_file):
    # 读取序列文件
    with open(fna_file, "r") as handle:
        genome_seq_dict = SeqIO.to_dict(SeqIO.parse(handle, "fasta"))

    forward_labels = {}
    reverse_labels = {}
    sequences = {}
    print_count = 0

    # 解析GFF文件
    with open(gff_file) as gff:
        for rec in GFF.parse(gff):
            if print_count < 5:
                print(f"Extracting record ID: {rec.id}")
                print_count += 1

            sequence = str(genome_seq_dict[rec.id].seq)
            seq_len = len(sequence)
            forward_label = np.zeros(seq_len, dtype=np.uint8)
            reverse_label = np.zeros(seq_len, dtype=np.uint8)

            for feature in rec.features:
                if feature.type == "gene":
                    max_cds = None
                    for sub_feature in feature.sub_features:
                        if sub_feature.type == "CDS":
                            if max_cds is None or len(sub_feature.location) > len(max_cds.location):
                                max_cds = sub_feature

                    if max_cds is not None:
                        strand = max_cds.location.strand
                        cds_start = max_cds.location.start
                        cds_end = max_cds.location.end

                        if strand == 1:
                            forward_label[cds_start:cds_end] = 1
                        else:
                            reverse_label[cds_start:cds_end] = 1

            forward_labels[rec.id] = forward_label
            reverse_labels[rec.id] = reverse_label[::-1]
            sequences[rec.id] = sequence

    return forward_labels, reverse_labels, sequences


def store_in_hdf5(forward_labels, reverse_labels, sequences, output_file, window):
    with h5py.File(output_file, 'w') as f:
        print_count = 0
        for rec_id, forward_label in forward_labels.items():
            if print_count < 5:
                print(f"Processing record ID: {rec_id}")
                print_count += 1

            reverse_label = reverse_labels[rec_id]
            sequence = sequences[rec_id]
            seq_len = len(sequence)

            grp_forward = f.create_group(f"{rec_id}/forward")
            for start in range(0, seq_len, window):
                end = min(start + window, seq_len)
                start_str = str(start).zfill(7)
                end_str = str(end).zfill(7)

                grp_label = forward_label[start:end]
                grp_label = np.pad(grp_label, (0, window - len(grp_label)), 'constant')
                if np.all(grp_label == 0):
                    continue

                grp_forward.create_dataset(f"label_{start_str}:{end_str}", data=grp_label)
                sequence_chunk = sequence[start:end]
                if end != seq_len:
                    grp_forward.create_dataset(f"sequence_{start_str}:{end_str}",
                                               data=np.array(list(sequence_chunk), dtype='|S1'))
                else:
                    grp_forward.create_dataset(f"sequence_{start_str}:{end_str}",
                                               data=np.array(list(sequence_chunk) + ['N'] * (window - (end - start)), dtype='|S1'))

            grp_reverse = f.create_group(f"{rec_id}/reverse")
            for start in range(0, seq_len, window):
                end = min(start + window, seq_len)
                start_str = str(start).zfill(7)
                end_str = str(end).zfill(7)

                grp_label = reverse_label[start:end]
                grp_label = np.pad(grp_label, (0, window - len(grp_label)), 'constant')
                if np.all(grp_label == 0):
                    continue

                grp_reverse.create_dataset(f"label_{start_str}:{end_str}", data=grp_label)
                reverse_complement_seq = str(Seq(sequence[start:end]).reverse_complement())
                if end != seq_len:
                    grp_reverse.create_dataset(f"sequence_{start_str}:{end_str}",
                                               data=np.array(list(reverse_complement_seq), dtype='|S1'))
                else:
                    grp_reverse.create_dataset(f"sequence_{start_str}:{end_str}",
                                               data=np.array(list(reverse_complement_seq) + ['N'] * (window - (end - start)), dtype='|S1'))


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("用法: python process_to_h5.py <gff_file> <fna_file> <output_h5_file>")
        sys.exit(1)

    gff_file = sys.argv[1]
    fna_file = sys.argv[2]
    output_h5_file = sys.argv[3]

    process_genome(gff_file, fna_file, output_h5_file)
