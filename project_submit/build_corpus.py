# -*- encoding: utf-8 -*-
'''
@File        :build_corpus.py
@Time        :2021/05/06 08:55:11
@Author      :xiaoqifeng
@Version     :1.0
@Contact:    :unknown
'''

def build_pretrain_wo_aug(data_path, to_path):
    path_list = ['gaiic_track3_round1_train_20210228.tsv', 'gaiic_track3_round2_train_20210407.tsv', 'gaiic_track3_round1_testA_20210228.tsv', 'gaiic_track3_round1_testB_20210317.tsv']
    all_data = []

    for path in path_list:
        with open(origin_dir + path, 'r', encoding='utf-8') as reader:
            for record in reader:
                record_list = record.strip().split('\t')
                text_a, text_b = record_list[0], record_list[1]
                all_data.append(text_a + ' ' + text_b)

    with open(to_path, 'w', encoding='utf-8') as writer:
        for record in all_data:
            writer.write(record + '\n')

    print('saved successfully!')


def build_pretrain_aug(data_path, to_path):
    path_list = ['gaiic_track3_round1_train_20210228.tsv', 'gaiic_track3_round2_train_20210407.tsv', 'gaiic_track3_round1_testA_20210228.tsv', 'gaiic_track3_round1_testB_20210317.tsv']
    all_data = []

    for path in path_list:
        with open(origin_dir + path, 'r', encoding='utf-8') as reader:
            for record in reader:
                record_list = record.strip().split('\t')
                text_a, text_b = record_list[0], record_list[1]
                all_data.append(text_a + ' ' + text_b)
                all_data.append(text_b + ' ' + text_a)

    with open(to_path, 'w', encoding='utf-8') as writer:
        for record in all_data:
            writer.write(record + '\n')

    print('saved successfully!')


def build_train(data_path, to_path):
    path_list = ['gaiic_track3_round1_train_20210228.tsv', 'gaiic_track3_round2_train_20210407.tsv']
    all_data = []

    for path in path_list:
        with open(origin_dir + path, 'r', encoding='utf-8') as reader:
            for record in reader:
                all_data.append(record.strip())

    with open(to_path, 'w', encoding='utf-8') as writer:
        for record in all_data:
            writer.write(record + '\n')

    print('saved successfully!')


def build_test(to_path):
    with open(to_path, 'w', encoding='utf-8') as writer:
        writer.write('1' + '\t' + '1' + '\t' + '1' + '\n')

    print('saved successfully!')


def build_labels(to_path):
    labels = ['0', '1']
    with open(to_path, 'w', encoding='utf-8') as writer:
        for label in labels:
            writer.write(label + '\n')
    
    print('saved successfully!')


if __name__ == '__main__':
    origin_dir = './tcdata/'
    to_pretrain_wo_aug = './data/pretrain_wo_aug.txt'
    to_pretrain_aug = './data/pretrain_aug.txt'
    to_train = './data/train.txt'
    to_test = './data/test.txt'
    to_label = './data/label.txt'
    build_pretrain_wo_aug(origin_dir, to_pretrain_wo_aug)
    build_pretrain_aug(origin_dir, to_pretrain_aug)
    build_train(origin_dir, to_train)
    build_test(to_test)
    build_labels(to_label)