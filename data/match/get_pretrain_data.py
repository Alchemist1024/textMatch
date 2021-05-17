# -*- encoding: utf-8 -*-
'''
@File        :get_pretrain_data.py
@Time        :2021/03/09 08:57:22
@Author      :xiaoqifeng
@Version     :1.0
@Contact:    :unknown
'''

def get_data(origin_dir, to_all_path):
    all_data = []
    datasets = ['train_1.tsv', 'train_2.tsv', 'testA.tsv', 'testB.tsv']

    for dataset in datasets:
        with open(origin_dir + dataset, 'r', encoding='utf-8') as reader:
            for record in reader:
                record_list = record.strip().split('\t')
                text_a, text_b = record_list[0], record_list[1]
                #预训练阶段就是让模型学习到句子中每一个单词的意思，这样我就可以把一份数据分成多个。
                #TODO:不用对偶试一下
                all_data.append(text_a + ' ' + text_b)
                # all_data.append(text_b + ' ' + text_a)

    # with open(origin_train_path, 'r', encoding='utf-8') as fr1, open(origin_test_path, 'r', encoding='utf-8') as fr2:
    #     for record in fr1:
    #         record_list = record.strip().split('\t')
    #         text_a, text_b = record_list[0], record_list[1]
    #         #预训练阶段就是让模型学习到句子中每一个单词的意思，这样我就可以把一份数据分成多个。
    #         all_data.append(text_a + ' ' + text_b)
    #         all_data.append(text_b + ' ' + text_a)

    #     for record in fr2:
    #         record_list = record.strip().split('\t')
    #         text_a, text_b = record_list[0], record_list[1]
    #         all_data.append(text_a + ' ' + text_b)
    #         all_data.append(text_b + ' ' + text_a)

    with open(to_all_path, 'w', encoding='utf-8') as writer:
        for record in all_data:
            writer.write(record + '\n')

    print('saved successfully!')


if __name__ == '__main__':
    # origin_train_path = './data_2_28/train.tsv'
    # origin_test_path = './data_2_28/testA.tsv'
    origin_dir = './data_2_28/'
    to_all_path = './pretrain_wo_aug.txt'
    get_data(origin_dir, to_all_path)
