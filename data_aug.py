# -*- encoding: utf-8 -*-
'''
@File        :data_aug.py
@Time        :2021/03/30 16:56:22
@Author      :xiaoqifeng
@Version     :1.0
@Contact:    :unknown
reference: https://github.com/huanghuidmml/epidemicTextMatch/blob/master/code/data_aug.py
'''

import pandas as pd
import csv
import itertools
from sklearn.model_selection import KFold
import os


def load_data(data_path):
    datas = []
    with open(data_path, 'r', encoding='utf-8') as reader:
        for record in reader:
            text_a, text_b, label = record.strip().split('\t')
            datas.append([text_a, text_b, label])
    return datas


def data_aug(datas):
    dic = {}
    for data in datas:
        if data[0] not in dic:
            dic[data[0]] = {'true': [], 'false': []}
            dic[data[0]]['true' if data[2] == '1' else 'false'].append(data[1])
        else:
            dic[data[0]]['true' if data[2] == '1' else 'false'].append(data[1])

    new_datas = []
    for sent1, sent2s in dic.items():
        trues = sent2s['true']
        falses = sent2s['false']
        for true in trues:
            new_datas.append([sent1, true, '1'])
        for false in falses:
            new_datas.append([sent1, false, '0'])
        temp_trues = []
        temp_falses = []
        if len(trues) != 0 and len(falses) != 0:
            ori_rate = len(trues) / len(falses)
            for i in itertools.combinations(trues, 2):
                temp_trues.append([i[0], i[1], '1'])
            for true in trues:
                for false in falses:
                    temp_falses.append([true, false, '0'])
            num_t = int(len(temp_falses) * ori_rate)
            num_f = int(len(temp_trues) / ori_rate)
            temp_rate = len(temp_trues) / len(temp_falses)
            if ori_rate < temp_rate:
                temp_trues = temp_trues[:num_t]
            else:
                temp_falses = temp_falses[:num_f]
        new_datas = new_datas + temp_trues + temp_falses
    return new_datas


def save(datas, to_path):
    cnt = 0
    with open(to_path, 'w', encoding='utf-8') as writer:
        for record in datas:
            print(f'第{cnt+1}条记录...')
            cnt += 1
            new_record = '\t'.join(record)
            writer.write(new_record + '\n')
    print('saved successfully!')


if __name__ == '__main__':
    data_path = 'data/match/train_new.txt'
    to_path = 'data/match/train_new_aug.txt'
    datas = load_data(data_path)
    new_datas = data_aug(datas)
    save(new_datas, to_path)