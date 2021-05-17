# -*- encoding: utf-8 -*-
'''
@File        :get_kf_res.py
@Time        :2021/04/07 15:24:33
@Author      :xiaoqifeng
@Version     :1.0
@Contact:    :unknown
'''

import pandas as pd


kf1 = pd.read_csv('data/match/data_2_28/kf1_res.tsv', sep='\t', header=None)
kf2 = pd.read_csv('data/match/data_2_28/kf2_res.tsv', sep='\t', header=None)
kf3 = pd.read_csv('data/match/data_2_28/kf3_res.tsv', sep='\t', header=None)
kf4 = pd.read_csv('data/match/data_2_28/kf4_res.tsv', sep='\t', header=None)
kf5 = pd.read_csv('data/match/data_2_28/kf5_res.tsv', sep='\t', header=None)

res = []
for i in range(len(kf1.values)):
    avg_score = (kf1.values[i][0] + kf2.values[i][0] + kf3.values[i][0] + kf4.values[i][0] + kf5.values[i][0]) / 5
    res.append(avg_score)

with open('data/match/data_2_28/new_res.tsv', 'w', encoding='utf-8') as writer:
    for record in res:
        writer.write(str(record) + '\n')

print('saved successfully!')