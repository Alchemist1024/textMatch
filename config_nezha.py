# -*- encoding: utf-8 -*-
'''
@File        :config.py
@Time        :2020/12/15 08:39:03
@Author      :xiaoqifeng
@Version     :1.0
@Contact:    :unknown
'''

import torch
from pytorch_pretrained_bert import BertTokenizer


class Config:
    '''配置参数
    '''
    def __init__(self, dataset):
        self.model_name = 'bert_nezha_fgm_0.3'
        # self.train_path = dataset + '/match/train_all.txt'
        self.train_path = dataset + '/match/train_last.txt'
        self.dev_path = dataset + '/match/test_last.txt'
        self.test_path = dataset + '/match/test_last.txt'
        self.class_path = dataset + '/match/label.txt'
        self.save_path = dataset + '/match/save_dict/' + self.model_name + '.pth'
        self.device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')

        self.require_improment = 1000
        self.num_epochs = 3 # 全量跑3个epoch
        self.batch_size = 64 #32
        self.max_seq_length = 32 # 两个句子合在一起最长长度为127, 需要[CLS] [SEP] [SEP]
        self.learning_rate = 5e-5
        self.bert_path = './output_nezha_base/checkpoint-351560'
        # self.bert_path = './output_nezha_base_v1/checkpoint-527340'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path) # 'pytorch_model.bin'这个名字是写死了，在modeling.py和file_utils.py文件中
        self.hidden_size = 768
        self.num_classes = 2