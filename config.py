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
        self.model_name = 'bert_last'
        self.train_path = dataset + '/match/train_new.txt'
        self.dev_path = dataset + '/match/test_new.txt'
        self.test_path = dataset + '/match/test_new.txt'
        self.class_path = dataset + '/match/label.txt'
        self.save_path = dataset + '/match/save_dict/' + self.model_name + '.pth'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.require_improment = 1000
        self.num_epochs = 10
        self.batch_size = 64 #32
        self.max_seq_length = 32 # 两个句子合在一起最长长度为127, 需要[CLS] [SEP] [SEP]
        self.learning_rate = 5e-5
        # self.bert_path = './chinese-bert-wwm'
        # self.bert_path = './output_nezha_1/checkpoint-54000'
        self.bert_path = './output_new_1/checkpoint-3908'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path) # 'pytorch_model.bin'这个名字是写死了，在modeling.py和file_utils.py文件中
        self.hidden_size = 768
        self.num_classes = 1