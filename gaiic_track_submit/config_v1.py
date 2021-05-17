# -*- encoding: utf-8 -*-
'''
@File        :config_v1.py
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
        self.train_path = dataset + '/train_new.txt'
        self.dev_path = dataset + '/test_new.txt'
        self.test_path = dataset + '/test_new.txt'
        self.class_path = dataset + '/label.txt'
        self.save_path = dataset + '/match/save_dict/' + self.model_name + '.pth'
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.require_improment = 1000
        self.num_epochs = 10
        self.batch_size = 64 #32 64
        self.max_seq_length = 32 
        self.learning_rate = 5e-5 # 5e-5
        self.bert_path = './checkpoint-468480' #对应3.0了
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path) # 'pytorch_model.bin'这个名字是写死了，在modeling.py和file_utils.py文件中
        self.hidden_size = 768
        self.num_classes = 2