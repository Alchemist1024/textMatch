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
        # self.model_name = 'bert_wwm_fgm_all'
        # self.model_name = 'roberta_fgm_all'
        # self.model_name = 'bert_wwm_fgm_part'
        self.model_name = 'roberta_fgm_part'
        # self.train_path = dataset + '/match/train_all.txt'
        self.train_path = dataset + '/match/train_last.txt'
        self.dev_path = dataset + '/match/test_last.txt'
        self.test_path = dataset + '/match/test_last.txt'
        self.class_path = dataset + '/match/label.txt'
        self.save_path = dataset + '/match/save_dict/' + self.model_name + '.pth'
        self.device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

        self.require_improment = 1000
        self.num_epochs = 3 # 10
        self.batch_size = 64 #32 64
        self.max_seq_length = 32 
        self.learning_rate = 5e-5 # 5e-5
        # self.bert_path = './output_bert/checkpoint-156160' #对应2.4 效果最好
        # self.bert_path = './output_bert/checkpoint-703120' #全量bert
        # self.bert_path = './output_bert_v1/checkpoint-527340' #全量bert
        # self.bert_path = './output_bert_wwm/checkpoint-351560' #全量bert
        self.bert_path = './output_roberta/checkpoint-351560' #全量robert
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path) # 'pytorch_model.bin'这个名字是写死了，在modeling.py和file_utils.py文件中
        self.hidden_size = 768
        self.num_classes = 2