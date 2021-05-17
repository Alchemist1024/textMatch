# -*- encoding: utf-8 -*-
'''
@File        :BERTForMultiClassification.py
@Time        :2020/12/10 09:36:29
@Author      :xiaoqifeng
@Version     :1.0
@Contact:    :unknown
'''

import torch
torch.cuda.set_device(4)
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCELoss, BCEWithLogitsLoss, CrossEntropyLoss
from torch.autograd import Variable
from pytorch_pretrained_bert.modeling import BertModel, BertPooler


class BERTForMultiLabelSequenceClassification(nn.Module):
    '''2分类最后预测的结果为一个浮点数
    '''
    def __init__(self, configML, num_classes=1):
        super(BERTForMultiLabelSequenceClassification, self).__init__()
        self.num_classes = num_classes
        self.bert = BertModel.from_pretrained(configML.bert_path)
        self.pooler = BertPooler(configML)
        for param in self.bert.parameters():
            param.required_grad = True
        self.fc = nn.Linear(configML.hidden_size, self.num_classes)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        encoded_layers, pooled_output1 = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=True)
        # last_5_layers_pool = []
        # for layer in encoded_layers[-5:]:
        #     last_5_layers_pool.append(self.pooler(layer))
        # last_5_layers_mean = torch.mean(torch.stack(last_5_layers_pool), 0)

        logits = self.fc(pooled_output1)
        # logits = self.fc(last_5_layers_mean)

        if labels is not None:
            loss_fn = BCEWithLogitsLoss()
            # loss_fn = CrossEntropyLoss()
            labels = labels.float()
            loss = loss_fn(logits.view(-1, self.num_classes), labels.view(-1, self.num_classes))
            # loss = loss_fn(logits.view(-1, self.num_classes), labels.view(-1))
            return loss  
        else:            
            return logits


if __name__ == '__main__':
    # a = torch.FloatTensor([[0, 0.2232, 0.6435]])
    # b = torch.FloatTensor([[0, 1, 1]])
    # c, d = torch.max(a, 0)
    # print(c)
    # print(d)
    from config import Config
    config = Config('data')
    bertClass = BERTForMultiLabelSequenceClassification(config)