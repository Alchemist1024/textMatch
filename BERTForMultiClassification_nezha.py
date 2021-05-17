# -*- encoding: utf-8 -*-
'''
@File        :BERTForMultiClassification.py
@Time        :2020/12/10 09:36:29
@Author      :xiaoqifeng
@Version     :1.0
@Contact:    :unknown
'''

import torch
# torch.cuda.set_device(7)
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCELoss, BCEWithLogitsLoss
from torch.autograd import Variable
# from modeling_nezha import NeZhaModel
from modeling_nezha import NeZhaModel


class BERTForMultiLabelSequenceClassification(nn.Module):
    '''2分类最后预测的结果为一个浮点数
       单模单折我理解指全数据训练固定Epoch
    '''
    def __init__(self, configML, num_classes=2):
        super(BERTForMultiLabelSequenceClassification, self).__init__()
        self.num_classes = num_classes
        self.bert = NeZhaModel.from_pretrained(configML.bert_path)
        for param in self.bert.parameters():
            param.required_grad = True
        self.fc = nn.Linear(configML.hidden_size, self.num_classes)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        # attention_mask = torch.ne(input_ids, 0)
        # attention_mask = attention_mask.long()
        # outputs = self.bert(input_ids, token_type_ids, attention_mask) #坑!!! 看源码，罪魁祸首在这，token_type_ids和attention_mask写反了，fuck!!!。
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        # last_5_layers_pool = []
        # for layer in encoded_layers[-5:]:
        #     last_5_layers_pool.append(self.pooler(layer))
        # last_5_layers_mean = torch.mean(torch.stack(last_5_layers_pool), 0)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)

        logits = self.fc(pooled_output)
        # logits = self.fc(last_5_layers_mean)

        if labels is not None:
            # loss_fn = BCEWithLogitsLoss()
            # labels = labels.float()
            # loss = loss_fn(logits.view(-1, self.num_classes), labels.view(-1, self.num_classes))
            loss = nn.CrossEntropyLoss()(nn.LogSoftmax(dim=-1)(logits), labels.view(-1))
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