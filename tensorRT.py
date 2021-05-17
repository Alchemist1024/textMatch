# -*- encoding: utf-8 -*-
'''
@File        :tensorRT.py
@Time        :2021/04/21 16:43:54
@Author      :xiaoqifeng
@Version     :1.0
@Contact:    :unknown
'''

import torch
from BERTForMultiClassification_v1 import BERTForMultiLabelSequenceClassification
from config_v1 import Config
from torch2trt import torch2trt, TRTModule



# 使用torch2trt直接把torch模型转tensorrt很多算子不支持，暂且不做，有空再研究...
def convert(model_path):
    config = Config('data')

    input_ids = torch.ones((1, 32), dtype=torch.int32).cuda()
    input_mask = torch.ones((1, 32), dtype=torch.int32).cuda()
    segment_ids = torch.ones((1, 32), dtype=torch.int32).cuda()

    model = BERTForMultiLabelSequenceClassification(config, config.num_classes) 
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval().cuda()

    model_trt = torch2trt(model, [input_ids, input_mask, segment_ids])

    torch.save(model_trt.state_dict(), 'bert_trt.pth')
    print('转化成功...')


def load(model_path):
    model_trt = TRTModule()
    model_trt.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    return model_trt


if __name__ == '__main__':
    model_path = 'data/match/save_dict/bert_cls_3.3.pth'
    convert(model_path)
    # segment_ids = torch.ones((1, 32), dtype=torch.long).cuda()
    # print(segment_ids.dtype)
