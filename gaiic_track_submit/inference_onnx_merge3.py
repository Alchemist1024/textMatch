# -*- encoding: utf-8 -*-
'''
@File        :inference.py
@Time        :2021/04/17 10:26:12
@Author      :xiaoqifeng
@Version     :1.0
@Contact:    :unknown
'''

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import json
from ai_hub import inferServer
import torch
from pytorch_pretrained_bert import BertTokenizer
from config_v1 import Config
from DataLoader_v1 import TextProcessor, convert_examples_to_features, convert_single_example, convert_single_example_dynamic
from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions


class AIHubInfer(inferServer):
    def __init__(self, model1, model2, model3):
        self.model2 = model2
        self.model3 = model3
        super().__init__(model1)

    #数据前处理
    def pre_process(self, req_data):
        input_batch = {}
        input_batch["input"] = req_data.form.getlist("input")
        input_batch["index"] = req_data.form.getlist("index")

        return input_batch
   
    # #数据后处理，如无，可空缺
    def post_process(self, predict_data):
        response = json.dumps(predict_data)
        return response

    #如需自定义，可覆盖重写
    def predict(self, preprocessed_data):
        input_list = preprocessed_data["input"]
        index_list = preprocessed_data["index"]
        
        response_batch = {}
        response_batch["results"] = []
        for i in range(len(index_list)):
            index_str = index_list[i]
        
            response = {}
            try:
                input_sample = input_list[i].strip()
                elems = input_sample.strip().split("\t")
                query_A = elems[0].strip()
                query_B = elems[1].strip()
                # predict = infer(model, query_A, query_B)
                predict = infer(self.model, self.model2, self.model3, query_A, query_B)
                response["predict"] = predict
                response["index"] = index_str
                response["ok"] = True
            except Exception as e:
                response["predict"] = 0
                response["index"] = index_str
                response["ok"] = False
            response_batch["results"].append(response)
        
        return response_batch


max_seq_len = 32
config = Config('data')
tokenizer = BertTokenizer.from_pretrained('./vocab')
processor = TextProcessor()


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


# 需要根据模型类型重写
def infer(bert, bert1, nezha, query_A, query_B):

    text_a, text_b = query_A, query_B
    example = processor._create_single_example(text_a, text_b)
    feature = convert_single_example_dynamic(example, max_seq_len, tokenizer)
    # feature = convert_single_example(example, max_seq_len, tokenizer)

    input_ids = torch.tensor(feature.input_ids, dtype=torch.long).unsqueeze(0)
    segment_ids = torch.tensor(feature.segment_ids, dtype=torch.long).unsqueeze(0)
    input_mask = torch.tensor(feature.input_mask, dtype=torch.long).unsqueeze(0)

    ort_inputs = {
        'input_ids': to_numpy(input_ids),
        'segment_ids': to_numpy(segment_ids),
        'input_mask': to_numpy(input_mask)
    }
    ort_outputs_bert = bert.run(None, ort_inputs)
    ort_outputs_bert1 = bert1.run(None, ort_inputs)
    ort_outputs_nezha = nezha.run(None, ort_inputs)
    ort_logits_bert = torch.from_numpy(ort_outputs_bert[0])
    ort_logits_bert1 = torch.from_numpy(ort_outputs_bert1[0])
    ort_logits_nezha = torch.from_numpy(ort_outputs_nezha[0])

    res = torch.mean(torch.stack([ort_logits_bert, ort_logits_bert1, ort_logits_nezha]), 0)
    prob = res.sigmoid()[:, 1].tolist()[0]

    return prob


# 需要根据模型类型重写
def init_model(model_path_bert, model_path_bert1, model_path_nezha):
    session_bert = InferenceSession(model_path_bert)
    session_bert1 = InferenceSession(model_path_bert1)
    session_nezha1 = InferenceSession(model_path_nezha)

    return session_bert, session_bert1, session_nezha1

if __name__ == "__main__":
    # import time
    # start = time.time()
    # model_path_bert = './model/bert_base_fgm_all_gpu.onnx'
    # model_path_bert1 = './model/bert_base_150_fgm_all_gpu.onnx'
    # model_path_nezha = './model/bert_nezha_all_fgm_gpu.onnx'
    model_path_bert = './model/bert_wwm_fgm_all_gpu.onnx'
    model_path_bert1 = './model/roberta_fgm_all_gpu.onnx'
    model_path_nezha = './model/bert_nezha_all_fgm_gpu.onnx'
    bert, bert1, nezha = init_model(model_path_bert, model_path_bert1, model_path_nezha)
    # record = [['12 2954 16', '12 32 126 5951 456 16'] for _ in range(50000)]
    # cnt = 0
    # for text_a, text_b in record:
    #     res = infer(bert, nezha, text_a, text_b)
    #     print(f'第{cnt+1}条记录...')
    #     cnt += 1
    #     print(res)
    # print(f'耗时:{(time.time()-start)/60}分钟...')
    aihub_infer = AIHubInfer(bert, bert1, nezha)
    aihub_infer.run(debuge=False)

    # docker build -t registry.cn-shanghai.aliyuncs.com/xiaoqifeng/gaiic_track3:3.0 .
    # 三个模型暂时跑不完，找一下原因。