# -*- encoding: utf-8 -*-
'''
@File        :inference.py
@Time        :2021/04/17 10:26:12
@Author      :xiaoqifeng
@Version     :1.0
@Contact:    :unknown
'''

import json
from ai_hub import inferServer
import torch
from pytorch_pretrained_bert import BertTokenizer
from BERTForMultiClassification_v1 import BERTForMultiLabelSequenceClassification
from config_v1 import Config
from DataLoader_v1 import TextProcessor, convert_examples_to_features, convert_single_example


class AIHubInfer(inferServer):
    def __init__(self, model):
        super().__init__(model)        

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
                predict = infer(model, query_A, query_B)
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
tokenizer = BertTokenizer.from_pretrained('./checkpoint-468480')
processor = TextProcessor()

# 需要根据模型类型重写
def infer(model, query_A, query_B):

    text_a, text_b = query_A, query_B
    example = processor._create_single_example(text_a, text_b)
    feature = convert_single_example(example, max_seq_len, tokenizer)

    input_ids = torch.tensor(feature.input_ids, dtype=torch.long).unsqueeze(0).to(config.device)
    input_mask = torch.tensor(feature.input_mask, dtype=torch.long).unsqueeze(0).to(config.device)
    segment_ids = torch.tensor(feature.segment_ids, dtype=torch.long).unsqueeze(0).to(config.device)

    logits = model(input_ids, segment_ids, input_mask).detach()
    prob = logits.sigmoid()[:, 1].tolist() #[0.123]

    return prob[0]


# 需要根据模型类型重写
def init_model(model_path):
    model = BERTForMultiLabelSequenceClassification(config)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    model.to(config.device)

    return model

if __name__ == "__main__":
    model_path = './model/bert_cls_3.3.pth'
    model = init_model(model_path)
    # text_a = '9890 552 553 9715 9716 9717 3296'	
    # text_b = '324 302 552 571 3526'	
    # res = infer(model, text_a, text_b)
    # print(res)
    aihub_infer = AIHubInfer(model)
    aihub_infer.run(debuge=False)