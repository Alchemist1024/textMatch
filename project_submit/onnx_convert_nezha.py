# -*- encoding: utf-8 -*-
'''
@File        :onnx_convert.py
@Time        :2021/04/22 08:25:29
@Author      :xiaoqifeng
@Version     :1.0
@Contact:    :unknown
'''

import onnx
import torch
# from BERTForMultiClassification_v1 import BERTForMultiLabelSequenceClassification
# from config_v1 import Config
from BERTForMultiClassification_nezha import BERTForMultiLabelSequenceClassification
from config_nezha import Config
from DataLoader_v1 import TextProcessor, convert_examples_to_features, convert_single_example
from pytorch_pretrained_bert import BertTokenizer
from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions


def convert(model_path, to_onnx_path):
    config = Config('.')
    opset_version=11
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_gpu else "cpu")
    print(device)
    # print(device)
    # device = torch.device('cpu')

    inputs = {
        'input_ids': torch.ones([1, 32], dtype=torch.long).to(device),
        'token_type_ids': torch.ones([1, 32], dtype=torch.long).to(device),
        'attention_mask': torch.ones([1, 32], dtype=torch.long).to(device)
    }
    # print(len(tuple(inputs.values())))

    model = BERTForMultiLabelSequenceClassification(config, config.num_classes) 
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    model.to(device)

    with torch.no_grad():
        symbolic_names = {0: 'batch_size', 1: 'max_seq_len'}
        logits_name = {0: 'batch_size', 1: 'num_class'}
        torch.onnx.export(model,
                   args=tuple(inputs.values()),
                   f=to_onnx_path,
                   verbose=True,
                   opset_version=opset_version,
                   do_constant_folding=True,
                   input_names=['input_ids', 'segment_ids', 'input_mask'],
                   output_names=['logits'],
                   dynamic_axes={'input_ids': symbolic_names,
                                 'segment_ids': symbolic_names,
                                 'input_mask': symbolic_names,
                                 'logits': logits_name})
    
    print('model exported at', to_onnx_path)


class Predict:
    def __init__(self, model_path, bert_path):
        self.processor = TextProcessor()
        # self.sess_options = SessionOptions()
        # self.sess_options.intra_op_num_threads = 1
        # self.sess_options.intra_op_num_threads=psutil.cpu_count(logical=True)
        # self.sess_options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
        # self.session = InferenceSession(model_path, self.sess_options)
        self.session = InferenceSession(model_path)
        # self.session_1 = InferenceSession(model_path_1) # 多个模型融合
        # print(self.session.get_inputs()[2].name)
        # print(len(self.session.get_inputs()))
        # self.use_gpu = torch.cuda.is_available()
        # self.device = torch.device("cuda:7" if self.use_gpu else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)

    # def to_numpy(self, tensor):
    #     return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    def to_numpy(self, tensor):
        return tensor.detach().cuda().numpy() if tensor.requires_grad else tensor.cuda().numpy()

    def run(self, record):
        text_a, text_b = record[0], record[1]
        example = self.processor._create_single_example(text_a, text_b)
        feature = convert_single_example(example, 32, self.tokenizer)

        input_ids = torch.tensor(feature.input_ids, dtype=torch.long).unsqueeze(0)
        segment_ids = torch.tensor(feature.segment_ids, dtype=torch.long).unsqueeze(0)
        input_mask = torch.tensor(feature.input_mask, dtype=torch.long).unsqueeze(0)
        
        # ort_inputs = {
        #     'input_ids': self.to_numpy(input_ids),
        #     'segment_ids': self.to_numpy(segment_ids),
        #     'input_mask': self.to_numpy(input_mask)
        # }
        # print(input_ids)
        # print(segment_ids)
        # print(input_mask)
        ort_inputs = {
            'input_ids': self.to_numpy(input_ids),
            'segment_ids': self.to_numpy(segment_ids),
            'input_mask': self.to_numpy(input_mask)
        }
        print(self.session)
        ort_outputs = self.session.run(None, ort_inputs)
        # print(ort_outputs)
        ort_logits = torch.from_numpy(ort_outputs[0])
        print(ort_logits) # tensor([[4.7433, -4.5335]])
        # ort_logits_1 = torch.from_numpy(ort_outputs_1[0]) # 二维向量
        prob = ort_logits.sigmoid()[:, 1].tolist()[0] #[0.123]
        # print(ort_logits)
        return prob

    def infer(self, data_path, to_path):
        pres = []
        cnt = 0 
        with open(data_path, 'r', encoding='utf-8') as reader:
            for record in reader:
                print(f'第{cnt+1}条记录...')
                cnt += 1
                text_a, text_b = record.strip().split('\t')
                pre = self.run([text_a, text_b])
                pres.append(pre)

        with open(to_path, 'w', encoding='utf-8') as writer:
            for pre in pres:
                writer.write(str(pre) + '\n')


if __name__ == '__main__':
    model_path = './model/bert_nezha_fgm_all.pth'
    to_onnx_path = './model/bert_nezha_fgm_all_gpu.onnx'
    convert(model_path, to_onnx_path)

    # model_path = 'data/match/save_dict/bert_nezha_test.pth'
    # to_onnx_path = 'data/match/save_dict/bert_nezha_test_gpu.onnx'
    # convert(model_path, to_onnx_path)

    # import time
    # start = time.time()
    # onnx_model_path = 'data/match/save_dict/bert_last_gpu.onnx'
    # # onnx_model = onnx.load(onnx_model_path)
    # # onnx.checker.check_model(onnx_model)
    # bert_path = 'output_bert_last'
    # pre = Predict(onnx_model_path, bert_path)
    # record = ['9890 552 553 9715 9716 9717 3296', '324 302 552 571 3526']
    # pre.run(record)

    # data_path = 'data/match/data_2_28/testB.tsv'
    # to_path = 'data/match/data_2_28/' + 'res_onnx.tsv'
    # pre.infer(data_path, to_path)
    # print(f'耗时:{(time.time()-start)/60}分钟...')