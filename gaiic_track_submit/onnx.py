# -*- encoding: utf-8 -*-
'''
@File        :onnx.py
@Time        :2021/04/22 08:25:29
@Author      :xiaoqifeng
@Version     :1.0
@Contact:    :unknown
'''

import onnx
import torch
from BERTForMultiClassification_v1 import BERTForMultiLabelSequenceClassification
from config_v1 import Config
from DataLoader_v1 import TextProcessor, convert_examples_to_features, convert_single_example
from pytorch_pretrained_bert import BertTokenizer
import psutil
from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions


def convert(model_path, to_onnx_path):
    config = Config('data')
    opset_version=11
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda:7" if use_gpu else "cpu")
    print(device)
    # device = torch.device('cpu')

    inputs = {
        'input_ids': torch.ones([1, 32], dtype=torch.long).to(device),
        'input_mask': torch.ones([1, 32], dtype=torch.long).to(device),
        'segment_ids': torch.ones([1, 32], dtype=torch.long).to(device)
    }

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
                   opset_version=opset_version,
                   do_constant_folding=True,
                   input_names=['input_ids', 'input_mask', 'segment_ids'],
                   output_names=['logits'],
                   dynamic_axes={'input_ids': symbolic_names,
                                 'input_mask': symbolic_names,
                                 'segment_ids': symbolic_names,
                                 'logits': logits_name})
    
    print('model exported at', to_onnx_path)


class Predict:
    def __init__(self, model_path, bert_path):
        self.processor = TextProcessor()
        self.sess_options = SessionOptions()
        # self.sess_options.intra_op_num_threads = 1
        self.sess_options.intra_op_num_threads=psutil.cpu_count(logical=True)
        # self.sess_options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = InferenceSession(model_path, self.sess_options)
        self.use_gpu = torch.cuda.is_available()
        self.device = torch.device("cuda:7" if use_gpu else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)

    def to_numpy(self, tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    def run(self, record):
        text_a, text_b = record[0], record[1]
        example = self.processor._create_single_example(text_a, text_b)
        feature = convert_single_example(example, self.max_seq_length, self.tokenizer)

        input_ids = torch.tensor(feature.input_ids, dtype=torch.long).unsqueeze(0)
        input_mask = torch.tensor(feature.input_mask, dtype=torch.long).unsqueeze(0)
        segment_ids = torch.tensor(feature.segment_ids, dtype=torch.long).unsqueeze(0)
        
        ort_inputs = {
            'input_ids': input_ids,
            'input_mask': input_mask,
            'segment_ids': segment_ids
        }
        ort_outputs = self.session.run(None, ort_inputs)
        print(ort_outputs)
        print(type(ort_outputs))

    def infer(self, data_path):
        pass


if __name__ == '__main__':
    # model_path = 'data/match/save_dict/bert_cls_3.3.pth'
    # to_onnx_path = 'data/match/save_dict/bert_cls_3.3.onnx'
    # convert(model_path, to_onnx_path)

    onnx_model_path = 'data/match/save_dict/bert_cls_3.3.onnx'
    bert_path = 'checkpint-468480'
    pre = Predict(onnx_model_path, bert_path)
    record = ['9890 552 553 9715 9716 9717 3296', '324 302 552 571 3526']
    pre.run(record)