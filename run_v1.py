# -*- encoding: utf-8 -*-
'''
@File        :run_v1.py
@Time        :2020/12/18 11:16:03
@Author      :xiaoqifeng
@Version     :1.0
@Contact:    :unknown
'''

import torch
import numpy as np
from pytorch_pretrained_bert import BertTokenizer
from BERTForMultiClassification_v1 import BERTForMultiLabelSequenceClassification
from config_v1 import Config
from DataLoader_v1 import TextProcessor, convert_examples_to_features, convert_single_example
from sklearn.metrics import roc_curve, auc


class Predict:
    def __init__(self, config, model_path, label_path, bert_path='chinese-bert-wwm', max_seq_length=32):
        self.config = config
        self.model_path = model_path
        self.label_path = label_path
        self.bert_path = bert_path

        self.model = BERTForMultiLabelSequenceClassification(self.config, self.config.num_classes) 
        self.model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))
        self.model.half()
        self.model.eval()
        self.model.to(self.config.device)

        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.max_seq_length = max_seq_length
        
        self.processor = TextProcessor()
        self.labels = self.processor.get_labels(self.label_path)
        self.label2id = {label: id_ for id_, label in enumerate(self.labels)}
        self.id2label = {id_: label for id_, label in enumerate(self.labels)}

    def run(self, record):
        '''
        预测小类标签
        '''
        text_a, text_b = record[0], record[1]
        example = self.processor._create_single_example(text_a, text_b)
        feature = convert_single_example(example, self.max_seq_length, self.tokenizer)

        input_ids = torch.tensor(feature.input_ids, dtype=torch.long).unsqueeze(0).to(self.config.device)
        segment_ids = torch.tensor(feature.segment_ids, dtype=torch.long).unsqueeze(0).to(self.config.device)
        input_mask = torch.tensor(feature.input_mask, dtype=torch.long).unsqueeze(0).to(self.config.device)
        # print(input_ids)
        # print(segment_ids)
        # print(input_mask)

        logits = self.model(input_ids, segment_ids, input_mask).detach()
        # print(logits)
        prob = logits.sigmoid()[:, 1].tolist() #[0.123]
        # prob = torch.sigmoid(logits)

        # return prob[0].cpu().tolist()[0]
        return prob[0]

    def collect_badcase(self, data_path):
        badcase = []
        cnt = 0
        with open(data_path, 'r', encoding='utf-8') as reader:
            for record in reader:
                print(f'第{cnt+1}条记录...')
                cnt += 1
                text_a, text_b, label = record.strip().split('\t')
                pre = self.run([text_a, text_b])
                if pre > 0.5:
                    pre_label = '1'
                else:
                    pre_label = '0'
                if pre_label != label:
                    badcase.append('\t'.join([text_a, text_b, label, pre_label, str(pre)]))

        return badcase

    def evaluate(self, data_path):
        '''在全部的数据集上对模型进行测试
        '''
        labels = []
        pres = []

        cnt = 0
        with open(data_path, 'r', encoding='utf-8') as reader:
            for record in reader:
                print(f'第{cnt+1}条记录...')
                cnt += 1
                text_a, text_b, label = record.strip().split('\t')
                pre = self.run([text_a, text_b])
                labels.append(int(label))
                pres.append(pre)

        fpr, tpr, th = roc_curve(labels, pres, pos_label=1)
        auc_score = auc(fpr, tpr)
        return auc_score, pres

    def inference(self, data_path, to_path):
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
    import time
    start = time.time()
    config = Config('data')
    print(f'正在使用{config.device}进行推理...')
    model_path = 'data/match/save_dict/bert_last.pth'
    label_path = 'data/match/label.txt'
    bert_path = 'output_bert_last'
    data_path = 'data/match/data_2_28/testB.tsv'
    to_path = 'data/match/data_2_28/' + 'torch_last.tsv'
    
    pre = Predict(config, model_path, label_path, bert_path)
    pre.inference(data_path, to_path)
    print(f'耗时:{(time.time()-start)/60}分钟...')

    # record = ['9890 552 553 9715 9716 9717 3296', '324 302 552 571 3526']
    # a = pre.run(record)
    # print(a)
    # for i in range(2, 6):
    #     model_path = 'data/match/save_dict/bert_kf_' + str(i) + '.pth'
    #     pre = Predict(config, model_path, label_path, bert_path)
    #     to_path = 'data/match/data_2_28/kf' + str(i) + '_res.tsv'
    #     pre.inference(data_path, to_path)

    # print(f'耗时:{(time.time()-start)/60}分钟...')

    # import time
    # # 5折预测
    # start = time.time()
    # config = Config('data')
    # label_path = 'data/match/label.txt'
    # bert_path = 'output_bert_v1'
    # cnt = 0
    # res = []
    # with open('data/match/data_2_28/testA.tsv', 'r', encoding='utf-8') as reader:
    #     for record in reader:
    #         print(f'第{cnt+1}条记录...')
    #         cnt += 1
    #         text_a, text_b = record.strip().split('\t')
    #         temp = []
    #         for i in range(1, 6):
    #             model_path = 'data/match/save_dict/bert_kf_' + str(i) + '.pth'
    #             pre = Predict(config, model_path, label_path, bert_path)
    #             temp.append(pre.run([text_a, text_b]))
    #         res.append(sum(temp)/5)
    # with open('data/match/data_2_28/res.tsv', 'w', encoding='utf-8') as writer:
    #     for pre in res:
    #         writer.write(str(pre) + '\n')

    # print(f'耗时:{(time.time()-start)/60}分钟...')

    # text_a, text_b = '更换手机怎么把手机中的资料传过去', '如何把旧手机资料传到新手机'
    # prob = pre.run([text_a, text_b])
    # print(prob)
    # data_path = 'data/test.txt'
    # auc_score, pres = pre.evaluate(data_path)
    # print(f'auc为:{auc_score}') # auc为:0.9615384115297171 数据划分 8:2。
    # print(f'两万条数据，用时{time.time()-start}s...') # 两万条数据，用时314.0333664417267s
    # data_path = 'data/data_2_28/gaiic_track3_round1_testA_20210228.tsv'
    # pres = pre.get_res(data_path)

    # import csv
    # to_path = 'data/result.tsv'
    # with open(to_path, 'w') as f:
    #     tsv_w = csv.writer(f, delimiter='\t')
    #     for pre in pres:
    #         tsv_w.writerow([pre])
    # print('saved successfully!')
    # import random
    # import csv
    # to_path = 'data/result.tsv'
    # with open(to_path, 'w') as f:
    #     tsv_w = csv.writer(f, delimiter='\t')
    #     for i in range(25000):
    #         tsv_w.writerow([random.random()])

    # to_path = 'data/res.txt'
    # with open(to_path, 'w', encoding='utf-8') as writer:
    #     for score in pres:
    #         writer.write(str(score) + '\n')
    # print('save successfully!')

    # badcase = pre.collect_badcase(data_path)
    # with open('data/badcase.txt', 'w', encoding='utf-8') as writer:
    #     for record in badcase:
    #         writer.write(record + '\n')
    # print('save successfully!')