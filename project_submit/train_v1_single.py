# -*- encoding: utf-8 -*-
'''
@File        :train_v1.py
@Time        :2020/12/14 08:44:17
@Author      :xiaoqifeng
@Version     :1.0
@Contact:    :unknown
'''

import torch
import numpy as np
from sklearn import metrics
from pytorch_pretrained_bert import BertTokenizer, BertAdam
from BERTForMultiClassification_v1 import BERTForMultiLabelSequenceClassification
from DataLoader_v1 import convert_examples_to_features, TextProcessor
from config_v1 import Config
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
# from tqdm import tqdm
import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def acc(y_pred, y_true):
    y_pred = y_pred.cpu()
    y_true = y_true.cpu()
    return np.mean(y_pred.numpy() == y_true.numpy())


def pre(y_pred, y_true):
    y_pred = y_pred.cpu()
    y_true = y_true.cpu()

    y_pred = y_pred.tolist()
    y_true = y_true.tolist()

    score = metrics.precision_score(y_true, y_pred)
    return score


def evaluate(config, model, dev_dataloader):
    model.eval()

    total_loss, total_accuracy, total_precision = 0, 0, 0
    cnt = 0
    true_labels = []
    pre_labels = []
    for batch in dev_dataloader:
        cnt += 1
        batch = tuple(t.to(config.device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch
        with torch.no_grad():
            loss = model(input_ids, segment_ids, input_mask, label_ids)
            logits = model(input_ids, segment_ids, input_mask)
            total_loss += loss
            batch_pred_val, batch_pred_idx = torch.max(logits.sigmoid(), -1)
            batch_true_labels = label_ids.squeeze(-1).tolist()
            label_1_scores = logits.sigmoid()[:, 1].tolist()

            total_accuracy += acc(batch_pred_idx, label_ids.squeeze(-1))
            total_precision += pre(batch_pred_idx, label_ids.squeeze(-1))
            batch_pred_labels = batch_pred_val.cpu().tolist()
            true_labels.extend(batch_true_labels)
            pre_labels.extend(label_1_scores)

    fpr, tpr, thresholds = metrics.roc_curve(true_labels, pre_labels, pos_label=1) #注意这里的pos_label参数
    auc_score = metrics.auc(fpr, tpr)

    return total_accuracy / cnt, total_precision / cnt, total_loss / cnt, auc_score


class FGM():
    '''对抗训练脚本
    '''
    def __init__(self, model):
        self.model = model
        self.backup = {}
    
    def attack(self, epsilon=0.1, emb_name='bert.embeddings.word_embeddings.weight'): #epsilon=0.1,0.3,0.5  用一下:0.1
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='bert.embeddings.word_embeddings.weight'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


def train(config, model):
    fgm = FGM(model)
    processor = TextProcessor()
    label_list = processor.get_labels(config.class_path)

    #加载训练数据
    train_examples = processor.get_train_examples(config.train_path)
    train_features = convert_examples_to_features(train_examples, label_list, config.max_seq_length, config.tokenizer)

    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in train_features], dtype=torch.long)

    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=config.batch_size, drop_last=True)

    #加载测试数据
    dev_examples = processor.get_dev_examples(config.dev_path)
    dev_features = convert_examples_to_features(dev_examples, label_list, config.max_seq_length, config.tokenizer)

    all_input_ids_dev = torch.tensor([f.input_ids for f in dev_features], dtype=torch.long)
    all_input_mask_dev = torch.tensor([f.input_mask for f in dev_features], dtype=torch.long)
    all_segment_ids_dev = torch.tensor([f.segment_ids for f in dev_features], dtype=torch.long)
    all_label_ids_dev = torch.tensor([f.label_ids for f in dev_features], dtype=torch.long)

    dev_data = TensorDataset(all_input_ids_dev, all_input_mask_dev, all_segment_ids_dev, all_label_ids_dev)
    dev_sampler = SequentialSampler(dev_data)
    dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=config.batch_size)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=config.learning_rate,
                        #  schedule='warmup_linear',
                         warmup=0.05,
                         t_total=config.num_epochs * len(train_dataloader))
    
    #lookahead
    # from optimizer import Lookahead
    # optimizer = Lookahead(optimizer, k=5, alpha=0.5)

    logger.info(f"正在使用GPU: {torch.cuda.current_device()}进行训练...")

    model.train()

    eval_steps = len(train_dataloader) // 2
    for i in range(config.num_epochs):
        total_batch = 0
        eval_best_loss = float('inf')
        eval_best_auc_score = float('-inf')
        eval_best_acc = float('-inf')
        last_improve = 0
        flag = False

        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(config.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch

            loss = model(input_ids, segment_ids, input_mask, label_ids)
            loss.backward()

            #对抗训练
            fgm.attack()
            loss_adv = model(input_ids, segment_ids, input_mask, label_ids)
            loss_adv.backward()
            fgm.restore()

            optimizer.step()
            model.zero_grad()

            logits = model(input_ids, segment_ids, input_mask)

            logger.info(f"Epoch: {i+1}, step: {step+1}, train_loss: {loss}")
        torch.save(model.state_dict(), config.save_path)


if __name__ == '__main__':
    config = Config('.')
    np.random.seed(2021)
    torch.manual_seed(2021)
    torch.cuda.manual_seed_all(2021)
    torch.backends.cudnn.deterministic = True
    model = BERTForMultiLabelSequenceClassification(config).to(config.device)

    # 从checkpoint处开始训练
    # model_path = 'data/save_dict/bert_new_data_part_1.pth'
    # model.load_state_dict(torch.load(model_path))
    # model.to(config_ml.device)
    
    print(model)
    train(config, model)