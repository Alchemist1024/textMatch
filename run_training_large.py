# -*- encoding: utf-8 -*-
'''
@File        :run_pretraining.py
@Time        :2021/03/09 09:35:04
@Author      :xiaoqifeng
@Version     :1.0
@Contact:    :unknown
'''

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2' #指定了这个以后id就是从0开始了
import tokenizers
from transformers import BertTokenizer, LineByLineTextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments, DataCollatorForNgramMask
from configuration_nezha import NeZhaConfig
from modeling_nezha import NeZhaForMaskedLM
import torch
import torch.nn as nn


# 在nezha base的基础上继续训练
# transformers的版本4.4.2

# def train_tokenizer():
#     filepath_vocab = 'data/match/pretrain_vocab.txt'
#     bwpt = tokenizers.BertWordPieceTokenizer(vocab_file=None)
#     bwpt.train(
#         files=[filepath_vocab],
#         min_frequency=2,
#         limit_alphabet=1000
#     )
#     bwpt.save('./pre_model_nezha')


# 暂时不跑了，太慢了
filepath = './data/match/pretrain_wo_aug.txt'
vocab_file_dir = './output_nezha_large/vocab.txt'
tokenizer = BertTokenizer.from_pretrained(vocab_file_dir)


dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path = filepath,
    block_size=32 # 32
)

config = NeZhaConfig(
    attention_probs_dropout_prob=0.1,  
    hidden_act='gelu',
    hidden_dropout_prob=0.1,
    hidden_size=1024,
    initializer_range=0.02,
    intermediate_size=4096,
    max_position_embeddings=512,
    # max_relative_position=64,
    num_attention_heads=16,
    num_hidden_layers=24,
    type_vocab_size=2,
    vocab_size=21128,
    use_relative_position=True
)

# model = NeZhaForMaskedLM(config)
model = NeZhaForMaskedLM.from_pretrained(pretrained_model_name_or_path='nezha-large', config=config)
model.resize_token_embeddings(new_num_tokens=23737) #这里写现在token的个数

# data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
data_collator = DataCollatorForNgramMask(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

training_args = TrainingArguments(
    output_dir='./output_nezha_large',
    overwrite_output_dir=True,
    num_train_epochs=100,
    per_device_train_batch_size=32, #128
    save_steps=int(450000/32*100/10), #250000
    logging_steps=int(450000/32*100/100),
    save_total_limit=30,
    fp16=True,
    prediction_loss_only=True,
    label_smoothing_factor=0.001,
    # weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset
)


if __name__ == '__main__':
    torch.cuda.manual_seed(2021)
    torch.cuda.manual_seed_all(2021)
    # train_tokenizer()
    trainer.train()
    trainer.save_model('./pre_model_nezha_large')