# -*- encoding: utf-8 -*-
'''
@File        :run_pretraining.py
@Time        :2021/03/09 09:35:04
@Author      :xiaoqifeng
@Version     :1.0
@Contact:    :unknown
'''

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
import tokenizers
import torch
from transformers import BertTokenizer, LineByLineTextDataset, BertConfig, BertForMaskedLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments, DataCollatorForNgramMask

'''
某些python包需要频繁修改的话，直接拉下来放在项目里就好了。
'''
# def train_tokenizer():
#     filepath_vocab = 'data/match/pretrain_vocab.txt'
#     bwpt = tokenizers.BertWordPieceTokenizer(vocab_file=None)
#     bwpt.train(
#         files=[filepath_vocab],
#         min_frequency=2,
#         limit_alphabet=1000,
#     )
#     bwpt.save('./output_bert_last')


filepath = 'data/match/pretrain_wo_aug.txt'
# filepath = 'data/match/pretrain.txt' #150个epoch的没有对偶数据增强
vocab_file_dir = './output_bert/vocab.txt'
tokenizer = BertTokenizer.from_pretrained(vocab_file_dir)


dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path = filepath,
    block_size=32 # 32
)

# config = BertConfig(
#     vocab_size=23737,
#     hidden_size=768,
#     num_hidden_layers=12,
#     num_attention_heads=12,
#     max_position_embeddings=512,
# )

# model = BertForMaskedLM(config)
model = BertForMaskedLM.from_pretrained(pretrained_model_name_or_path='RoBERTa-base')
model.resize_token_embeddings(new_num_tokens=23737) #这里写现在token的个数

# data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15) #0.15
data_collator = DataCollatorForNgramMask(tokenizer=tokenizer, mlm=True, mlm_probability=0.15) #0.15

training_args = TrainingArguments(
    output_dir='output_roberta',
    overwrite_output_dir=True,
    num_train_epochs=100,
    per_device_train_batch_size=128, #256
    save_steps=int(450000/128*100/10), #250000
    logging_steps=int(450000/128*100/100),
    save_total_limit=30,
    learning_rate=5e-5,
    fp16=True,
    prediction_loss_only=True,
    label_smoothing_factor=0.001,
    # seed=2021,
    # weight_decay=0.01,
)

# 参数效果更好
# training_args = TrainingArguments(
#     output_dir='./output_nezha_large',
#     overwrite_output_dir=True,
#     num_train_epochs=100,
#     per_device_train_batch_size=32, #128
#     save_steps=int(450000/32*100/10), #250000
#     logging_steps=int(450000/32*100/100),
#     save_total_limit=30,
#     fp16=True,
#     prediction_loss_only=True,
#     label_smoothing_factor=0.001,
#     # weight_decay=0.01,
# )


trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)


if __name__ == '__main__':
    torch.cuda.manual_seed(2021)
    torch.cuda.manual_seed_all(2021)
    trainer.train()
    trainer.save_model('./pre_model_roberta')
    # train_tokenizer()