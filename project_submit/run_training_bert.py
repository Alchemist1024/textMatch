# -*- encoding: utf-8 -*-
'''
@File        :run_pretraining.py
@Time        :2021/03/09 09:35:04
@Author      :xiaoqifeng
@Version     :1.0
@Contact:    :unknown
'''

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from transformers import BertTokenizer, LineByLineTextDataset, BertConfig, BertForMaskedLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments, DataCollatorForNgramMask


filepath = './data/pretrain_aug.txt' 
# filepath = './data/pretrain_wo_aug.txt'
vocab_file_dir = './data/vocab.txt'
tokenizer = BertTokenizer.from_pretrained(vocab_file_dir)


dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path = filepath,
    block_size=32 # 32
)

config = BertConfig(
    vocab_size=23737,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    max_position_embeddings=512,
)

model = BertForMaskedLM(config)

data_collator = DataCollatorForNgramMask(tokenizer=tokenizer, mlm=True, mlm_probability=0.15) #0.15

training_args = TrainingArguments(
    output_dir='./output_bert',
    overwrite_output_dir=True,
    num_train_epochs=100,
    per_device_train_batch_size=128, #256
    save_steps=int(900000/128*100), #就存最后一个
    logging_steps=int(900000/128*100/100),
    save_total_limit=30,
    learning_rate=5e-5,
    fp16=True,
    prediction_loss_only=True,
    label_smoothing_factor=0.001,
    # weight_decay=0.01,
)


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
    trainer.save_model('./pre_bert')