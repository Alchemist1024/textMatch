import os
import tokenizers
import json

def train_tokenizer():
    filepath_vocab = 'data/match/pretrain_vocab.txt'
    bwpt = tokenizers.BertWordPieceTokenizer(vocab=None)
    bwpt.train(
        files=[filepath_vocab],
        min_frequency=2,
        limit_alphabet=1000,
    )
    bwpt.save('./fuck/vocab.json')

def save_vocab(json_path, vocab_path):
    with open(json_path, 'r') as load_f:
        load_dict = json.load(load_f)
    
    with open(vocab_path, 'w', encoding='utf-8') as writer:
        for vocab in load_dict['model']['vocab']:
            writer.write(vocab + '\n')
    print('save successfully!')

if __name__ == '__main__':
    train_tokenizer()
    json_path = './fuck/vocab.json'
    vocab_path = './fuck/vocab.txt'
    save_vocab(json_path, vocab_path)