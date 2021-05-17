# -*- encoding: utf-8 -*-
'''
@File        :DataLoader.py
@Time        :2020/12/10 16:42:21
@Author      :xiaoqifeng
@Version     :1.0
@Contact:    :unknown
'''

import os


class InputExample:
    def __init__(self, text_a, text_b=None, labels=None):
        self.text_a = text_a
        self.text_b = text_b
        self.labels = labels


class InputFeatures:
    def __init__(self, input_ids, input_mask, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids


class DataProcessor:
    def get_train_examples(self, data_path):
        raise NotImplementedError()

    def get_dev_examples(self, data_path):
        raise NotImplementedError

    def get_test_examples(self, data_path):
        raise NotImplementedError
    
    def get_labels(self, data_path):
        raise NotImplementedError


class TextProcessor(DataProcessor):
    '''自定义的多标签和多分类的数据处理类，主要是获得句子和相应标签，可以是多标签或者单标签。
    '''
    def _create_examples(self, data, labels_available=True):
        examples = []
        for record in data:
            labels = []
            try:
                text_a, text_b, label = record.split('\t')
            except Exception:
                continue
            if labels_available:
                for sub_lab in label.split(','):
                    labels.append(sub_lab)
            # examples.append(InputExample(text_a=text_a, text_b=text_b, labels=labels))
            examples.append(InputExample(text_a=text_a, text_b=text_b, labels=label))
            # import pdb; pdb.set_trace()
        return examples

    def _create_single_example(self, text_a, text_b):
        label_ids = '0'
        example = InputExample(text_a=text_a, text_b=text_b, labels=label_ids)
        return example

    def get_train_examples(self, data_path, size=-1):
        # filename = 'train.text'
        data = []
        with open(data_path, 'r', encoding='utf-8') as reader:
            for record in reader:
                data.append(record.strip())
        if size == -1:
            return self._create_examples(data)
        else:
            import random
            random.seed(1)
            data_sample = random.sample(data, size)
            return self._create_examples(data_sample)

    def get_dev_examples(self, data_path, size=-1):
        # filename = 'dev.text'
        data = []
        with open(data_path, 'r', encoding='utf-8') as reader:
            for record in reader:
                data.append(record.strip())
        if size == -1:
            return self._create_examples(data)
        else:
            import random
            random.seed(1)
            data_sample = random.sample(data, size)
            return self._create_examples(data_sample)

    def get_test_examples(self, data_path, size=-1):
        # filename = 'test.text'
        data = []
        with open(data_path, 'r', encoding='utf-8') as reader:
            for record in reader:
                data.append(record.strip())
        if size == -1:
            return self._create_examples(data)
        else:
            import random
            random.seed(1)
            data_sample = random.sample(data, size)
            return self._create_examples(data_sample)

    def get_labels(self, data_path):
        # filename = 'label.txt'
        data = []
        with open(data_path, 'r', encoding='utf-8') as reader:
            for record in reader:
                data.append(record.strip())
        return data


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    '''
    对于句子对形式进行截断的时候，每次对长的句子进行截断，因为短的句子携带更多的信息。
    '''
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for ex_idx, example in enumerate(examples):
        print(ex_idx)
        # if ex_idx > 1000:
        #     break
        tokens_a = tokenizer.tokenize(example.text_a)
        # tokens_a = tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # tokens_b = tokenize(example.text_b)
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length-3)
        else:
            if len(tokens_a) > max_seq_length-2:
                tokens_a = tokens_a[:max_seq_length-2]

        tokens = ['[CLS]'] + tokens_a + ['[SEP]']
        segment_ids = [0] * len(tokens)
        
        if tokens_b:
            tokens += tokens_b + ['[SEP]']
            segment_ids += [1] * (len(tokens_b)+1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # input_ids = convert_tokens_to_ids(tokens)

        input_mask = [1] * len(input_ids)

        padding = [0] * (max_seq_length-len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        # labels_ids = []
        # for label in example.labels:
        #     labels_ids.append(float(label_map[label]))
        labels_ids = float(label_map[example.labels])

        features.append(InputFeatures(input_ids=input_ids,
                                      input_mask = input_mask,
                                      segment_ids=segment_ids,
                                      label_ids=labels_ids))
    return features


def convert_single_example(example, max_seq_length, tokenizer):
    tokens_a = tokenizer.tokenize(example.text_a)
    # tokens_a = tokenize(example.text_a)

    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)
        # tokens_b = tokenize(example.text_b)
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length-3)
    else:
        if len(tokens_a) > max_seq_length-2:
            tokens_a = tokens_a[:max_seq_length-2]

    tokens = ['[CLS]'] + tokens_a + ['[SEP]']
    segment_ids = [0] * len(tokens)
    
    if tokens_b:
        tokens += tokens_b + ['[SEP]']
        segment_ids += [1] * (len(tokens_b)+1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    # input_ids = convert_tokens_to_ids(tokens)

    input_mask = [1] * len(input_ids)

    padding = [0] * (max_seq_length-len(input_ids))
    input_ids += padding
    input_mask += padding
    segment_ids += padding

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    # labels_ids = []
    # for label in example.labels:
    #     labels_ids.append(int(label))
    labels_ids = int(example.labels)

    feature = InputFeatures(input_ids=input_ids,
                            input_mask=input_mask,
                            segment_ids=segment_ids,
                            label_ids=labels_ids)
    return feature


def convert_single_example_dynamic(example, max_seq_length, tokenizer):
    '''单条推理，不需要pad
    '''
    tokens_a = tokenizer.tokenize(example.text_a)
    # tokens_a = tokenize(example.text_a)

    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)
        # tokens_b = tokenize(example.text_b)
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length-3)
    else:
        if len(tokens_a) > max_seq_length-2:
            tokens_a = tokens_a[:max_seq_length-2]

    tokens = ['[CLS]'] + tokens_a + ['[SEP]']
    segment_ids = [0] * len(tokens)
    
    if tokens_b:
        tokens += tokens_b + ['[SEP]']
        segment_ids += [1] * (len(tokens_b)+1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    input_mask = [1] * len(input_ids)

    labels_ids = int(example.labels)

    feature = InputFeatures(input_ids=input_ids,
                            input_mask=input_mask,
                            segment_ids=segment_ids,
                            label_ids=labels_ids)
    return feature