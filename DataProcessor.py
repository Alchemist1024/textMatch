# -*- encoding: utf-8 -*-
'''
@File        :DataProcessor.py
@Time        :2021/02/23 09:11:20
@Author      :xiaoqifeng
@Version     :1.0
@Contact:    :unknown
'''

from sklearn.model_selection import train_test_split


class Processor:
    def __init__(self):
        pass

    def get_data_from_json(self, data_path):
        dataSet = []

        with open(data_path, 'r', encoding='utf-8') as reader:
            data = eval(reader.read())
        for record in data:
            content = record['Content'].replace('\n', ' ').replace('\t', ' ')
            questions_info = record['Questions']
            for question_info in questions_info:
                question = question_info['Question']
                answer = question_info['Answer']
                choices = question_info['Choices']
                for choice in choices:
                    choice_answer = choice.split('.')[0]
                    if choice_answer == answer:
                        dataSet.append([content, question, choice, '1'])
                    else:
                        dataSet.append([content, question, choice, '0'])

        return dataSet

    def split_dataset(self, data):
        dataSet = []
        labels = []
        for record in data:
            content, label = record[:3], record[3]
            # import pdb; pdb.set_trace()
            dataSet.append(content)
            labels.append(label)

        X_train, X_test, y_train, y_test = train_test_split(dataSet, labels, test_size=0.2, random_state=0)
        train = []
        test = []
        for x, y in zip(X_train, y_train):
            # print(x)
            # print(y)
            # import pdb; pdb.set_trace()
            train.append(x + [y])
        for x, y in zip(X_test, y_test):
            test.append(x + [y])

        return train, test

    def save(self, data_path, to_train_path, to_test_path):
        dataSet = self.get_data_from_json(data_path)
        train, test = self.split_dataset(dataSet)

        with open(to_train_path, 'w', encoding='utf-8') as writer:
            for record in train:
                writer.write('\t'.join(record) + '\n')

        with open(to_test_path, 'w', encoding='utf-8') as writer:
            for record in test:
                writer.write('\t'.join(record) + '\n')

        print('save successfully')


if __name__ == '__main__':
    data_path = 'data/multichoice/train.json'
    to_train_path = 'data/multichoice/train.txt'
    to_test_path = 'data/multichoice/test.txt'
    processor = Processor()
    processor.save(data_path, to_train_path, to_test_path)