from sklearn.model_selection import KFold
import numpy as np


def get_KFold(data_path, to_path):
    X = []
    labels = []
    with open(data_path, 'r', encoding='utf-8') as reader:
        for record in reader:
            sen1, sen2, label = record.strip().split('\t')
            X.append(sen1 + '\t' + sen2)
            labels.append(label)
    kf = KFold(n_splits=5, shuffle=True, random_state=10)

    k_res = kf.split(X)
    for idx, val in enumerate(k_res):
        train, test = val
        train_data = []
        test_data = []
        for i in train:
            record = X[i] + '\t' + labels[i]
            train_data.append(record)
        for i in test:
            record = X[i] + '\t' + labels[i]
            test_data.append(record)

        with open(to_path+'kf_'+str(idx+1)+'_train.txt', 'w', encoding='utf-8') as writer:
            for record in train_data:
                writer.write(record + '\n')
        with open(to_path+'kf_'+str(idx+1)+'_test.txt', 'w', encoding='utf-8') as writer:
            for record in test_data:
                writer.write(record + '\n')

    print('saved successfully!')


if __name__ == '__main__':
    data_path = 'data_2_28/train.tsv'
    to_path = './'
    get_KFold(data_path, to_path)