from sklearn.model_selection import train_test_split


def split_dataset(data_path_list, to_train, to_test):
    dataSet = []
    labels = []

    for data_path in data_path_list:
        with open(data_path, 'r', encoding='utf-8') as reader:
            for record in reader:
                record_list = record.strip().split('\t')
                content, label = '\t'.join(record_list[:2]), record_list[2]
                dataSet.append(content)
                labels.append(label)

    X_train, X_test, y_train, y_test = train_test_split(dataSet, labels, test_size=0.1, random_state=2021)

    train = []
    test = []
    for x, y in zip(X_train, y_train):
        train.append(x + '\t' + y)
    for x, y in zip(X_test, y_test):
        test.append(x + '\t' + y)

    with open(to_train, 'w', encoding='utf-8') as writer:
        for record in train:
            writer.write(record + '\n')

    with open(to_test, 'w', encoding='utf-8') as writer:
        for record in test:
            writer.write(record + '\n')
    print('save successfully!')


if __name__ == '__main__':
    data_path_list = ['data_2_28/train_1.tsv', 'data_2_28/train_2.tsv']
    to_train = 'train_last.txt'
    to_test = 'test_last.txt'
    split_dataset(data_path_list, to_train, to_test)