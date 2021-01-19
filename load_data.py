import scipy.io as scio
import numpy as np


def load_mat():
    train_data = scio.loadmat('./2020/train/1.mat')['de_feature']
    train_label = scio.loadmat('./2020/train/1.mat')['label']
    for i in range(2, 11):
        temp_data = scio.loadmat('./2020/train/' + str(i) + '.mat')['de_feature']
        temp_label = scio.loadmat('./2020/train/' + str(i) + '.mat')['label']
        train_data = np.concatenate((train_data, temp_data), axis=0)
        train_label = np.concatenate((train_label, temp_label), axis=0)

    test_data = scio.loadmat('./2020/test/11.mat')['de_feature']
    test_label = scio.loadmat('./2020/test/11.mat')['label']
    for i in range(12, 14):
        temp_data = scio.loadmat('./2020/test/' + str(i) + '.mat')['de_feature']
        temp_label = scio.loadmat('./2020/test/' + str(i) + '.mat')['label']
        test_data = np.concatenate((test_data, temp_data), axis=0)
        test_label = np.concatenate((test_label, temp_label), axis=0)

    train_data = (train_data - train_data.min(axis=0)) / (train_data.max(axis=0) - train_data.min(axis=0))
    test_data = (test_data - test_data.min(axis=0)) / (test_data.max(axis=0) - test_data.min(axis=0))
    train_label = np.array(train_label).reshape(len(train_label))
    test_label = np.array(test_label).reshape(len(test_label))

    return train_data, train_label, test_data, test_label


if __name__ == "__main__":
    train_data, train_label, test_data, test_label = load_mat()
    print(train_data.shape)
    print(train_label.shape)
    print(test_data.shape)
    print(test_label.shape)


