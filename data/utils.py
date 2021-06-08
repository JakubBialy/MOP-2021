import numpy as np
from matplotlib import pyplot as plt
from pandas import DataFrame


def divide_data(data: DataFrame, valid_percentage: int = 20, test_percentage: int = 10):
    valid_set_size = int(np.round(valid_percentage / 100 * data.shape[0]))
    test_set_size = int(np.round(test_percentage / 100 * data.shape[0]))
    train_set_size = data.shape[0] - (valid_set_size + test_set_size)

    assert train_set_size + valid_set_size + test_set_size == data.shape[0]

    data = data.to_numpy()

    x_train = data[1:train_set_size, :-1]
    y_train = data[1:train_set_size, -1]

    x_valid = data[1+train_set_size:train_set_size + valid_set_size, :-1]
    y_valid = data[1+train_set_size:train_set_size + valid_set_size, -1]

    x_test = data[train_set_size + valid_set_size:, :-1]
    y_test = data[train_set_size + valid_set_size:, -1]

    return [x_train, y_train, x_valid, y_valid, x_test, y_test]


def shuffle_dataset(data: DataFrame):
    shuffled = data.sample(frac=1).reset_index(drop=True)
    return shuffled