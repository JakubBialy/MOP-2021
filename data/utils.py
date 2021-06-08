import numpy as np
from pandas import DataFrame


def divide_data(data: DataFrame, valid_percentage: int = 20, test_percentage: int = None):
    # data = data.to_numpy()

    if test_percentage is not None:
        assert 100 >= valid_percentage >= 0 and \
               100 >= test_percentage >= 0 and \
               100 >= test_percentage + valid_percentage >= 0

        valid_set_size = int(np.round(valid_percentage / 100 * len(data)))
        test_set_size = int(np.round(test_percentage / 100 * len(data)))
        train_set_size = len(data) - (valid_set_size + test_set_size)

        train_data = data[:train_set_size]
        valid_data = data[train_set_size:train_set_size + valid_set_size]
        test_data = data[train_set_size + valid_set_size:]

        return [train_data, valid_data, test_data]
    else:
        assert 100 >= valid_percentage >= 0

        valid_set_size = int((valid_percentage / 100 * len(data)) + 0.5)
        train_set_size = len(data) - valid_set_size

        train_data = data[:train_set_size]
        valid_data = data[train_set_size:train_set_size + valid_set_size]

        return [train_data, valid_data]


def shuffle_dataset(data: DataFrame):
    shuffled = data.sample(frac=1).reset_index(drop=True)
    return shuffled
