import numpy as np
from pandas import DataFrame



def create_time_series(data, batches, series_length):
    assert series_length > 0
    assert batches > 0
    assert len(data) > 0
    assert len(data) >= series_length
    step = (len(data)) // (batches + 1)

    result = []
    for batch_index in range(batches):
        region_of_interest = data[batch_index * step: (batch_index * step) + series_length]
        result.append(np.asarray(region_of_interest).reshape((-1, 1)))

    return np.asarray(result)


def split_x_y_batches(df, batch_size, x_col_name, y_col_name):
    train_x = create_time_series(df[x_col_name].values, batches=batch_size, series_length=100)
    train_y = np.asarray(
        [(ts[-1],) for ts in create_time_series(df[y_col_name], batches=batch_size, series_length=100)])

    return train_x, train_y


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
