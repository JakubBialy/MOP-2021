import numpy as np
from pandas import DataFrame


def create_time_series(data, batches, series_length):
    assert series_length > 0
    assert batches > 0
    assert len(data) > 0
    assert len(data) >= series_length
    # step = (len(data)) // (batches + 1)
    step = (len(data) - series_length) / (batches)

    result = []
    for batch_index in range(batches):
        start_index = int(batch_index * step + 0.5)
        end_index = (int(batch_index * step + 0.5)) + series_length
        region_of_interest = data[start_index: end_index]

        result.append(np.asarray(region_of_interest))

    return np.asarray(result)


def split_x_y_batches(df, batches, steps_per_batch, x_col_name, y_col_name, future_distance=5):
    assert future_distance >= 0

    x_data = df[x_col_name].values
    y_data = []

    for i in range(future_distance, len(df)):
        y_data.append(df[y_col_name].iloc[i])

    x_data = x_data[:len(df) - future_distance]  # trim
    x_y_data = np.stack([x_data, y_data], axis=1)
    x_y_time_series = create_time_series(x_y_data, batches=batches, series_length=steps_per_batch)

    x_data = np.asarray([x[:, 0] for x in x_y_time_series]).reshape((batches, -1, 1))
    y_data = np.asarray([x[:, 1][-1] for x in x_y_time_series]).reshape(-1, 1)

    return x_data, y_data


def divide_data(data: DataFrame, test_percentage: int = 20, valid_percentage: int = None):
    # data = data.to_numpy()

    if valid_percentage is not None:
        assert 100 >= test_percentage >= 0 and \
               100 >= valid_percentage >= 0 and \
               100 >= valid_percentage + test_percentage >= 0

        valid_set_size = int(np.round(test_percentage / 100 * len(data)))
        test_set_size = int(np.round(valid_percentage / 100 * len(data)))
        train_set_size = len(data) - (valid_set_size + test_set_size)

        train_data = data[:train_set_size]
        valid_data = data[train_set_size:train_set_size + valid_set_size]
        test_data = data[train_set_size + valid_set_size:]

        return [train_data, valid_data, test_data]
    else:
        assert 100 >= test_percentage >= 0

        valid_set_size = int((test_percentage / 100 * len(data)) + 0.5)
        train_set_size = len(data) - valid_set_size

        train_data = data[:train_set_size]
        valid_data = data[train_set_size:train_set_size + valid_set_size]

        return [train_data, valid_data]


def shuffle_dataset(data: DataFrame):
    shuffled = data.sample(frac=1).reset_index(drop=True)
    return shuffled
