def divide_data(data: list, train_percentage: int = 75, valid_percentage: int = 25):
    num_samples = len(data)
    val_size = int(num_samples * (valid_percentage / 100.))
    train_size = num_samples - val_size

    assert train_percentage + valid_percentage == 100

    train_dataset = data[:train_size]
    val_dataset = data[train_size:]

    assert len(train_dataset) + len(val_dataset) == num_samples

    return train_dataset, val_dataset


def split_dataset(data):
    pass
