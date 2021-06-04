import tensorflow as tf

class RNNModel:
    def __init__(self):
        pass

    def train(self, train_x, train_y, valid_x, valid_y):
        pass

    def predict(self, x):
        pass


class ModelLoadError(Exception):
    pass


@staticmethod
def load(filepath) -> RNNModel:
    pass


@staticmethod
def load_else_create(filepath) -> RNNModel:
    try:
        return load(filepath)
    except ModelLoadError as e:
        return create(filepath)


@staticmethod
def create(filepath) -> RNNModel:
    pass
