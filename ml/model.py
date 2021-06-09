import tensorflow as tf
from keras.layers import SimpleRNN, LSTM
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras.models import Sequential


class RNNModel:
    def __init__(self, input_shape):
        super(RNNModel, self).__init__()
        self.model = self.__create_model(input_shape)

    def train(self, train_x, train_y, epochs, batch_size, valid_x=None, valid_y=None):
        if valid_x is not None and valid_y is not None:
            self.model.fit(x=train_x, y=train_y, validation_data=(valid_x, valid_y), epochs=epochs,
                           batch_size=batch_size)
        else:
            self.model.fit(x=train_x, y=train_y, epochs=epochs, batch_size=batch_size)

    def save(self, filepath):
        self.model.save(filepath)

    def predict(self, test_x):
        return self.model.predict(x=test_x)

    def summary(self):
        self.model.summary()

    @staticmethod
    def __create_model(input_shape):
        model = Sequential()
        model.add(LSTM(units=20, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        # model.add(LSTM(units=96, return_sequences=True))
        # model.add(Dropout(0.2))
        # model.add(LSTM(units=96, return_sequences=True))
        # model.add(Dropout(0.2))
        model.add(LSTM(units=96))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))

        model.compile(loss='mean_squared_error', optimizer='adam')

        return model


def load(input_shape) -> RNNModel:
    try:
        model = RNNModel(input_shape)
        return model
    except Exception as e:
        raise ModelLoadError()


def load_else_create(input_shape) -> RNNModel:
    try:
        return load(input_shape)
    except ModelLoadError:
        return create(input_shape)


def create(input_shape) -> RNNModel:
    model = RNNModel(input_shape)
    return model


class ModelLoadError(Exception):
    pass
