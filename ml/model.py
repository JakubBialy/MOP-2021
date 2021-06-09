from keras.layers import LSTM
from tensorflow.python.keras.layers import Dense, Dropout, GRU
from tensorflow.python.keras.models import Sequential


class RNNModel:
    def __init__(self, input_shape):
        super(RNNModel, self).__init__()
        self.model = self.__create_model(input_shape)
        self.model.build(input_shape)

    def train(self, train_x, train_y, epochs, batch_size, valid_x=None, valid_y=None):
        if valid_x is not None and valid_y is not None:
            self.model.fit(x=train_x, y=train_y, validation_data=(valid_x, valid_y), epochs=epochs,
                           batch_size=batch_size)
        else:
            self.model.fit(x=train_x, y=train_y, epochs=epochs, batch_size=batch_size)

    def save(self, filepath):
        self.model.save_weights(filepath, save_format='tf')
        # self.model.save(filepath)

    def predict(self, test_x):
        return self.model.predict(x=test_x)

    def summary(self):
        self.model.summary()

    @staticmethod
    def __create_model(input_shape):
        model = Sequential()
        model.add(LSTM(units=32, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.05))
        model.add(LSTM(units=64, return_sequences=True))
        model.add(Dropout(0.05))
        model.add(GRU(units=64))
        model.add(Dense(units=1))

        model.compile(loss='mean_squared_error', optimizer='adam')

        return model


def load(filepath, input_shape) -> RNNModel:
    try:
        model = RNNModel(input_shape)
        model.model.load_weights(filepath)
        return model
    except Exception as e:
        raise ModelLoadError()


def load_else_create(filepath, input_shape) -> RNNModel:
    try:
        return load(filepath, input_shape)
    except ModelLoadError:
        return create(input_shape)


def create(input_shape) -> RNNModel:
    model = RNNModel(input_shape)
    return model


class ModelLoadError(Exception):
    pass
