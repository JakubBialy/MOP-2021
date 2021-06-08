import tensorflow as tf
from keras.layers import SimpleRNN
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras.models import Sequential


class RNNModel:
    def __init__(self, filepath, input_shape):
        super(RNNModel, self).__init__()

        self.filepath = filepath

        self.model = self.__create_model(input_shape)

    def train(self, train_x, train_y, valid_x, valid_y, epochs, batch_size):
        # self.model.fit(x=train_x, y=train_y, validation_data=(valid_x, valid_y), epochs=200)
        self.model.fit(x=train_x, y=train_y, epochs=2, batch_size=batch_size)
        # self.model.save(filepath=self.filepath) # todo

    def predict(self, test_x):
        return self.model.predict(x=test_x)

    def summary(self):
        self.model.summary()

    @staticmethod
    def __create_model(input_shape):
        model = Sequential()
        model.add(SimpleRNN(units=20, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        # model.add(LSTM(units=96, return_sequences=True))
        # model.add(Dropout(0.2))
        # model.add(LSTM(units=96, return_sequences=True))
        # model.add(Dropout(0.2))
        model.add(SimpleRNN(units=96))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))

        model.compile(loss='mean_squared_error', optimizer='adam')

        return model


def load(filepath) -> RNNModel:
    model = tf.keras.models.load_model(filepath)
    return model


def load_else_create(filepath, X_train) -> RNNModel:
    try:
        return load(filepath)
    except OSError:
        return create(filepath, X_train)


def create(filepath, X_train) -> RNNModel:
    model = RNNModel(filepath, X_train)
    return model


class ModelLoadError(Exception):
    pass
