import tensorflow as tf
from keras.layers import SimpleRNN
from tensorflow.python.keras.layers import Embedding, Masking, LSTM, Dense, Dropout
from tensorflow.python.keras.models import Sequential


class RNNModel:
    def __init__(self, filepath, X_train):
        super(RNNModel, self).__init__()

        self.filepath = filepath

        self.model = self.__create_model(X_train)

        self.model.summary()

    def train(self, train_x, train_y, valid_x, valid_y):
        self.model.fit(x=train_x, y=train_y, validation_data=(valid_x, valid_y), epochs=200)
        self.model.save(filepath=self.filepath)

    def predict(self, test_x):
        self.model.predict(x=test_x)

    @staticmethod
    def __create_model(X_train):
        cell = tf.keras.layers.SimpleRNNCell(10)
        rnn = tf.keras.layers.RNN(cell)
        fc = tf.keras.layers.Dense(1)

        # model = Sequential()
        # model.add(Embedding(input_dim=3, output_dim=64))
        # model.add(rnn)
        # model.add(fc)

        model = Sequential()
        model.add(LSTM(units=96, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=96, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=96, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=96))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))

        model.compile(optimizer='adam', loss='mean_squared_error')

        # # # Embedding layer
        # model.add(Embedding(input_dim=3,
        #                     input_length=4,
        #                     output_dim=100,
        #                     mask_zero=True))
        #
        # # Masking layer for pre-trained embeddings
        # model.add(Masking(mask_value=0.0))
        #
        # # Recurrent layer
        # model.add(LSTM(64, return_sequences=False,
        #                dropout=0.1, recurrent_dropout=0.1))
        #
        # # Fully connected layer
        # model.add(Dense(64, activation='relu'))
        #
        # # Dropout for regularization
        # model.add(Dropout(0.5))
        #
        # # Output layer
        # model.add(Dense(1, activation='softmax'))

        # Compile the model

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
