import numpy as np
from keras.layers import SimpleRNN
from keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Dense

from config.config import get_model_dir
from data.crypto_archive_data_loader import CryptoArchiveDataLoader
from data.utils import divide_data
from utils.utils import colab_detected


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


if __name__ == '__main__':
    BATCH_SIZE = 8
    model_dir = get_model_dir()

    print(f'Colab detected: {colab_detected()}')

    # Data
    # CryptoArchiveDataLoader.clear_all_caches()  # Optional
    data_loader = CryptoArchiveDataLoader()
    data = data_loader.load('ETHUSDT')
    data = data.iloc[:1_000]
    normalized_data, norm_meta = CryptoArchiveDataLoader.normalize(data, selected_cols=['open', 'close'])

    train_data, valid_data = divide_data(normalized_data, valid_percentage=20)
    train_x = create_time_series(train_data['open'].values, batches=BATCH_SIZE, series_length=100)
    train_y = np.asarray(
        [(ts[-1],) for ts in create_time_series(train_data['close'], batches=BATCH_SIZE, series_length=100)])

    valid_x = create_time_series(valid_data['open'].values, batches=BATCH_SIZE, series_length=100)
    valid_y = np.asarray(
        [(ts[-1],) for ts in create_time_series(valid_data['close'], batches=BATCH_SIZE, series_length=100)])

    model = Sequential()
    model.add(SimpleRNN(units=20, return_sequences=True, input_shape=(None, 1)))
    model.add(Dropout(0.2))
    # model.add(LSTM(units=96, return_sequences=True))
    # model.add(Dropout(0.2))
    # model.add(LSTM(units=96, return_sequences=True))
    # model.add(Dropout(0.2))
    model.add(SimpleRNN(units=96))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(loss='mean_squared_error', optimizer='adam')

    # if os.path.exists('eth_prediction.h5'):
    # model = load_model('eth_prediction.h5')
    # else:
    model.fit(train_x, train_y, epochs=2, batch_size=BATCH_SIZE, verbose=2)
    # model.save('eth_prediction.h5')

    predictions = model.predict(valid_y)
    # predictions = scaler.inverse_transform(predictions)

    # fig, ax = plt.subplots(figsize=(8, 4))
    # plt.plot(data, color='red', label="True Price")
    # ax.plot(range(len(y_train) + 50, len(y_train) + 50 + len(predictions)), predictions, color='blue',
    #         label='Predicted Testing Price')
    # plt.legend()
    #
    # y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))
    #
    # fig, ax = plt.subplots(figsize=(8, 4))
    # ax.plot(y_test_scaled, color='red', label='True Testing Price')
    # plt.plot(predictions, color='blue', label='Predicted Testing Price')
    # plt.legend()
    #
    # plt.show()
