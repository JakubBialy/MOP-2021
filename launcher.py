import os

import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from tensorflow.python.keras.layers import LSTM, Dropout, Dense

from config.config import get_model_dir
from data.crypto_archive_data_loader import CryptoArchiveDataLoader
from data.utils import divide_data
from ml.model import load_else_create, create
from stats.data_statistics import generate_prediction_xy_plot, plot_results
from utils.utils import colab_detected

if __name__ == '__main__':
    model_dir = get_model_dir()

    print(f'Colab detected: {colab_detected()}')

    # Data
    # CryptoArchiveDataLoader.clear_all_caches()  # Optional
    data_loader = CryptoArchiveDataLoader()
    data = data_loader.load('ETHUSDT')

    data = data.iloc[0:1000]

    data = data['open'].values
    data = data.reshape(-1, 1)

    # Data Normalziation

    # normalized_data, norm_meta = CryptoArchiveDataLoader.normalize(data)

    dataset_train = np.array(data[:int(data.shape[0] * 0.8)])
    dataset_test = np.array(data[int(data.shape[0] * 0.8) - 50:])
    scaler = MinMaxScaler(feature_range=(0, 1))

    dataset_train = scaler.fit_transform(dataset_train)
    dataset_test = scaler.fit_transform(dataset_test)


    def create_dataset(df):
        x = []
        y = []
        for i in range(50, df.shape[0]):
            x.append(df[i - 50:i, 0])
            y.append(df[i, 0])
        x = np.array(x)
        y = np.array(y)
        return x, y


    x_train, y_train = create_dataset(dataset_train)
    x_test, y_test = create_dataset(dataset_test)

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    model = Sequential()
    model.add(LSTM(units=96, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=96, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=96, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=96))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(loss='mean_squared_error', optimizer='adam')

    # if os.path.exists('eth_prediction.h5'):
        # model = load_model('eth_prediction.h5')
    # else:
    model.fit(x_train, y_train, epochs=50, batch_size=32)
        # model.save('eth_prediction.h5')

    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    fig, ax = plt.subplots(figsize=(8, 4))
    plt.plot(data, color='red', label="True Price")
    ax.plot(range(len(y_train) + 50, len(y_train) + 50 + len(predictions)), predictions, color='blue',
            label='Predicted Testing Price')
    plt.legend()

    y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(y_test_scaled, color='red', label='True Testing Price')
    plt.plot(predictions, color='blue', label='Predicted Testing Price')
    plt.legend()

    plt.show()