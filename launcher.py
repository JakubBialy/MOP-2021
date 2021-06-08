import os

import tensorflow as tf

from config.config import get_model_dir
from data.crypto_archive_data_loader import CryptoArchiveDataLoader
from data.utils import divide_data, split_x_y_batches
from ml.model import load_else_create
from stats.data_statistics import generate_prediction_xy_plot

if __name__ == '__main__':
    tf.config.set_visible_devices([], 'GPU')

    BATCH_SIZE = 8
    model_dir = get_model_dir()
    model_filepath = os.path.join(model_dir, 'model.h5')

    # Data
    # CryptoArchiveDataLoader.clear_all_caches()  # Optional
    data_loader = CryptoArchiveDataLoader()
    data = data_loader.load('ETHUSDT')

    # Take first 1k
    data = data.iloc[:1_000]

    # Normalization
    normalized_data, norm_meta = CryptoArchiveDataLoader.normalize(data, selected_cols=['open', 'close'])

    # Divide + split x/y (input/target)
    train_data, valid_data = divide_data(normalized_data, valid_percentage=20)
    train_x, train_y = split_x_y_batches(train_data, BATCH_SIZE, 'open', 'close')
    valid_x, valid_y = split_x_y_batches(valid_data, BATCH_SIZE, 'open', 'close')

    # Create model
    # model = RNNModel((None, 1))
    model = load_else_create(model_filepath, (None, 1))

    model.train(train_x, train_y, epochs=2, batch_size=BATCH_SIZE)
    model.summary()

    predictions = model.predict(valid_y)
    predictions_denormalized = CryptoArchiveDataLoader.denormalize(norm_meta, predictions, 'close')
    valid_y_denormalized = CryptoArchiveDataLoader.denormalize(norm_meta, valid_y, 'close')

    generate_prediction_xy_plot(predictions_denormalized, valid_y_denormalized)

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
