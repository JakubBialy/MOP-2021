import os

import tensorflow as tf

from config.config import get_model_dir
from data.crypto_archive_data_loader import CryptoArchiveDataLoader
from data.utils import divide_data, split_x_y_batches
from ml.model import load_else_create
from stats.data_statistics import generate_prediction_xy_plot



def main():
    tf.config.set_visible_devices([], 'GPU')

    #Hyperparameters
    BATCH_SIZE = 125
    DATA_SIZE = 25_000
    EPOCHS = 25

    BATCHES = DATA_SIZE // BATCH_SIZE

    model_dir = get_model_dir()
    model_filepath = os.path.join(model_dir, 'model_weights')

    # Data
    # CryptoArchiveDataLoader.clear_all_caches()  # Optional
    data_loader = CryptoArchiveDataLoader()
    data = data_loader.load('ETHUSDT')

    # Take last DATA_SIZE rows
    data = data.iloc[-DATA_SIZE:]

    # Normalization
    normalized_data, norm_meta = CryptoArchiveDataLoader.normalize(data, selected_cols=['open', 'close'])

    # Divide + split x/y (input/target)
    train_data, valid_data = divide_data(normalized_data, valid_percentage=20)
    train_x, train_y = split_x_y_batches(train_data, BATCHES, BATCH_SIZE, 'open', 'close')
    valid_x, valid_y = split_x_y_batches(valid_data, BATCHES, BATCH_SIZE, 'open', 'close')

    # Create model
    model = load_else_create(model_filepath, (BATCH_SIZE, 1))

    model.train(train_x, train_y, EPOCHS, BATCH_SIZE)
    model.summary()
    model.plot_model('model.png')
    model.save(model_filepath)

    predictions = model.predict(valid_x)
    predictions_denormalized = CryptoArchiveDataLoader.denormalize(norm_meta, predictions, 'close')
    valid_y_denormalized = CryptoArchiveDataLoader.denormalize(norm_meta, valid_y, 'close')

    generate_prediction_xy_plot(predictions_denormalized, valid_y_denormalized, 'image')

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


if __name__ == '__main__':
    main()
