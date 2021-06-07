import os

from config.config import get_model_dir
from data.crypto_archive_data_loader import CryptoArchiveDataLoader
from data.utils import divide_data, plot_results
from ml.model import load_else_create, create
from stats.data_statistics import generate_prediction_xy_plot
from utils.utils import colab_detected

if __name__ == '__main__':
    model_dir = get_model_dir()

    print(f'Colab detected: {colab_detected()}')

    # Data
    # CryptoArchiveDataLoader.clear_all_caches()  # Optional
    data_loader = CryptoArchiveDataLoader()
    data = data_loader.load('ETHUSDT')

    data = data.iloc[0:1000]

    # Data Normalziation
    normalized_data, norm_meta = CryptoArchiveDataLoader.normalize(data)

    # Dataset divide
    x_train, y_train, x_valid, y_valid, x_test, y_test = divide_data(normalized_data, 20, 10)

    # Delete Garbage:
    data = None
    normalized_data = None
    train_data = None
    valid_data = None

    # Model Train
    model = create(os.path.join(model_dir, 'model_0'))
    model.train(x_train, y_train, x_valid, y_valid)

    # Data Predict
    pred_x = x_test
    pred_y = model.predict(pred_x)

    # Statistics Generation
    plot_results(x_train, pred_y, y_test)
    generate_prediction_xy_plot(pred_x, pred_y)
