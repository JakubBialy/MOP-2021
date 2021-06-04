import os

from config.config import get_model_dir
from data.crypto_archive_data_loader import CryptoArchiveDataLoader
from data.utils import divide_data, split_dataset
from ml.model import load_else_create
from stats.data_statistics import generate_prediction_xy_plot
from utils.utils import colab_detected

if __name__ == '__main__':
    model_dir = get_model_dir()

    print(f'Colab detected: {colab_detected()}')

    # Data
    # CryptoArchiveDataLoader.clear_all_caches()  # Optional
    data_loader = CryptoArchiveDataLoader()
    data = data_loader.load('ETHUSDT')

    # Data Normalziation
    normalized_data, norm_meta = CryptoArchiveDataLoader.normalize(data)

    # todo Data Shuffle

    # Dataset divide
    train_data, valid_data = divide_data(normalized_data, 75, 25)

    # Dataset split
    train_x, train_y = split_dataset(train_data)
    valid_x, valid_y = split_dataset(valid_data)

    # Delete Garbage:
    data = None
    normalized_data = None
    train_data = None
    valid_data = None

    # Model Train
    model = load_else_create(os.path.join(model_dir, 'model_0'))
    model.train(train_x, train_y, valid_x, valid_y)

    # Data Predict
    pred_x = None
    pred_y = model.predict(pred_x)

    # Statistics Generation
    generate_prediction_xy_plot(pred_x, pred_y)
