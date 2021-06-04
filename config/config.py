import os

from utils.utils import colab_detected


def get_model_dir():
    if colab_detected():  # Google Colab
        return '/content/drive/MyDrive/mop/model'
    elif os.name == 'nt':  # Windows
        return './model/'
    if os.name == 'posix':  # Linux
        return './model/'
    else:
        raise Exception('Unknown os.')


def get_downloaded_dataset_dir():
    if colab_detected():  # Google Colab
        return '/content/drive/MyDrive/mop/dataset_cache'
    elif os.name == 'nt':  # Windows
        return './dataset_cache/'
    if os.name == 'posix':  # Linux
        return './dataset_cache/'
    else:
        raise Exception('Unknown os.')
