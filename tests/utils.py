from pandas import DataFrame
import numpy as np


def generate_random_df(cols, rows):
    return DataFrame(np.random.random((rows, cols)))
