import matplotlib.pyplot as plt
import numpy as np


def generate_prediction_xy_plot(x, y, filepath=None):
    x = x.flatten()
    y = y.flatten()

    plt.figure()
    num_x = len(x)
    num_y = len(y)

    plt.plot(list(range(num_x)), x, color='b', label='predicted')
    plt.plot(list(range(num_y)), y, color='r', label='validation')
    plt.legend()

    if filepath is not None:
        plt.savefig(filepath)
    else:
        plt.show()
        plt.clf()
        plt.cla()


def plot_results(train_x, predictions, actual, filename=None):
    plt.figure()
    num_train = len(train_x)
    plt.plot(list(range(num_train)), train_x, color='b', label='training data')
    plt.plot(list(range(num_train, num_train + len(predictions))), predictions, color='r', label='predicted')
    plt.plot(list(range(num_train, num_train + len(actual))), actual, color='g', label='test data')
    plt.legend()
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()
