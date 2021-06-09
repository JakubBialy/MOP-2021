import matplotlib.pyplot as plt
import numpy as np


def generate_prediction_xy_plot(predicted, original, filepath=None):
    predicted = predicted.flatten()
    original = original.flatten()

    plt.figure()
    num_x = len(predicted)
    num_y = len(original)

    plt.plot(list(range(num_x)), predicted, color='b', label='Predicted prizes at close')
    plt.plot(list(range(num_y)), original, color='r', label='Original prizes at close')
    plt.title('Newest ETH/USD predictions prize')
    plt.legend()

    if filepath is not None:
        plt.savefig(filepath)
    else:
        plt.show()
        plt.clf()
        plt.cla()