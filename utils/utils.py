import sys


def colab_detected():
    return 'google.colab' in sys.modules
