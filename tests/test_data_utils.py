import unittest

from data.utils import divide_data
from tests.utils import generate_random_df


class TestDataUtils(unittest.TestCase):
    def test_divide_data_three_sets(self):
        rows = 1000
        valid_percentage = 50
        test_percentage = 25

        df = generate_random_df(cols=4, rows=1000)
        x_train, y_train, x_valid, y_valid, x_test, y_test = divide_data(df, valid_percentage, test_percentage)

        self.assertEqual(len(x_train), len(y_train))
        self.assertEqual(len(x_valid), len(y_valid))
        self.assertEqual(len(x_test), len(y_test))

        self.assertEqual(len(x_train), int((rows * ((100 - test_percentage - valid_percentage) / 100)) + 0.5))
        self.assertEqual(len(x_valid), int((rows * (valid_percentage / 100)) + 0.5))
        self.assertEqual(len(x_test), int((rows * (test_percentage / 100)) + 0.5))

    def test_divide_data_two_sets(self):
        rows = 1000
        valid_percentage = 50

        df = generate_random_df(cols=4, rows=1000)
        x_train, y_train, x_valid, y_valid = divide_data(df, valid_percentage)

        self.assertEqual(len(x_train), len(y_train))
        self.assertEqual(len(x_valid), len(y_valid))

        self.assertEqual(len(x_train), int((rows * ((100 - valid_percentage) / 100)) + 0.5))
        self.assertEqual(len(x_valid), int((rows * (valid_percentage / 100)) + 0.5))
