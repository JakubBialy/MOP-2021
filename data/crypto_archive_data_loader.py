import math
import os
import shutil

import pandas as pd
import requests

from config.config import get_downloaded_dataset_dir
from data.normalization_meta import NormalizationMetaData


class CryptoArchiveDataLoader:
    def __init__(self,
                 base_url='https://www.cryptoarchive.com.au/download?filename=bars/',
                 extension='csv.gz'):
        self.base_url = base_url
        self.extension = extension

    def __generate_session_cookies(self):
        return {
            'PLAY_SESSION': 'eyJhbGciOiJIUzI1NiJ9.eyJkYXRhIjp7ImVtYWlsIjoibG9yb2RlMjI3OUBwaWRob2VzLmNvbSIsInJvbGVzIjoiVVNFUiIsImNzcmZUb2tlbiI6IjVjN2M3YzYwMzMxOWQ1MWY5MmFiOWM4NzRhYjU1M2JjOTYyYzk3NjctMTYyMjgwNDAwNzI4Mi0xZTBmZTQwY2FmOTJmZmEzMGRhYzhmNzcifSwibmJmIjoxNjIyODA0MDA3LCJpYXQiOjE2MjI4MDQwMDd9.-iJHgLdJ-aMpFg6p0PZvPme-kH1uaOi2Rl-70PCgNLo'}

    def __download_archive(self, url, filepath):
        file_dir = os.path.split(filepath)[0]
        if not os.path.exists(file_dir):
            os.makedirs(file_dir, exist_ok=True)

        with requests.get(url, stream=True, cookies=self.__generate_session_cookies()) as r:
            r.raise_for_status()
            with open(filepath, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return filepath

    def __create_url(self, symbol):
        return f'{self.base_url}{symbol}.{self.extension}'

    @staticmethod
    def clear_all_caches():
        shutil.rmtree(get_downloaded_dataset_dir())

    def load(self, symbol) -> pd.DataFrame:
        cache_dir = get_downloaded_dataset_dir()
        data_filename = f'{symbol}.{self.extension}'
        data_filepath = os.path.join(cache_dir, data_filename)

        if not os.path.exists(data_filepath):
            self.__download_archive(self.__create_url(symbol), data_filepath)

        df = pd.read_csv(data_filepath, compression='gzip', header=None, sep='|', quotechar='"', error_bad_lines=False,
                         names=['open timestamp', 'open', 'high', 'low', 'close', 'volume',
                                'taker buy quote asset volume', 'taker buy base asset volume', 'quote asset volume',
                                'number of trades'])
        df.set_index('open timestamp')

        df = self.__remove_unhelpful_columns(df)

        return df

    @staticmethod
    def __remove_unhelpful_columns(data: pd.DataFrame):
        return data.drop(['volume',
                          'taker buy quote asset volume',
                          'taker buy base asset volume',
                          'quote asset volume',
                          'number of trades'], axis=1)

    @staticmethod
    def minmax_normalize(x, min_val, max_val):
        assert math.fabs(max_val - min_val) > 0.0001

        return (x - min_val) / (max_val - min_val)

    @staticmethod
    def normalize(data: pd.DataFrame) -> (pd.DataFrame, NormalizationMetaData):
        norm_info = NormalizationMetaData()
        df = pd.DataFrame()
        for col_name in data:
            col_min = data[col_name].min()
            col_max = data[col_name].max()
            # col_mean = data[col_name].mean()
            # col_std = data[col_name].std()

            norm_info.set_col_metadata(col_name, {'min': col_min, 'max': col_max})

            df[col_name] = data[col_name].apply(lambda x: CryptoArchiveDataLoader.minmax_normalize(x, col_min, col_max))

        return df, norm_info

    @staticmethod
    def denormalize(normalization_meta_data: NormalizationMetaData, data: pd.DataFrame) -> pd.DataFrame:
        pass  # todo
