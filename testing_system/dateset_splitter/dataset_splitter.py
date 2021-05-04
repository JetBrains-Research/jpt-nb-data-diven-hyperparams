import os
import re

import pandas as pd
from sklearn.model_selection import train_test_split
from testing_system.util.file_util import get_dir_name_from_full_path, find_all_files_with_ext, detect_csv_delimiter
from testing_system.kaggle_enviroment_creator.kaggle_enviroment_creator import KaggleEnvCreator
from typing import List


def split_tabular_dataset(path_to_csv_data,
                          path_to_produced_train,
                          path_to_produced_test,
                          test_size=0.3) -> None:
    csv_sep = detect_csv_delimiter(path_to_csv_data)

    df = pd.read_csv(path_to_csv_data, sep=csv_sep)
    df_train, df_test = train_test_split(df, test_size=test_size)

    df_train.to_csv(path_or_buf=path_to_produced_train, sep=csv_sep)
    df_test.to_csv(path_or_buf=path_to_produced_test, sep=csv_sep)


class KaggleEnvTabularDatasetSplitter:
    """
    Bypasses path_to_kaggle_env_dir replacing original data in both dirs
    with train and validation pieces in train and val dirs respectively
    """

    def __init__(self,
                 path_to_kaggle_env_dir: str,
                 test_size=0.3
                 ):
        self.path_to_kaggle_env_dir = path_to_kaggle_env_dir
        self.kaggle_env_dir_name = get_dir_name_from_full_path(path_to_kaggle_env_dir)
        self.test_size = test_size

    def split_data(self) -> None:
        train_dir_path = os.path.join(self.path_to_kaggle_env_dir,
                                      f'{self.kaggle_env_dir_name}_{KaggleEnvCreator.TRAIN_DIR_SUFFIX}')
        val_dir_path = os.path.join(self.path_to_kaggle_env_dir,
                                    f'{self.kaggle_env_dir_name}_{KaggleEnvCreator.VAL_DIR_SUFFIX}')

        possible_train_csv_files = KaggleEnvTabularDatasetSplitter.__filter_train_related_csv_files(
            find_all_files_with_ext(train_dir_path, 'csv', full_paths=True)
        )

        possible_train_csv_files = [os.path.relpath(p, train_dir_path) for p in possible_train_csv_files]

        for possible_train_csv_file in possible_train_csv_files:
            path_to_file_in_train_dir = os.path.join(train_dir_path, possible_train_csv_file)
            path_to_file_in_val_dir = os.path.join(val_dir_path, possible_train_csv_file)
            split_tabular_dataset(path_to_file_in_train_dir,
                                  path_to_file_in_train_dir,  # replaces data in train dir with only it's "train" part
                                  path_to_file_in_val_dir,
                                  test_size=self.test_size)

    @staticmethod
    def __filter_train_related_csv_files(file_paths: List[str]):  # Heuristic
        NON_TRAIN_CSV_TOKENS = ['test', 'submission']
        return list(filter(lambda path: all(re.search(stop_token, path) is not None
                                            for stop_token in NON_TRAIN_CSV_TOKENS),
                           file_paths)
                    )
