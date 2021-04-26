from shutil import copyfile
from shutil import copytree as copytree_to_new_dir

from testing_system.util.file_util import *


class KaggleEnvCreator:
    TRAIN_DIR_SUFFIX = 'train'
    VAL_DIR_SUFFIX = 'val'
    KAGGLE_MAIN_DIR_NAME = 'kaggle'
    KAGGLE_SOLUTION_DIR_NAME = 'solution'
    KAGGLE_DATA_DIR_NAME = 'input'

    def __init__(self,
                 env_dir_path: str,
                 path_to_ntb: str,
                 path_to_data_dir: str,
                 is_kaggle_competition: bool,
                 env_parent_dir=None):
        self.env_dir_path = env_dir_path
        self.path_to_ntb = path_to_ntb
        self.path_to_data_dir = path_to_data_dir
        self.is_kaggle_competition = is_kaggle_competition
        self.env_parent_dir = env_parent_dir

    def create_env(self):
        common_paths = [os.path.join(self.KAGGLE_MAIN_DIR_NAME,
                                     self.KAGGLE_DATA_DIR_NAME),
                        os.path.join(self.KAGGLE_MAIN_DIR_NAME,
                                     self.KAGGLE_SOLUTION_DIR_NAME)]

        env_train_dir_path = os.path.join(self.env_dir_path,
                                          self.__add_train_dir_suffix(self.env_dir_path))
        env_val_dir_path = os.path.join(self.env_dir_path,
                                        self.__add_val_dir_suffix(self.env_dir_path))

        for common_path in common_paths:
            os.makedirs(os.path.join(env_train_dir_path, common_path))

        for common_path in common_paths:
            os.makedirs(os.path.join(env_val_dir_path, common_path))

        path_to_copied_ntb_train = os.path.join(env_train_dir_path,
                                                common_paths[1],
                                                get_file_name_from_full_path(self.path_to_ntb))

        copyfile(self.path_to_ntb, path_to_copied_ntb_train)

        path_to_copied_ntb_val = os.path.join(env_val_dir_path,
                                              common_paths[1],
                                              get_file_name_from_full_path(self.path_to_ntb))

        copyfile(self.path_to_ntb, path_to_copied_ntb_val)

        if not self.is_kaggle_competition:
            path_to_copied_data_train_dir = os.path.join(env_train_dir_path,
                                                         common_paths[0],
                                                         get_dir_name_from_full_path(self.path_to_data_dir))
            copytree_to_new_dir(self.path_to_data_dir, path_to_copied_data_train_dir)

            path_to_copied_data_val_dir = os.path.join(env_val_dir_path,
                                                       common_paths[0],
                                                       get_dir_name_from_full_path(self.path_to_data_dir))
            copytree_to_new_dir(self.path_to_data_dir, path_to_copied_data_val_dir)

        else:  # self.is_kaggle_competition == True
            path_to_copied_data_train_dir = os.path.join(env_train_dir_path,
                                                         common_paths[0],
                                                         )
            copytree_to_existing_dir(self.path_to_data_dir, path_to_copied_data_train_dir)

            path_to_copied_data_val_dir = os.path.join(env_val_dir_path,
                                                       common_paths[0]
                                                       )
            copytree_to_existing_dir(self.path_to_data_dir, path_to_copied_data_val_dir)

    def __add_train_dir_suffix(self, dir_name):
        return dir_name + '_' + self.TRAIN_DIR_SUFFIX

    def __add_val_dir_suffix(self, dir_name):
        return dir_name + '_' + self.VAL_DIR_SUFFIX


def get_ntb_path_from_env(env_dir_path, train_or_val):
    if train_or_val not in ['train', 'val']:
        raise RuntimeError(f'train_or_val argument should be either "train" or "val"')

    if train_or_val == 'train':
        kaggle_dir_path = os.path.join(env_dir_path, f'{env_dir_path}_{KaggleEnvCreator.TRAIN_DIR_SUFFIX}',
                                       KaggleEnvCreator.KAGGLE_MAIN_DIR_NAME)
    else:
        kaggle_dir_path = os.path.join(env_dir_path, f'{env_dir_path}_{KaggleEnvCreator.VAL_DIR_SUFFIX}',
                                       KaggleEnvCreator.KAGGLE_MAIN_DIR_NAME)
    ipynb_files = find_all_files_with_ext(kaggle_dir_path, 'ipynb')
    return ipynb_files[0]  # expecting only one ipynb file in solution dir


def get_kaggle_dir_path_from_env(env_dir_path, train_or_val):
    if train_or_val not in ['train', 'val']:
        raise RuntimeError(f'train_or_val argument should be either "train" or "val"')

    if train_or_val == 'train':
        kaggle_dir_path = os.path.join(env_dir_path, f'{env_dir_path}_{KaggleEnvCreator.TRAIN_DIR_SUFFIX}',
                                       KaggleEnvCreator.KAGGLE_MAIN_DIR_NAME)
    else:
        kaggle_dir_path = os.path.join(env_dir_path, f'{env_dir_path}_{KaggleEnvCreator.VAL_DIR_SUFFIX}',
                                       KaggleEnvCreator.KAGGLE_MAIN_DIR_NAME)
    return kaggle_dir_path
