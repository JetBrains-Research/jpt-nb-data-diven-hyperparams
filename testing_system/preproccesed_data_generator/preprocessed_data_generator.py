import importlib.machinery
import os
import zlib
from typing import Callable

from testing_system.code_extractor.code_extractor import NtbCodeExtractor
from testing_system.code_transformers.io_unifier import replace_abs_path_with_relative_path
from testing_system.code_wrapper.code_wrapper import CodeWrapper
from testing_system.util.file_util import create_py_file

from testing_system.util.file_util import cwd

class PreprocessedDataGenerator:
    def __init__(self,
                 path_to_ntb: str,
                 kaggle_dir: str):
        self.path_to_ntb = path_to_ntb
        self.kaggle_dir = kaggle_dir
        self.kaggle_solution_dir = os.path.join(kaggle_dir, 'solution')

    def get_preprocessed_data(self):
        nce = NtbCodeExtractor(self.path_to_ntb)
        preproc_script_src = nce.get_preprocessing_script()
        preproc_script_src = replace_abs_path_with_relative_path(preproc_script_src)

        preproc_func_name = self.__generate_wrapper_function_name()
        preproc_func_src = CodeWrapper.wrap_with_function(preproc_script_src, preproc_func_name)

        preproc_py_fname = self.__generate_training_py_fname()
        create_py_file(preproc_func_src, self.kaggle_solution_dir, preproc_py_fname)

        preproc_f_obj = self.__get_preproc_func_obj(preproc_py_fname, preproc_func_name)
        X, y = self.__run_preproc_f(preproc_f_obj)

        return X, y

    def __generate_wrapper_function_name(self) -> str:
        part_1 = f'auto_generated_preproc_f'
        common_str = f'{self.path_to_ntb}'
        part_2 = f'{abs(zlib.adler32(bytes(common_str, encoding="utf-8")))}'
        return part_1 + '_' + part_2

    def __generate_training_py_fname(self) -> str:
        return self.__generate_wrapper_function_name() + '.py'

    def __get_preproc_func_obj(self, preproc_py_fname, preproc_function_name) -> Callable:

        path_to_train_func = os.path.join(self.kaggle_solution_dir, preproc_py_fname)
        train_func_module = importlib.machinery.SourceFileLoader('train_func_module',
                                                                  path_to_train_func).load_module()
        return getattr(train_func_module, preproc_function_name)

    def __run_preproc_f(self, preproc_f):
        with cwd(self.kaggle_solution_dir):
            return preproc_f()

