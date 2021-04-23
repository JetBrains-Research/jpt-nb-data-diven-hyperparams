import ast
import importlib.machinery
import os
import sys
import zlib
from typing import Dict, Callable

import astor

from common_util.ast_util import get_func_name
from testing_system.code_extractor.code_extractor import NtbCodeExtractor
from testing_system.code_transformers.io_unifier import replace_abs_path_with_relative_path
from testing_system.code_wrapper.code_wrapper import CodeWrapper
from testing_system.util.file_util import create_py_file, cwd


class TrainedModelWrapper:
    def __init__(self, d: Dict):
        try:
            self.pickled_model_obj = d['pickled_model_obj']
            self.training_meta_inf = d['training_meta_inf']
        except KeyError as ke:
            raise RuntimeError(f'Invalid argument provided: {str(ke)}')


class HyperparamsChanger:
    def __init__(self, src, model_name, new_hyperparams):
        self.src = src
        self.model_name = model_name
        self.new_hyperparams = new_hyperparams

    def get_source(self) -> str:
        tree = ast.parse(self.src)
        self.__traverse(tree)
        return astor.to_source(tree)

    def __traverse(self, node):
        if not isinstance(node, ast.AST):
            return

        if not isinstance(node, ast.Call) or get_func_name(node) != self.model_name:
            for field_name in node._fields:
                field = getattr(node, field_name)
                if isinstance(field, list):
                    for elem in field:
                        self.__traverse(elem)
                self.__traverse(field)

        else:
            new_keywords = []
            for hp_name, hp_value in self.new_hyperparams.items():
                kwd = ast.keyword()
                kwd.arg = hp_name
                if sys.version_info.minor < 8:  # before Python 3.8:
                    kwd.value = ast.Str(hp_value) if isinstance(hp_value, str) else ast.Num(hp_value)
                else:  # Python 3.8+
                    kwd.value = ast.Constant(hp_value)
                new_keywords.append(kwd)
            node.keywords = new_keywords


class TrainedModelGenerator:
    def __init__(self, path_to_ntb: str,
                       kaggle_dir: str,
                       how_to_pickle: str = 'pickle'):

        self.path_to_ntb = path_to_ntb
        self.kaggle_dir = kaggle_dir
        self.kaggle_solution_dir = os.path.join(kaggle_dir, 'solution')
        self.how_to_pickle = how_to_pickle
        if not self.how_to_pickle == 'pickle':
            if self.how_to_pickle == 'joblib':
                raise RuntimeError(f'{self.how_to_pickle} pickling is not supported')
            else:
                raise RuntimeError(f'Unknown value for parameter "how_to_pickle": {self.how_to_pickle}')

    def get_trained_model(self, model_name: str, hyperparams: str = 'original') -> TrainedModelWrapper:
        nce = NtbCodeExtractor(self.path_to_ntb)
        if model_name not in nce.get_model_names():
            raise RuntimeError(f"Notebook {self.path_to_ntb} doesn't contain {model_name} model")
        if not self.check_valid_hyperparams(model_name, hyperparams):
            raise RuntimeError(f'Provided hyperparams={hyperparams} '
                               f'are not suitable for {model_name} model')

        training_script_src = nce.generate_training_script(model_name, how_to_pickle=self.how_to_pickle)
        training_script_src = replace_abs_path_with_relative_path(training_script_src)

        if hyperparams != 'original':
            training_script_src = HyperparamsChanger(training_script_src, model_name, new_hyperparams=hyperparams)\
                .get_source()

        training_func_name = self.__generate_wrapper_function_name(model_name, hyperparams)
        training_func_code = CodeWrapper.wrap_with_function(training_script_src,
                                                            training_func_name)

        training_py_fname = self.__generate_training_py_fname(model_name, hyperparams)
        create_py_file(training_func_code,
                       self.kaggle_solution_dir,
                       training_py_fname)

        training_f = self.__get_training_func_obj(training_py_fname, training_func_name)

        trained_model = self.__run_training_f(training_f)
        trained_model = TrainedModelWrapper(trained_model)

        return trained_model

    def check_valid_hyperparams(self, model_name, hyperparams) -> bool:
        # TODO: implement
        return True

    def __generate_wrapper_function_name(self, model_name, hyperparams) -> str:
        part_1 = f'auto_generated_training_f_{model_name}'
        common_str = f'{self.path_to_ntb + model_name + str(hyperparams)}'
        part_2 = f'{abs(zlib.adler32(bytes(common_str, encoding="utf-8")))}'
        return part_1 + '_' + part_2

    def __generate_training_py_fname(self, model_name, hyperparams) -> str:
        return self.__generate_wrapper_function_name(model_name, hyperparams) + '.py'

    def __get_training_func_obj(self, training_py_fname, training_function_name) -> Callable:

        path_to_train_func = os.path.join(self.kaggle_solution_dir, training_py_fname)
        train_func_module = importlib.machinery.SourceFileLoader('train_func_module',
                                                                  path_to_train_func).load_module()
        return getattr(train_func_module, training_function_name)

    def __run_training_f(self, training_f):
        with cwd(self.kaggle_solution_dir):
            return training_f()



