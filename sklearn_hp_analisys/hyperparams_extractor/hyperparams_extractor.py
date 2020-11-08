#!/usr/bin/env python3
# Copyright (c) Aniskov N.

import json
from typing import Dict, Any

import ast

from sklearn_hp_analisys.util.const import ML_MODELS_LIST
from sklearn_hp_analisys.util.file_util import *


class HyperparamsExtractor:
    """
    This class is extracting hyperparameters from python 3 source code
    of machine learning model training
    """
    def __init__(self, filename: str = None, src: str = None) -> None:
        """
        :param filename: path to .py file with python 3 source code
        """
        if filename is None and src is None:
            raise RuntimeError("One of the parameters filename or src_str must be not None")
        if filename:
            src = read_file_to_string(filename)
        self.__tree = ast.parse(src)
        self.__models_to_hyperparams_list = []

    def __extract_hyperparams(self, node: ast.AST) -> None:
        """
        recursively traverses the AST (starting at given node) and adds all found and collected
        dicts of the form {<MODEL_NAME>: <HYPERPARAMS>} to self.__models_to_hyperparams_list
        """
        if not isinstance(node, ast.AST):
            return

        if not isinstance(node, ast.Call) or self.__get_func_name(node) not in ML_MODELS_LIST:
            for field_name in node._fields:
                field = getattr(node, field_name)
                if isinstance(field, list):
                    for elem in field:
                        self.__extract_hyperparams(elem)
                self.__extract_hyperparams(field)

        else:
            hyperparams = {}
            self.__models_to_hyperparams_list.append(
                {'model_name': self.__get_func_name(node), 'params': hyperparams})
            for keyword in node.keywords:
                if isinstance(keyword.value, ast.Constant):
                    hyperparams[keyword.arg] = keyword.value.value  # Python 3.8+
                elif isinstance(keyword.value, ast.Num):
                    hyperparams[keyword.arg] = keyword.value.n  # before Python 3.8
                elif isinstance(keyword.value, ast.Str):
                    hyperparams[keyword.arg] = keyword.value.s  # before Python 3.8

    @staticmethod
    def __get_func_name(node: ast.Call) -> str:
        # Assume that 'foo' can't be model name in case of such Call: foo(...)(...)
        # Only in case of such Call: foo(...)
        if not isinstance(node, ast.Call):
            raise TypeError(f'Expected ast.Call node, but {type(node).__name__} received')
        NOT_A_MODEL_NAME = 'NOT_A_MODEL_NAME'
        try:
            return node.func.id
        except AttributeError:
            try:
                return node.func.attr
            except AttributeError:
                return NOT_A_MODEL_NAME

    def get_hyperparams(self) -> List[Dict[str, Any]]:
        if not self.__models_to_hyperparams_list:
            self.__extract_hyperparams(self.__tree)
        return self.__models_to_hyperparams_list


def write_results_to_file(result, out_filename):
    """
    This function extracts hyperparameters from all files located in samples_dir folder,
    creates out_filename file and writes the result of work to it.
    """

    with open(out_filename, 'w') as out_f:
        json.dump(result, out_f)


