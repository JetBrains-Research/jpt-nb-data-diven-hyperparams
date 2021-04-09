#!/usr/bin/env python3
# Copyright (c) Aniskov N.

from typing import Dict, Any

import ast

from sklearn_hp_analisys.util.const import ML_MODELS_LIST
from sklearn_hp_analisys.util.file_util import *
from sklearn_hp_analisys.util.ast_util import get_func_name


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

        if not isinstance(node, ast.Call) or get_func_name(node) not in ML_MODELS_LIST:
            for field_name in node._fields:
                field = getattr(node, field_name)
                if isinstance(field, list):
                    for elem in field:
                        self.__extract_hyperparams(elem)
                self.__extract_hyperparams(field)

        else:
            hyperparams = {}
            self.__models_to_hyperparams_list.append(
                {'model_name': get_func_name(node), 'params': hyperparams})
            for keyword in node.keywords:
                if isinstance(keyword.value, ast.Constant):
                    hyperparams[keyword.arg] = keyword.value.value  # Python 3.8+
                elif isinstance(keyword.value, ast.Num):
                    hyperparams[keyword.arg] = keyword.value.n  # before Python 3.8
                elif isinstance(keyword.value, ast.Str):
                    hyperparams[keyword.arg] = keyword.value.s  # before Python 3.8

    def get_hyperparams(self) -> List[Dict[str, Any]]:
        if not self.__models_to_hyperparams_list:
            self.__extract_hyperparams(self.__tree)
        return self.__models_to_hyperparams_list


