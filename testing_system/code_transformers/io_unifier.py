import ast
import os
from typing import Optional, List
import re

import astor


def replace_abs_path_with_relative_path(src: str) -> str:
    return re.sub(r'/kaggle', '..', src)


# class IoUnifier: #
#     """
#     Class provides functionality to replace user-specific paths
#     to input/output files with unified ones.
#
#
#
#     """
#
#     def __init__(self, common_dir_name: str, io_func_names: List[str]):
#         self.common_dir_name = common_dir_name
#         self.io_func_names = io_func_names
#
#     def replace_io_path(self, node: ast.Call) -> None:
#         if not isinstance(node, ast.Call):
#             raise TypeError(f'Expected ast.Call node, but {type(node).__name__} received')
#         orig_path = node.args[0].s
#         unified_path = os.path.join(self.common_dir_name, orig_path.split('/')[-1])
#         node.args[0].s = unified_path
#
#     @staticmethod
#     def __get_func_name(node: ast.Call) -> Optional[str]:
#         if not isinstance(node, ast.Call):
#             raise TypeError(f'Expected ast.Call node, but {type(node).__name__} received')
#         try:
#             return node.func.id
#         except AttributeError:
#             try:
#                 return node.func.attr
#             except AttributeError:
#                 return None
#
#     def unify(self, src: str):
#         tree = ast.parse(src)
#
#         def traverse(node: ast.AST):
#             if not isinstance(node, ast.AST):
#                 return
#
#             if not isinstance(node, ast.Call) or IoUnifier.__get_func_name(node) not in self.io_func_names:
#                 for field_name in node._fields:
#                     field = getattr(node, field_name)
#                     if isinstance(field, list):
#                         for elem in field:
#                             self.replace_io_path(elem)
#                     self.replace_io_path(field)
#             else:
#                 self.replace_io_path(node)
#
#         traverse(tree)
#         return astor.to_source(tree)

