import ast
from dataclasses import dataclass
from typing import List

from common_util.const import NOT_A_MODEL_NAME, ML_MODELS_LIST


def get_func_name(node: ast.Call) -> str:
    # Assume that 'foo' can't be model name in case of such Call: foo(...)(...)
    # Only in case of such Call: foo(...)
    if not isinstance(node, ast.Call):
        raise TypeError(f'Expected ast.Call node, but {type(node).__name__} received')

    try:
        return node.func.id
    except AttributeError:
        try:
            return node.func.attr
        except AttributeError:
            return NOT_A_MODEL_NAME


def continue_traverse(traverse, node, cell_no):
    for field_name in node._fields:
        field = getattr(node, field_name)
        if isinstance(field, list):
            for elem in field:
                traverse(elem, cell_no)
        traverse(field, cell_no)


@dataclass
class NtbTokenLocationInfo:
    cell_no: int = None
    line_no: int = None


@dataclass
class NtbModelInfo:
    model_name: str = None
    variable_name: str = None
    constructor_location: NtbTokenLocationInfo = None
    fit_location: NtbTokenLocationInfo = None
    X_train_var_name: str = None
    y_train_var_name: str = None


class NtbModelInfoAstTraverser:
    model_info: List[NtbModelInfo]

    def __init__(self):
        self.model_info = []

    def traverse_collect_model_info(self, node: ast.AST, cell_no: int):
        if not isinstance(node, ast.AST):
            return

        candidate_node_predicate = isinstance(node, ast.Assign) and \
                                   isinstance(node.value, ast.Call) and \
                                   get_func_name(node.value) in ML_MODELS_LIST + ['fit']

        if candidate_node_predicate:

            call_node = node.value
            func_name = get_func_name(call_node)

            if func_name in ML_MODELS_LIST:
                ntb_model_info = self.__get_model_info_for_model_node(node, func_name, cell_no)

            else:  # func_name == 'fit'
                ntb_model_info = self.__get_model_info_for_fit_node(call_node, node, cell_no)

            if ntb_model_info is not None:
                self.model_info.append(ntb_model_info)
        else:
            continue_traverse(self.traverse_collect_model_info, node, cell_no)

    def traverse_fill_missing_fit_loc(self, node: ast.AST, cell_no: int):
        if not isinstance(node, ast.AST):
            return

        missing_fit_loc_models = [mi for mi in self.model_info if mi.fit_location is None]

        if len(missing_fit_loc_models) == 0:
            return

        if isinstance(node, ast.Call) and get_func_name(node) == 'fit':
            try:
                mi = next((mi for mi in missing_fit_loc_models if mi.variable_name == node.func.value.id), None)
                if mi is not None:
                    mi.fit_location = NtbTokenLocationInfo(cell_no, node.lineno)
                    mi.X_train_var_name = node.args[0].id
                    mi.y_train_var_name = node.args[1].id
            except AttributeError:
                pass
        else:
            continue_traverse(self.traverse_fill_missing_fit_loc, node, cell_no)

    @staticmethod
    def __get_model_info_for_model_node(node, func_name, cell_no):
        ntb_model_info = NtbModelInfo()
        ntb_model_info.constructor_location = NtbTokenLocationInfo(cell_no, node.lineno)
        ntb_model_info.variable_name = node.targets[0].id
        ntb_model_info.model_name = func_name
        return ntb_model_info

    @staticmethod
    def __get_model_info_for_fit_node(call_node, node, cell_no):
        try:
            func_name_2 = get_func_name(call_node.func.value)
            if func_name_2 in ML_MODELS_LIST:
                ntb_model_info = NtbModelInfo()
                ntb_model_info.constructor_location = NtbTokenLocationInfo(cell_no, node.lineno)
                ntb_model_info.variable_name = node.targets[0].id
                ntb_model_info.model_name = func_name_2
                ntb_model_info.fit_location = NtbTokenLocationInfo(cell_no, node.lineno)
                ntb_model_info.X_train_var_name = call_node.args[0].id
                ntb_model_info.y_train_var_name = call_node.args[1].id
                return ntb_model_info
        except AttributeError:  # Everything ok, just wrong node
            return None
