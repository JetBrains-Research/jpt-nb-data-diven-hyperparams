import ast
import json
import re
from dataclasses import dataclass
from typing import List

from common_util.ast_util import get_func_name
from common_util.const import ML_MODELS_LIST


class IPYNB_GRAMMAR:
    METADATA_KEY = 'metadata'
    CELLS_KEY = 'cells'
    CELL_TYPE_KEY = 'cell_type'
    EXECUTION_COUNT_KEY = 'execution_count'
    OUTPUTS_KEY = 'outputs'
    SOURCE_KEY = 'source'
    NBFORMAT_KEY = 'nbformat'
    NBFORMAT_MINOR_KEY = 'nbformat_minor'

    class DEFAULTS:
        DEFAULT_CELL_METADATA = {}
        DEFAULT_EXECUTION_COUNT = None
        DEFAULT_OUTPUTS = []

        DEFAULT_NB_METADATA = {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.7.8"
            }
        }
        DEFAULT_NBFORMAT = 4
        DEFAULT_NBFORMAT_MINOR = 1


class NtbCodeExtractorHelper:
    ntb_specific_patterns = [r'^%']

    @staticmethod
    def generate_code_cells_from_ipynb(path_to_ipynb_file):
        with open(path_to_ipynb_file) as ipynb_f:
            content = json.load(ipynb_f)
            for cell in content[IPYNB_GRAMMAR.CELLS_KEY]:
                if cell[IPYNB_GRAMMAR.CELL_TYPE_KEY] == 'code':
                    yield cell

    @staticmethod
    def all_cell_source(cell):
        res = ''
        stop_patterns = "(" + ")|(".join(NtbCodeExtractorHelper.ntb_specific_patterns) + ")"
        if isinstance(cell[IPYNB_GRAMMAR.SOURCE_KEY], list):
            for src in cell[IPYNB_GRAMMAR.SOURCE_KEY]:
                if re.match(stop_patterns, src) is None:
                    res += src + '\n'
        else:
            for src in cell[IPYNB_GRAMMAR.SOURCE_KEY].split('\n'):
                if re.match(stop_patterns, src) is None:
                    res += src + '\n'
        return res

    @staticmethod
    def generate_code_up_to(cell_no_, line_no_, path_to_ntb_file, including_last_line=False):
        res = ''
        for cell_no, cell in enumerate(NtbCodeExtractorHelper.generate_code_cells_from_ipynb(path_to_ntb_file)):
            if cell_no < cell_no_:
                res += NtbCodeExtractorHelper.all_cell_source(cell) + '\n'
            else:
                res += '\n'.join(NtbCodeExtractorHelper.all_cell_source(cell)
                                 .split('\n')[:line_no_ + including_last_line])
                break
        return res


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


class NtbCodeExtractor:
    model_info: List[NtbModelInfo]

    def __init__(self, path_to_ntb_file: str):
        self.path_to_ntb_file = path_to_ntb_file
        self.model_info = []
        self.__collect_model_info()

    class AstTraverser:
        model_info: List[NtbModelInfo]

        def __init__(self):
            self.model_info = []

        @staticmethod
        def continue_traverse(traverse, node, cell_no):
            for field_name in node._fields:
                field = getattr(node, field_name)
                if isinstance(field, list):
                    for elem in field:
                        traverse(elem, cell_no)
                traverse(field, cell_no)

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
                    ntb_model_info = NtbModelInfo()
                    ntb_model_info.constructor_location = NtbTokenLocationInfo(cell_no, node.lineno)
                    ntb_model_info.variable_name = node.targets[0].id
                    ntb_model_info.model_name = func_name
                    self.model_info.append(ntb_model_info)
                else:  # func_name == 'fit'
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
                            self.model_info.append(ntb_model_info)
                    except AttributeError:  # Everything ok, just wrong node
                        pass
            else:
                self.continue_traverse(self.traverse_collect_model_info, node, cell_no)

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
                self.continue_traverse(self.traverse_fill_missing_fit_loc, node, cell_no)

    def get_model_info(self):
        return self.model_info

    def get_model_names(self):
        return [mi.model_name for mi in self.model_info]

    def get_preprocessing_code(self) -> str:
        """
        By default returns code above constructor of first encountered model
        """
        first_model = self.model_info[0]
        first_model_cell_no, first_model_line_no = first_model.constructor_location.cell_no, \
                                                   first_model.constructor_location.line_no - 1

        return NtbCodeExtractorHelper.generate_code_up_to(first_model_cell_no,
                                                          first_model_line_no,
                                                          path_to_ntb_file=self.path_to_ntb_file,
                                                          including_last_line=False)

    def get_preprocessing_script(self) -> str:
        first_model = self.model_info[0]

        res = ''
        res += self.get_preprocessing_code()
        res += self.__generate_preproc_return_stmt(first_model.X_train_var_name, first_model.y_train_var_name)

        return res

    def generate_training_script(self, model_name=None, how_to_pickle='pickle') -> str:
        if model_name is not None:
            encountered_model_names = self.get_model_names()
            if model_name not in encountered_model_names:
                raise RuntimeError(f'Model {model_name} was not found in {self.path_to_ntb_file} notebook.')
            if model_name != encountered_model_names[0]:
                raise RuntimeError(f'Not Implemented. Required model {model_name} is not the first to occur'
                                   f'in {self.path_to_ntb_file} notebook.')

        if how_to_pickle not in ['pickle', 'joblib']:
            raise RuntimeError(f'Invalid argument for pickling strategy: {how_to_pickle}')

        first_model = self.model_info[0]
        if first_model.fit_location is None:
            raise RuntimeError(f'Fit location is missing for model {model_name}'
                               f' in {self.path_to_ntb_file} notebook')
        first_model_fit_cell_no, first_model_fit_line_no = first_model.fit_location.cell_no, \
                                                           first_model.fit_location.line_no - 1
        res = ''
        res += NtbCodeExtractorHelper.generate_code_up_to(first_model_fit_cell_no,
                                                          first_model_fit_line_no,
                                                          path_to_ntb_file=self.path_to_ntb_file,
                                                          including_last_line=True) + '\n'

        model_meta_inf_str = f'{{"training_sample_size": len({first_model.X_train_var_name})}}'
        res += self.__generate_training_return_stmt(first_model.variable_name,
                                                    how_to_pickle,
                                                    meta_inf_as_str=model_meta_inf_str)

        return res

    @staticmethod
    def __generate_training_return_stmt(model_var_name, how_to_pickle, meta_inf_as_str):
        stmt = ''
        if how_to_pickle == 'pickle':
            stmt += f'import pickle\n' \
                    f'return {{"pickled_model_obj": pickle.dumps({model_var_name}),' \
                    f'"training_meta_inf": {meta_inf_as_str}}}\n'
        elif how_to_pickle == 'joblib':
            raise RuntimeError(f'Joblib pickling is not supported')

        return stmt

    @staticmethod
    def __generate_preproc_return_stmt(X_train_var_name, y_train_var_name):
        return f'return {X_train_var_name}, {y_train_var_name}'

    # def __get_training_sample_size(self, model_name, model_var_name):
    #     mi = next((mi for mi in self.model_info if mi.variable_name == model_var_name and
    #                                                mi.model_name == model_name), None)
    #     fit_loc = mi.fit_location
    #     fit_cell = list(NtbCodeExtractorHelper.
    #                         generate_code_cells_from_ipynb(self.path_to_ntb_file))[fit_loc.cell_no]
    #     fit_line_src = (NtbCodeExtractorHelper.all_cell_source(fit_cell)).split('\n')[fit_loc.line_no]
    #     tree = ast.parse(fit_line_src)

    def __collect_model_info(self) -> None:
        ast_traverser = self.AstTraverser()
        for cell_no, cell in enumerate(NtbCodeExtractorHelper.generate_code_cells_from_ipynb(self.path_to_ntb_file)):
            tree = ast.parse(NtbCodeExtractorHelper.all_cell_source(cell))
            ast_traverser.traverse_collect_model_info(tree, cell_no)

        # filling missing fit() locations
        for cell_no, cell in enumerate(NtbCodeExtractorHelper.generate_code_cells_from_ipynb(self.path_to_ntb_file)):
            tree = ast.parse(NtbCodeExtractorHelper.all_cell_source(cell))
            ast_traverser.traverse_fill_missing_fit_loc(tree, cell_no)

        self.model_info = ast_traverser.model_info

















