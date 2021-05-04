import ast
import json
import re
from typing import List

from common_util.ast_util import NtbModelInfo, NtbModelInfoAstTraverser
from common_util.ipynb_grammar import IPYNB_GRAMMAR


class NtbCodeExtractor:
    model_info: List[NtbModelInfo]

    __ntb_specific_patterns = [r'^%']

    def __init__(self, path_to_ntb_file: str):
        self.path_to_ntb_file = path_to_ntb_file
        with open(path_to_ntb_file) as ipynb_f:
            self.ntb_json_content = json.load(ipynb_f)
        self.model_info = []
        self.__collect_model_info()

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
        src_str = '\n'.join(list(self.__generate_code_up_to(first_model_cell_no,
                                                            first_model_line_no,
                                                            including_last_line=False)))
        return src_str

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

        first_model = self.model_info[0]
        if first_model.fit_location is None:
            raise RuntimeError(f'Fit location is missing for model {model_name}'
                               f' in {self.path_to_ntb_file} notebook')
        first_model_fit_cell_no, first_model_fit_line_no = first_model.fit_location.cell_no, \
                                                           first_model.fit_location.line_no - 1
        src_str = ''
        src_str += '\n'.join(list(self.__generate_code_up_to(first_model_fit_cell_no,
                                                             first_model_fit_line_no,
                                                             including_last_line=True))) + '\n'

        model_meta_inf_str = f'{{"training_sample_size": len({first_model.X_train_var_name})}}'
        src_str += self.__get_training_return_stmt(first_model.variable_name,
                                                   how_to_pickle,
                                                   meta_inf_as_str=model_meta_inf_str)

        return src_str

    @staticmethod
    def __get_training_return_stmt(model_var_name, how_to_pickle, meta_inf_as_str):
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

    def __collect_model_info(self) -> None:
        ast_traverser = NtbModelInfoAstTraverser()
        for cell_no, cell in enumerate(self.__generate_code_cells_from_ipynb()):
            tree = ast.parse(self.__all_cell_source(cell))
            ast_traverser.traverse_collect_model_info(tree, cell_no)

        # filling missing fit() locations
        for cell_no, cell in enumerate(self.__generate_code_cells_from_ipynb()):
            tree = ast.parse(self.__all_cell_source(cell))
            ast_traverser.traverse_fill_missing_fit_loc(tree, cell_no)

        self.model_info = ast_traverser.model_info

    def __generate_code_cells_from_ipynb(self):
        for cell in self.ntb_json_content[IPYNB_GRAMMAR.CELLS_KEY]:
            if cell[IPYNB_GRAMMAR.CELL_TYPE_KEY] == 'code':
                yield cell

    def __all_cell_source(self, cell):
        res = ''
        stop_patterns = "(" + ")|(".join(self.__ntb_specific_patterns) + ")"
        if isinstance(cell[IPYNB_GRAMMAR.SOURCE_KEY], list):
            for src in cell[IPYNB_GRAMMAR.SOURCE_KEY]:
                if re.match(stop_patterns, src) is None:
                    res += src + '\n'
        else:
            for src in cell[IPYNB_GRAMMAR.SOURCE_KEY].split('\n'):
                if re.match(stop_patterns, src) is None:
                    res += src + '\n'
        return res

    def __generate_code_up_to(self, cell_no_, line_no_, including_last_line=False):
        for cell_no, cell in enumerate(self.__generate_code_cells_from_ipynb()):
            if cell_no < cell_no_:
                yield self.__all_cell_source(cell)
            else:
                yield '\n'.join(self.__all_cell_source(cell)
                                .split('\n')[:line_no_ + including_last_line])
                break
