import importlib
import inspect
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

LOG_DIR = os.path.join(ROOT_DIR, 'logs')

RESULT_DIR = os.path.join(ROOT_DIR, 'results')
EXTRACTED_DATA_DIR = os.path.join(RESULT_DIR, 'extracted_data')
EXTRACTED_PARAMS_FILENAME = os.path.join(EXTRACTED_DATA_DIR, 'dataset_results.json')
PLOTS_DIR = os.path.join(RESULT_DIR, 'plots')
TABLES_DIR = os.path.join(RESULT_DIR, 'tables')

PATH_TO_DATA = os.path.join(Path(ROOT_DIR).resolve().parents[1], 'dataset', 'sklearn_full_cells.csv')
SKLEARN_PACKAGE_NAME = 'sklearn'


def get_classes_names_to_module_names(package_name, modules_list):
    cls_names_to_full_module_names = {}
    for module_name in modules_list:
        full_module_name = '.'.join([package_name, module_name])
        module = importlib.import_module(full_module_name)
        for class_name, obj in inspect.getmembers(module, inspect.isclass):
            cls_names_to_full_module_names[class_name] = full_module_name
    return cls_names_to_full_module_names


def remove_preprocessing_classes(classes_to_module_names,
                                 package_name=SKLEARN_PACKAGE_NAME,
                                 preprocessing_module_name='preprocessing'):
    preprocessing_classes = set(get_classes_names_to_module_names(package_name,
                                                                  [preprocessing_module_name]).keys())
    return {cls: classes_to_module_names[cls]
            for cls in classes_to_module_names.keys()
            if cls not in preprocessing_classes}


SKLEARN_MODULES_WITH_MODELS = ['cluster', 'discriminant_analysis', 'ensemble',
                               'kernel_ridge', 'linear_model', 'naive_bayes',
                               'neighbors', 'neural_network', 'semi_supervised',
                               'svm', 'tree']

MODEL_NAMES_TO_MODULE_NAMES = remove_preprocessing_classes(
    get_classes_names_to_module_names(SKLEARN_PACKAGE_NAME, SKLEARN_MODULES_WITH_MODELS)
)

ML_MODELS_LIST = list(MODEL_NAMES_TO_MODULE_NAMES.keys())
NOT_A_MODEL_NAME = 'NOT_A_MODEL_NAME'


