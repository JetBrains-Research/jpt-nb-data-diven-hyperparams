import importlib
import json
from typing import *

import numpy as np
from scipy.stats import wilcoxon, pearsonr
from sklearn.datasets import load_iris, load_digits, load_wine, load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from sklearn_hp_analisys.util.const import MODEL_NAMES_TO_MODULE_NAMES

CLASSIFICATION_DATASETS_LOADERS = [load_iris, load_digits, load_wine, load_breast_cancer]
DATASETS_NAMES = ['iris', 'digits', 'wine', 'breast_cancer']
DS_NAME_TO_LOADER = {DATASETS_NAMES[i]: CLASSIFICATION_DATASETS_LOADERS[i] for i in range(len(DATASETS_NAMES))}


class ModelPerformanceEstimator:

    def __init__(self, model_name: str, hp_name: str, hp_values: List[Any], default_hp_value: Any):
        self.model_name = model_name
        self.hp_name = hp_name
        self.hp_values = hp_values
        self.default_hp_value = default_hp_value

        hp_values_holder = {str(hp_val): [] for hp_val in hp_values}
        self.estimated_performance = {ds_name: hp_values_holder.copy() for ds_name in DATASETS_NAMES}

    def __estimate_on_dataset(self, ds_name: str, hp_val: Any, n_iter: int):
        try:
            print(f'dataset: {ds_name}, {self.hp_name} = {hp_val}')
            dataset = DS_NAME_TO_LOADER[ds_name]()
            module = importlib.import_module(MODEL_NAMES_TO_MODULE_NAMES[self.model_name])
            model = getattr(module, self.model_name)
            model_params = {self.hp_name: hp_val}

            X = dataset.data
            y = dataset.target

            accuracies = []
            for _ in tqdm(range(n_iter)):
                kf = KFold(n_splits=5, shuffle=True)
                for train_index, test_index in kf.split(X):
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]
                    clf = make_pipeline(StandardScaler(), model(**model_params))
                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)
                    acc = accuracy_score(y_test, y_pred)
                    accuracies.append(acc)
            print(np.mean(accuracies))
            print()
        except ValueError as e:
            print(f'SKIPPING {self.hp_name} = {hp_val} for {ds_name}: {str(e)}')
            return []

        return accuracies

    def estimate_on_all_datasets(self, n_iter: int):
        for ds_name in DATASETS_NAMES:
            for hp_val in self.hp_values:
                self.estimated_performance[ds_name][str(hp_val)] = self.__estimate_on_dataset(ds_name, hp_val, n_iter)

    def print_estimated_mean_accuracies(self):
        for ds_name in DATASETS_NAMES:
            print(f'{ds_name}\n')
            dataset_results = self.estimated_performance[ds_name]
            dr_view = [(np.mean(v), k) for k, v in dataset_results.items()]
            dr_view.sort(reverse=True)
            for v, k in dr_view:
                if k == str(self.default_hp_value):
                    print(f'default({k}): {v}\n')
                else:
                    print(f'{k}: {v}\n')
            print('-------------------------------')

    def get_hp_val_rating(self, ds_name):
        dataset_results = self.estimated_performance[ds_name]
        dr_view = [(np.mean(v), k) for k, v in dataset_results.items()]
        dr_view.sort(reverse=True)
        return [hp_val for (_, hp_val) in dr_view]

    def get_average_hp_rating(self):
        average_rating = {str(hp_val): [] for hp_val in self.hp_values}
        for ds_name in DATASETS_NAMES:
            ds_rating = self.get_hp_val_rating(ds_name)
            for hp_val in self.hp_values:
                average_rating[str(hp_val)].append(ds_rating.index(str(hp_val)))
        average_rating = {hp_val: np.mean(average_rating[hp_val]) for hp_val in average_rating.copy()}
        return average_rating

    def get_correlation_between_rating_and_freq(self):
        average_rating = self.get_average_hp_rating()
        sorted_by_rating = [hp_val for hp_val in sorted(average_rating, key=average_rating.get)]
        sorted_by_freq = self.hp_values

        sorted_by_rating = [float(e) for e in sorted_by_rating]
        sorted_by_freq = [float(e) for e in sorted_by_freq]

        return pearsonr(sorted_by_rating, sorted_by_freq)

    def perform_wilcoxon_on_estimated_accuracies(self, alpha=0.05):
        for dataset_name in DATASETS_NAMES:
            print(f'{dataset_name.upper()}\n')
            dataset_results = self.estimated_performance[dataset_name]
            dr_view = [(np.mean(v), k) for k, v in dataset_results.items()]
            dr_view.sort(reverse=True)
            for i in range(len(dr_view) - 1):

                hp_val_str_1 = dr_view[i][1]
                hp_val_str_2 = dr_view[i + 1][1]
                try:
                    _, p = wilcoxon(self.estimated_performance[dataset_name][hp_val_str_1],
                                    self.estimated_performance[dataset_name][hp_val_str_2])
                    if p > alpha:
                        print(f'{hp_val_str_1} VS {hp_val_str_2}: no statistical difference (alpha={alpha})')
                    else:
                        print(f'{hp_val_str_1} VS {hp_val_str_2}: {hp_val_str_1} significantly better than {hp_val_str_2}')
                except ValueError:
                    pass
                print()

    def save_estimated_performance_to_json(self, filename=None):
        if filename is None:
            filename = '.'.join(['estimated_performance_' + self.hp_name, 'json'])
        with open(filename, 'w') as out_json:
            json.dump(self.estimated_performance, out_json)

    def load_estimated_performance_from_json(self, filename=None):
        if filename is None:
            filename = '.'.join(['estimated_performance_' + self.hp_name, 'json'])
        with open(filename, 'r') as in_json:
            self.estimated_performance = json.load(in_json)



