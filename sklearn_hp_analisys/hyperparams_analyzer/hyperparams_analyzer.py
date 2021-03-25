#!/usr/bin/env python3
# Copyright (c) Aniskov N.

import importlib
import inspect
import json
from typing import Dict
from matplotlib import pyplot as plt
import seaborn as sns
import os
import numpy as np

import pandas as pd

from sklearn_hp_analisys.util.const import ML_MODELS_LIST,\
    MODEL_NAMES_TO_MODULE_NAMES,\
    EXTRACTED_PARAMS_FILENAME,\
    PLOTS_DIR


def only_sklearn_hyperparams(ml_model_name: str, hyperparams: Dict):

    module_name = MODEL_NAMES_TO_MODULE_NAMES[ml_model_name]
    module = importlib.import_module(module_name)

    ModelClass = getattr(module, ml_model_name)

    sklearn_hyperparams_set = set(inspect.signature(ModelClass.__init__).parameters.keys())

    return dict(filter(lambda elem: elem[0] in sklearn_hyperparams_set, hyperparams.items()))


class CommonHyperparamsAnalyzer:
    def __init__(self, json_filename: str):
        with open(json_filename) as input_json:
            self.raw_data = json.load(input_json)

    def print_stats(self, ml_model_name, hyperparams=None, stats=None):

        if ml_model_name not in ML_MODELS_LIST:
            raise RuntimeError(f'Unknown model: {ml_model_name}')

        samples = []
        for entry in self.raw_data:
            if list(entry.keys())[0] == ml_model_name:
                sample = list(entry.values())[0]
                filtered_sample = only_sklearn_hyperparams(ml_model_name, sample)
                samples.append(filtered_sample)

        df = pd.DataFrame(samples)

        print('-' * 25, f'Stats for {ml_model_name} model', '-' * 25)
        if hyperparams is not None:
            print(df[hyperparams].describe().loc[stats])
        else:
            print(df.describe().loc[stats])


class HyperparamsAnalyzer:
    def __init__(self, json_filename, ml_model_name):
        with open(json_filename) as input_json:
            self.raw_data = json.load(input_json)

        if ml_model_name not in ML_MODELS_LIST:
            raise RuntimeError(f'Unknown model: {ml_model_name}')

        self.ml_model_name = ml_model_name

        samples = []
        for entry in self.raw_data:
            if list(entry.keys())[0] == ml_model_name:
                sample = list(entry.values())[0]
                filtered_sample = only_sklearn_hyperparams(ml_model_name, sample)
                samples.append(filtered_sample)

        self.df = pd.DataFrame(samples)

    def get_numeric_hp_analyzer(self, hp_name):
        if hp_name not in self.df.columns.values:
            raise RuntimeError(f'Unknown hyperparameter for {self.ml_model_name}: {hp_name}')
        return self.NumericHyperparamHandler(self.df[hp_name], hp_name)

    class NumericHyperparamHandler:
        def __init__(self, values: pd.Series, hp_name: str):
            self.hp_name = hp_name
            self.values = pd.to_numeric(values, errors='coerce').dropna()

        def values_with_dropped_outliers(self, percentile=0.15):
            Q1 = self.values.quantile(percentile)
            Q3 = self.values.quantile(1 - percentile)
            IQR = Q3 - Q1
            return self.values.where(lambda x: ~((x < (Q1 - 1.5 * IQR)) | (x > (Q3 + 1.5 * IQR)))).dropna()

        def describe_table(self, drop_outliers=True):
            if drop_outliers:
                return self.values_with_dropped_outliers().describe()
            return self.values.describe()

        def show_and_save_distplot(self, title, filename):
            sns_plot = sns.distplot(self.values_with_dropped_outliers())
            sns_plot.set_title(title)
            plt.show()
            sns_fig = sns_plot.get_figure()
            sns_fig.savefig(os.path.join(PLOTS_DIR, filename))
            plt.close(sns_fig)

        def get_distplot(self, title=None):
            if title is None:
                title = f'Hyperparameter: {self.hp_name}'
            sns_plot = sns.histplot(np.log10(self.values_with_dropped_outliers()), kde=True)
            sns_plot.set_title(title)
            return sns_plot

        def get_values(self, drop_outliers=True):
            if drop_outliers:
                return self.values_with_dropped_outliers()
            else:
                return self.values


def main():

    required_stats = ['mean', 'std', 'min', 'max']
    required_model = 'SVC'

    hp_analyzer = CommonHyperparamsAnalyzer(EXTRACTED_PARAMS_FILENAME)
    hp_analyzer.print_stats(required_model, stats=required_stats)


if __name__ == '__main__':
    main()
