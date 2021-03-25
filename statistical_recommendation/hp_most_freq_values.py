import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.signal import argrelextrema

from sklearn_hp_analisys.hyperparams_analyzer.hyperparams_analyzer import HyperparamsAnalyzer as HpAnalyzer
from sklearn_hp_analisys.util.const import EXTRACTED_PARAMS_FILENAME


def most_frequent_discrete(values):
    counts = values.value_counts()
    counts = pd.Series(counts.index.values, index=counts)
    sns.histplot(counts, kde=True).set(title='"n_estimators" for RandomForestClassifier', xlabel='n_estimators')
    plt.show()
    return counts.keys()


def most_frequent_non_discrete(values):
    sns_plot = sns.histplot(values, kde=True)
    plt.show()
    plot_data = sns_plot.get_lines()[0].get_data()
    values_axs = plot_data[0]
    density_axs = plot_data[1]

    extrema_idxs = argrelextrema(density_axs,
                                 comparator=np.greater,
                                 order=5)
    return values_axs[extrema_idxs]


def drop_outliers(arr: np.array, quantile):
    percentile = quantile * 100
    Q1 = arr.percentile(percentile)
    Q3 = arr.percentile(1 - percentile)
    IQR = Q3 - Q1
    return arr.where(lambda x: ~((x < (Q1 - 1.5 * IQR)) | (x > (Q3 + 1.5 * IQR)))).dropna()


def main():
    model_name = 'RandomForestClassifier'
    hp_name = 'n_estimators'

    hp_analyzer = HpAnalyzer(EXTRACTED_PARAMS_FILENAME, model_name)
    num_hp_analyzer = hp_analyzer.get_numeric_hp_analyzer(hp_name)

    values = num_hp_analyzer.get_values()
    # values_log_10 = np.log10(values)
    # values_log_10 = values_log_10[values_log_10 > -20]
    most_freq = most_frequent_discrete(values)
    print(most_freq[:7])



if __name__ == '__main__':
    main()