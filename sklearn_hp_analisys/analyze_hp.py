from hyperparams_analyzer.hyperparams_analyzer import HyperparamsAnalyzer as HpAnalyzer
from util.const import EXTRACTED_PARAMS_FILENAME, ML_MODELS_LIST


def print_top_models(raw_data):
    ml_model_to_cnt = {model_name: 0 for model_name in ML_MODELS_LIST}

    for sample in raw_data:
        current_model_name = list(sample.keys())[0]
        ml_model_to_cnt[current_model_name] += 1

    print('Models frequencies in dataset:')
    idx = 0
    for model_name, cnt in sorted(ml_model_to_cnt.items(), key=lambda x: x[1], reverse=True):
        if model_name != 'StandardScaler':
            print(f'{idx + 1}. {model_name}: {cnt}')
            idx += 1


if __name__ == '__main__':
    top_models = ['LogisticRegression', 'SVC']
    simple_stats = ['mean', 'std', 'min', 'max']

    # model_name = 'LogisticRegression'
    hp_name = 'C'
    for model_name in top_models:
        hp_analyzer = HpAnalyzer(EXTRACTED_PARAMS_FILENAME, model_name)

        num_hp_analyzer = hp_analyzer.get_numeric_hp_analyzer(hp_name)
        num_hp_analyzer.distplot(f'"{hp_name}" hyperparameter for {model_name}', '.'.join([model_name, hp_name, 'jpg']))
        print(num_hp_analyzer.describe_table())




