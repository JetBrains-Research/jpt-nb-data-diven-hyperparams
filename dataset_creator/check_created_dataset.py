import os
import json

from sklearn_hp_analisys.util.const import EXTRACTED_DATA_DIR


def main():
    path_to_dataset = 'results/created_dataset_v2.json'
    path_to_hyperparams_data = os.path.join(EXTRACTED_DATA_DIR, 'dataset_results_v3.json')
    model_name = 'RandomForestClassifier'

    with open(path_to_hyperparams_data) as hp_data_json:
        hp_data = json.load(hp_data_json)

    model_cnt = 0
    for entry in hp_data:
        if entry['model']['model_name'] == model_name:
            model_cnt += 1

    with open(path_to_dataset) as dataset_json:
        dataset = json.load(dataset_json)

    dataset_size = len(dataset)

    print(f'{model_name} dataset size = {dataset_size}, Number of records about {model_name} = {model_cnt} ')


if __name__ == '__main__':
    main()
