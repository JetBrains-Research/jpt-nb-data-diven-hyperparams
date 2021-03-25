from itertools import tee, zip_longest

import pandas as pd
from tqdm import tqdm

# from dataset_creator.util.const import RESULT_DIR
RESULT_DIR = 'results'
from sklearn_hp_analisys.util.const import PATH_TO_DATA, EXTRACTED_DATA_DIR
from sklearn_hp_analisys.util.file_util import *


class Code2HpDatasetCreator:
    def __init__(self, original_dataset_fname: str, hp_data_fname: str, ml_model_name: str):
        self.ml_model_name = ml_model_name
        self.hp_data_fname = hp_data_fname
        self.raw_data_fname = original_dataset_fname

    def create_dataset(self, chunksize=1e5):

        dataset = []
        # num_records_in_raw_data = self.__raw_data_size()
        ntb_id_to_params = self.__make_ntb_id_to_params_dict()
        remaining_ntb_ids = set(ntb_id_to_params.keys())

        for curr_chunk, next_chunk in tqdm(self.__pairwise(pd.read_csv(self.raw_data_fname, chunksize=chunksize))):
            curr_chunk_ntb_ids = set(curr_chunk['notebook_id'].unique())
            relevant_ntb_ids = curr_chunk_ntb_ids.intersection(remaining_ntb_ids)

            chunk = pd.concat([curr_chunk, next_chunk], axis=0)
            for ntb_id in relevant_ntb_ids:
                sorted_notebook_data = chunk[chunk['notebook_id'] == ntb_id].sort_values(by='index')
                top_idx, models_params = self.__get_ntb_top_idxs_and_params(ntb_id_to_params, ntb_id)
                sample = self.__prepare_sample(sorted_notebook_data, top_idx)
                for model_params in models_params:
                    dataset.append({'sample': sample, 'target': model_params})
            remaining_ntb_ids -= relevant_ntb_ids
        return dataset

    def __make_ntb_id_to_params_dict(self):
        with open(self.hp_data_fname) as hp_data_json:
            data = json.load(hp_data_json)

        result = {}
        for entry in data:
            ntb_id = entry['location']['notebook_id']
            cell_idx = entry['location']['cell_idx']
            try:
                result[ntb_id].append(entry['model'])
            except KeyError:
                result[ntb_id] = [entry['model']]
            entry['model']['cell_idx'] = cell_idx
        return result

    @staticmethod
    def __prepare_sample(sorted_notebook_data, top_idx):
        sample = []
        for _, row in sorted_notebook_data.iterrows():
            if row['index'] >= top_idx:
                break
            cell_data = dict(row)
            del cell_data['Unnamed: 0'], cell_data['id'], cell_data['repository_id']
            sample.append(cell_data)
        return sample

    def __get_ntb_top_idxs_and_params(self, ntb_id_to_params_dict, ntb_id):
        ntb_models_params = ntb_id_to_params_dict[ntb_id]
        top_model = min(ntb_models_params, key=lambda model: model['cell_idx'])
        top_idx = top_model['cell_idx']

        result = []
        for record in ntb_models_params:
            if record['model_name'] == self.ml_model_name:
                record_copy = record.copy()
                del record_copy['cell_idx']
                result.append(record_copy)
        return top_idx, result

    def __raw_data_size(self, chunksize=1e5):
        return sum(len(chunk) for chunk in tqdm(pd.read_csv(self.raw_data_fname, chunksize=chunksize)))

    @staticmethod
    def __pairwise(iterable):
        """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
        first, second = tee(iterable)
        next(second, None)
        return zip_longest(first, second)


def main():
    dataset_creator = Code2HpDatasetCreator(original_dataset_fname=PATH_TO_DATA,
                                            hp_data_fname=os.path.join(EXTRACTED_DATA_DIR, 'dataset_results_v3.json'),
                                            ml_model_name='RandomForestClassifier')
    dataset = dataset_creator.create_dataset()
    write_json_to_file(dataset,
                       os.path.join(RESULT_DIR, 'created_dataset_v2.json'))


if __name__ == '__main__':
    main()
