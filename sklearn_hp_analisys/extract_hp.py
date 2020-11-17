import logging
import os

import pandas as pd

from sklearn_hp_analisys.hyperparams_extractor.hyperparams_extractor import HyperparamsExtractor as HpExtractor, write_results_to_file
from sklearn_hp_analisys.util.const import PATH_TO_DATA, LOG_DIR, EXTRACTED_PARAMS_FILENAME, EXTRACTED_DATA_DIR

LOG_FILE_NAME = 'log_extract.txt'
logging.basicConfig(filename=os.path.join(LOG_DIR, LOG_FILE_NAME), level=logging.INFO)


class HyperparamsLocation:
    def __init__(self, df_record):
        self.repo_id = df_record.loc['repository_id']
        self.notebook_id = df_record.loc['notebook_id']
        self.cell_idx = df_record.loc['index']
        self.df_row_idx = df_record.loc['id']

    def as_dict(self):
        return self.__dict__


# def extract_code_from_record(record):
#     if record['cell_type'] == 'code':
#         return record['source']
#     return None  # No python code


def extract_hp_from_dataset(path_to_data=PATH_TO_DATA, chunksize=1e5):
    result = []
    skipped_records_cnt = 0
    NUM_RECORDS_IN_DATASET = 10923813
    for chunk in pd.read_csv(path_to_data, chunksize=chunksize):
        chunk = chunk[chunk['source'].notna()]
        chunk = chunk[chunk['cell_type'] == 'code']
        for idx, row in chunk.iterrows():
            python_src = row['source']
            hp_location = HyperparamsLocation(row)
            if python_src is not None:
                try:
                    hp_extractor = HpExtractor(src=python_src)
                    extracted_hps = hp_extractor.get_hyperparams()
                    extracted_hps_with_locations = [{'model': model,
                                                     'location': hp_location.as_dict()}
                                                    for model in extracted_hps]
                    result += extracted_hps_with_locations
                    print(f'STATUS:Records skipped: {skipped_records_cnt * 100./ (idx + 1) :.2f}%\t'
                          f'Number Records processed: {(idx + 1) * 100. / NUM_RECORDS_IN_DATASET :.2f}%')

                except SyntaxError as syn_err:
                    logging.info(f'Invalid python3 code in cell (record no.{idx}). Error: {syn_err}')
                    skipped_records_cnt += 1

                except RecursionError as rec_err:
                    logging.info(f'{rec_err.__class__.__name__} in code in cell (record no.{idx}). '
                                  f'Error: {str(rec_err)}\n'
                                  f'Code from cell:\n'
                                  f'{python_src}')

                    skipped_records_cnt += 1

                except Exception as e:
                    logging.error(f'record no. {idx}: {str(e)}\n'
                                  f'Code from cell:\n'
                                  f'{python_src}')

                    skipped_records_cnt += 1
                    continue

    return result


if __name__ == '__main__':
    hp_list = extract_hp_from_dataset(chunksize=1e6)
    write_results_to_file(hp_list,
                          os.path.join(EXTRACTED_DATA_DIR, 'dataset_results_v2.json'))
