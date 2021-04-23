from testing_system.dateset_splitter.dataset_splitter import KaggleEnvTabularDatasetSplitter
from testing_system.kaggle_enviroment_creator.kaggle_enviroment_creator import KaggleEnvCreator, \
    get_ntb_path_from_env, get_kaggle_dir_path_from_env
from testing_system.trained_model_evaluater.trained_model_evaluater import TrainedModelEvaluater
from testing_system.trained_model_generator.trained_model_generator import TrainedModelGenerator


def main():

    # 1) Creating environment from given notebook (pulled from Kaggle)
    #    and given directory with data (also pulled from Kaggle)

    # Note: you should delete env_dir if it is already exists in order to run script
    env_dir_path = 'env_dir'
    kg_env_cr = KaggleEnvCreator(env_dir_path=env_dir_path,
                                 path_to_ntb='/Users/Nick/Main/Diploma/HyperOptimJBR/jpt-nb-data-driven-hyperparams'
                                             '/testing_system/code_extractor/example_jpt_ntbs/100-rain-prediction'
                                             '-with-random-forest-classifier.ipynb',
                                 path_to_data_dir='/Users/Nick/did-it-rain-in-seattle-19482017',
                                 is_kaggle_competition=False
                                 )
    kg_env_cr.create_env()

    # 2) Preparing data inside created environment for training and evaluation process

    kgl_env_dataset_splitter = KaggleEnvTabularDatasetSplitter(env_dir_path,
                                                               test_size=0.3)
    kgl_env_dataset_splitter.split_data()

    # 3) Training model on "train" data part
    path_to_ntb_train = get_ntb_path_from_env(env_dir_path, train_or_val='train')
    kaggle_dir_train = get_kaggle_dir_path_from_env(env_dir_path, train_or_val='train')

    tmg = TrainedModelGenerator(path_to_ntb_train, kaggle_dir_train)
    model = tmg.get_trained_model('RandomForestClassifier', hyperparams='original')

    # 4) Evaluating model on "val" data part
    path_to_ntb_val = get_ntb_path_from_env(env_dir_path, train_or_val='val')
    kaggle_dir_val = get_kaggle_dir_path_from_env(env_dir_path, train_or_val='val')
    evaluater = TrainedModelEvaluater(kaggle_dir_val, path_to_ntb_val)
    score = evaluater.evaluate_trained_model(model)

    print(f'score = {score}')


if __name__ == '__main__':
    main()
