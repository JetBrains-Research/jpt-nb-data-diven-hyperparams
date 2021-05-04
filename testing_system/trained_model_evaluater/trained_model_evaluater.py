import pickle
import os
from testing_system.trained_model_generator.trained_model_generator import TrainedModelWrapper
from testing_system.preproccesed_data_generator.preprocessed_data_generator import PreprocessedDataGenerator
from numbers import Number


class TrainedModelEvaluater:
    def __init__(self, kaggle_dir, path_to_ntb, eval_metric='f1_score'):
        """

        @param kaggle_dir: path to directory, that contains part of original data as
        validation data
        @param path_to_ntb: path to notebook related to the data
        @param eval_metric: evaluation metric to calculate validation score with
        """
        self.kaggle_dir = kaggle_dir
        self.kaggle_solution_dir = os.path.join(kaggle_dir, 'solution')
        self.kaggle_input_dir = os.path.join(kaggle_dir, 'input')
        self.path_to_ntb = path_to_ntb
        try:
            import sklearn.metrics as sm
            self.eval_metric = getattr(sm, eval_metric)
        except AttributeError:
            raise RuntimeError(f'Invalid sklearn evaluation metric provided: {eval_metric}')

    def evaluate_trained_model(self, trained_model: TrainedModelWrapper) -> Number:
        pdg = PreprocessedDataGenerator(self.path_to_ntb, self.kaggle_dir)
        X_test, y_true = pdg.get_preprocessed_data()
        model = pickle.loads(trained_model.pickled_model_obj)

        y_pred = model.predict(X_test)

        score = self.eval_metric(y_true, y_pred)
        return score
