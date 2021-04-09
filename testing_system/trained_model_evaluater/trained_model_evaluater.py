import pickle
import os
from testing_system.trained_model_generator.trained_model_generator import TrainedModelWrapper


class TrainedModelEvaluater:
    def __init__(self, kaggle_dir, eval_metric='f1_score'):
        self.kaggle_dir = kaggle_dir
        self.kaggle_input_dir = os.path.join(kaggle_dir, 'input')
        try:
            import sklearn.metrics as sm
            eval_metric = getattr(sm, eval_metric)
        except AttributeError:
            raise RuntimeError(f'Invalid sklearn evaluation metric provided: {eval_metric}')

    def evaluate_trained_model(self, trained_model: TrainedModelWrapper):
        pass
