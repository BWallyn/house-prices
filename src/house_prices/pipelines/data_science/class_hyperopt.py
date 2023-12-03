"""Create a hyperopt class to find the best hyperparameters"""
# =================
# ==== IMPORTS ====
# =================

# Essential
import numpy as np
import pandas as pd

from typing import Any, Optional

# Machine Learning
from hyperopt import fmin, tpe, Trials
import mlflow
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error


# ===============
# ==== Class ====
# ===============

class hyperopt_mlflow():
    """Class to run bayesian optimization to find the best hyperparameters
    """
    def __init__(
        self, run_name: str, max_evals: int,
        alpha: float=0., verbose: Optional[int]=None
    ):
        """
        Args:
            run_name: Name of the run for MLflow
            max_evals: Number of maximum trials for the bayesian optimization search
            feat_cat: Names of the categorical features
            alpha: Penalization coefficient to add to the optimization metric (delta between train and eval)
            monotonic_cst: Monotonic constraints to add to the features
            verbose: Verbose of the bayesian optimization
        """
        self.run_name = run_name
        self.max_evals = max_evals
        self.alpha = alpha
        self.verbose = verbose
        # Initialize empty trails
        self.trials = Trials()
    

    def objective(
        self, params: dict[str, Any], estimator: HistGradientBoostingRegressor,
        df_train: pd.DataFrame, y_train: np.array, df_eval: pd.DataFrame, y_eval: np.array
    ) -> float:
        """Train a HistGradientBoostingRegressor with its hyperparameters values. Log the model and its metrics to MLflow.

        Args:
            params: Hyperparameters of the HistGradientBoostingRegressor
            estimator: Machine learning estimator
            df_train: Train dataset
            y_train: Target of the train dataset
            df_eval: Evaluation dataset
            y_eval: Target of the evaluation dataset
        Returns:
            score_optim: Optimization score
        """
        # Start a nested MLflow run for logging
        with mlflow.start_run(nested=True) as run:
            # Set the hyperparameters in the estimator
            params['model__regressor__loss'] = 'squared_error'
            params['model__regressor__max_iter'] = 1000
            params['model__regressor__random_state'] = 42
            estimator = estimator.set_params(**params)

            # Predict
            estimator.fit(df_train, y_train)
            y_pred_train = estimator.predict(df_train)
            y_pred_eval = estimator.predict(df_eval)

            # Score metrics
            score_train = mean_squared_error(y_true=y_train, y_pred=y_pred_train, squared=False)
            score_eval = mean_squared_error(y_true=y_eval, y_pred=y_pred_eval, squared=False)
            # Define the score to minimize
            score_optim = score_eval + self.alpha * (score_eval - score_train)
            dict_metrics = {
                'rmse_train': score_train, 'rmse_eval': score_eval, 'n_iter': estimator['model'].regressor_.n_iter_,
                'score_optim': score_optim,
            }

            # Log the metrics
            mlflow.log_metrics(dict_metrics)

            # Log params of the model
            dict_params = estimator['model'].regressor_.get_params()
            dict_params.pop('categorical_features')
            mlflow.log_params(dict_params)

            # Log the fitted model
            mlflow.sklearn.log_model(estimator, "model", input_example=df_train.head())

        # Return the optimization score
        return score_optim
    

    def fit(
        self, estimator: HistGradientBoostingRegressor, search_space: dict[str, Any],
        df_train: pd.DataFrame, y_train: np.array, df_eval: pd.DataFrame, y_eval: np.array
    ) -> None:
        """Run the bayesian optimization

        Args:
            estimator: Estimator, machine learning estimator
            search_space: Search space of the hyperparameters for the bayesian optimization
            df_train: Training dataset
            y_train: Target of the train dataset
            df_eval: Evaluation dataset
            y_eval: Target of the evaluation dataset
        """
        with mlflow.start_run():
            best_params = fmin(
                fn=lambda params: self.objective(params, estimator, df_train, y_train, df_eval, y_eval),
                space=search_space,
                algo=tpe.suggest,
                max_evals=self.max_evals,
                trials=self.trials,
                rstate=np.random.default_rng(42),
                verbose=self.verbose
            )
        self.best_params = best_params

    
    def get_best_params(self) -> dict[str, Any]:
        """Get the best hyperparameters
        """
        return self.best_params
