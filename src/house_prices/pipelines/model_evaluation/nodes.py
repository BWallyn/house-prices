"""
This is a boilerplate pipeline 'model_evaluation'
generated using Kedro 0.18.14
"""
# =================
# ==== IMPORTS ====
# =================

import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error


# ===================
# ==== FUNCTIONS ====
# ===================

def evaluate_model(y_train: np.array, pred_train: np.array, y_eval: np.array, pred_eval: np.array):
    """Evaluate the model on the train and eval sets

    Args:
        y_train: Target on the train set
        pred_train: Prediction on the train set
        y_eval: Target on the evaluation set
        pred_eval: Prediction on the evaluation set
    """
    dict_metrics = {
        "rmse_train": mean_squared_error(y_train, pred_train, squared=False),
        "rmse_eval": mean_squared_error(y_eval, pred_eval, squared=False),
        "mape_train": mean_absolute_percentage_error(y_train, pred_train),
        "mape_eval": mean_absolute_percentage_error(y_eval, pred_eval),
    }
    return dict_metrics
    