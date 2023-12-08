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

def evaluate_model(y_true: np.array, y_pred: np.array):
    """Evaluate the model on a dataset

    Args:
        y_true: Target on the train set
        y_pred: Prediction on the train set
    """
    dict_metrics = {
        "rmse": mean_squared_error(y_true, y_pred, squared=False),
        "mape": mean_absolute_percentage_error(y_true, y_pred),
    }
    print("Metrics model:\n", dict_metrics)
    return dict_metrics


def evaluate_model_from_df(df: pd.DataFrame, target_name: str, y_pred: np.array):
    """
    """
    return evaluate_model(y_true=df[target_name].values, y_pred=y_pred)