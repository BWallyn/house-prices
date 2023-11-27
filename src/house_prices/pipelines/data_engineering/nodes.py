"""
This is a boilerplate pipeline 'data_engineering'
generated using Kedro 0.18.14
"""
# ================
# ==== IMPORT ====
# ================

import pandas as pd
from sklearn.model_selection import train_test_split


# ===================
# ==== FUNCTIONS ====
# ===================

def split_training(df: pd.DataFrame, target_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the training set into train and validation

    Args:
        df: Training dataset
        target_name: name of the target column
    Returns:
        df_train: train dataset
        df_valid: validation dataset
    """
    y_training = df[target_name]
    df.drop(columns=[target_name], inplace=True)
    df_train, df_valid, y_train, y_valid = train_test_split(
        df, y_training, test_size=0.25, random_state=42
    )
    df_train[target_name] = y_train
    df_valid[target_name] = y_valid
    return df_train, df_valid

