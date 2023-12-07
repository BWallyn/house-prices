"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.18.14
"""
# =================
# ==== IMPORTS ====
# =================

# Essential
import numpy as np
import pandas as pd

from typing import Any

# Machine Learning
from hyperopt import hp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

# from .class_meanimputergroup import WithinClassMeanImputer
from .class_hyperopt import hyperopt_mlflow
from .class_mlflow import mlflow_run


# ===================
# ==== FUNCTIONS ====
# ===================

def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Remove outliers

    Args:
        df: DataFrame with outliers
    Returns:
        df: DataFrame without outliers
    """
    df.drop(df[(df['SalePrice'] > 700000) & (df['OverallQual'] < 5)].index, inplace=True)
    df.drop(df[(df['GrLivArea'] > 4500) & (df["SalePrice"] < 300000)].index, inplace=True)
    return df


def create_feats(df: pd.DataFrame) -> pd.DataFrame:
    """Create new features based on other features

    Args:
        df: DataFrame
    Returns:
        df: DataFrame with new features
    """
    # Is present
    df["BsmtFinType1_Unf"] = 1 * (df["BsmtFinType1"] == "Unf")
    df["HasWoodDeck"] = 1 * (df["WoodDeckSF"] == 0)
    df["HasOpenPorch"] = 1 * (df["OpenPorchSF"] == 0)
    df["HasEnclosedPorch"] = 1 * (df["EnclosedPorch"] == 0)
    df["Has3SsnPorch"] = 1 * (df["3SsnPorch"] == 0)
    df["HasScreenPorch"] = 1 * (df["ScreenPorch"] == 0)
    df["haspool"] = df["PoolArea"].apply(lambda x: 1 if x > 0 else 0)
    df["has2ndfloor"] = df["2ndFlrSF"].apply(lambda x: 1 if x > 0 else 0)
    df["hasgarage"] = df["GarageArea"].apply(lambda x: 1 if x > 0 else 0)
    df["hasbsmt"] = df["TotalBsmtSF"].apply(lambda x: 1 if x > 0 else 0)
    df["hasfireplace"] = df["Fireplaces"].apply(lambda x: 1 if x > 0 else 0)
    # Compute years
    df["YearsSinceRemodel"] = df["YrSold"].astype(int) - df["YearRemodAdd"].astype(int)
    df["YrBltAndRemod"] = df["YearBuilt"] + df["YearRemodAdd"]
    # Compute total
    df["Total_Home_Quality"] = df["OverallQual"] + df["OverallCond"]
    df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
    df["Total_sqr_footage"] = df["BsmtFinSF1"] + df["BsmtFinSF2"] + df["1stFlrSF"] + df["2ndFlrSF"]
    df["Total_Bathrooms"] = df["FullBath"] + 0.5 * df["HalfBath"] + df["BsmtFullBath"] + 0.5 * df["BsmtHalfBath"]
    df["Total_porch_sf"] = df["OpenPorchSF"] + df["3SsnPorch"] + df["EnclosedPorch"] + df["ScreenPorch"] + df["WoodDeckSF"]
    df["TotalBsmtSF"] = df["TotalBsmtSF"].apply(lambda x: np.exp(6) if x <= 0.0 else x)
    return df


def feature_engineering(df: pd.DataFrame, remove_out: bool=False) -> pd.DataFrame:
    """Remove outliers and create new features

    Args:
        df: DataFrame
        remove_out: Whether to remove outliers
    Returns:
        df: DataFrame without outliers and with new features
    """
    if remove_out:
        df = df.pipe(remove_outliers)
    df = df.pipe(create_feats)
    return df


def feature_imputer():
    """Create a feature imputer for the missing values

    Returns:
        feat_imp: Feature imputer element from sklearn
    """
    feat_imp = ColumnTransformer(
        [
            ('functional', SimpleImputer(strategy="constant", fill_value="Typ"), ["Functional"]),
            ('mode', SimpleImputer(strategy='most_frequent'), ["Electrical", "KitchenQual", "Exterior1st", "Exterior2nd", "SaleType", "MSZoning"]),
            ('pool', SimpleImputer(strategy="constant", fill_value='No'), ["PoolQC"]),
            ('grg_count', SimpleImputer(strategy="constant", fill_value=0), ["GarageYrBlt", "GarageArea", "GarageCars"]),
            ('garage', SimpleImputer(strategy='constant', fill_value='No'), ["GarageType", "GarageFinish", "GarageQual", "GarageCond"]),
            ('bsmt', SimpleImputer(strategy='constant', fill_value='No'), ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']),
            ('lot_front', SimpleImputer(strategy="mean"), ["LotFrontage"]),
            ('other_cat', SimpleImputer(strategy='constant', fill_value='No'), ["MiscFeature", "Alley", "Fence", "MasVnrType", "FireplaceQu"]),
            ('other_cont', SimpleImputer(strategy="mean"), ["MasVnrArea"]),
        ], remainder='passthrough', verbose_feature_names_out=False
    )
    # ('lot_front', WithinClassMeanImputer(replace_col_index="LotFrontage", class_col_index="Neighborhood")),
    return feat_imp


# def column_transformer():
#     """Create a column transformer
#     """
#     col_transf = ColumnTransformer(
#         [
#             ('ordinal_enc', OrdinalEncoder(handle_unknown='use_encoded_value'), [
#                 'functional__Functional',
#                 'mode__Electrical', 'mode__KitchenQual', 'mode__Exterior1st', 'mode__Exterior2nd', 'mode__SaleType', 'mode__MSZoning',
#                 'pool__PoolQC',
#                 'grg_count__GarageYrBlt', 'grg_count__GarageArea', 'grg_count__GarageCars',
#                 'garage__GarageType', 'garage__GarageFinish', 'garage__GarageQual', 'garage__GarageCond',
#                 'bsmt__BsmtQual', 'bsmt__BsmtCond', 'bsmt__BsmtExposure', 'bsmt__BsmtFinType1', 'bsmt__BsmtFinType2',
#                 'lot_front__LotFrontage',
#                 'other_cat__MiscFeature', 'other_cat__Alley', 'other_cat__Fence', 'other_cat__MasVnrType', 'other_cat__FireplaceQu',
#                 'other_cont__MasVnrArea',
#                 'Street', 'LotShape', 'LandContour', 'Utilities',
#                 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
#                 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl',
#                 'ExterQual', 'ExterCond', 'Foundation',
#                 'Heating', 'HeatingQC', 'CentralAir', 'KitchenQual',
#                 'PavedDrive',
#                 'SaleCondition'
#             ])
#         ], remainder='passthrough',
#     )
#     return col_transf

def column_transformer():
    """Create a column transformer

    Returns:
        col_transf: Column transformer element from sklearn
    """
    col_transf = ColumnTransformer(
        [
            ('ordinal_enc', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan), [
                'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities',
                'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
                'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
                'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
                'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
                'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',
                'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature',
                'SaleType', 'SaleCondition'
            ])
        ], remainder='passthrough', verbose_feature_names_out=False,
    )
    return col_transf


def pipe_estimator(feat_imp: ColumnTransformer, col_transf: ColumnTransformer, **kwargs):
    """Create a regressor estimator from sklearn using a pipeline

    Args:
        feat_imp: Feature imputer element
        col_transf: Column transformer element
        **kwargs: Hyperparameters for the HistGradientBoosting method
    Returns:
        estimator: Estimator element from sklearn
    """
    estimator = Pipeline(
        steps=[
            ('feature_imp', feat_imp),
            ('column_transf', col_transf),
            ('model', TransformedTargetRegressor(
                regressor=HistGradientBoostingRegressor(**kwargs),
                func=np.log1p, inverse_func=np.expm1,
            )),
        ]
    ).set_output(transform="pandas")
    return estimator


def run_cross_val(estimator: Pipeline, df: pd.DataFrame, target: pd.Series, **kwargs) -> dict[np.array]:
    """Run the estimator on cross validation

    Args:
        estimator: Estimator regressor
        df: DataFrame
        target: Target of the dataframe
    Returns:
        scores: Scores of the cross validation
    """
    scoring = ['neg_root_mean_squared_error']
    scores = cross_validate(estimator, X=df, y=target, scoring=scoring, **kwargs)
    return scores


def define_search_space() -> dict[str, Any]:
    """Define the search space for the bayesian optimization

    Returns:
        : Search space
    """
    return {
        'model__regressor__learning_rate': hp.loguniform('learning_rate', -5, 0),
        'model__regressor__max_depth': hp.randint('max_depth', 2, 10),
        'model__regressor__l2_regularization': hp.loguniform('l2_regularization', -3, 0),
    }


def find_best_hyperparameters(
        search_space: dict[str, Any], df_train: pd.DataFrame, df_valid: pd.DataFrame,
        max_evals: int=10, alpha: float=0., 
    ) -> dict[str, Any]:
    """Find the best hyperparameters using bayesian optimization

    Args:
        search_space: Search space of the hyperparameters for the bayesian optimization
        df_train: Train dataset
        df_valid: Validation dataset
        max_evals: Number of evaluations for the bayesian optimization
        alpha: Coefficient for the penalization
    Returns:
        best_hyperparameters: Best hyperparameters found by the bayesian optimization
    """
    estimator = pipe_estimator(feat_imp=feature_imputer(), col_transf=column_transformer())
    # Prepare datasets
    y_train = df_train["SalePrice"]
    df_train.drop(columns=["SalePrice"], inplace=True)
    y_valid = df_valid["SalePrice"]
    df_valid.drop(columns=["SalePrice"], inplace=True)
    # Categorical features
    feat_cat = df_train.select_dtypes(exclude=['int16', 'int32', 'int64', 'float16', 'float32', 'float64']).columns.tolist()
    feat_cat += [
        "BsmtFinType1_Unf", "HasWoodDeck", "HasOpenPorch", "HasEnclosedPorch", "Has3SsnPorch", "HasScreenPorch",
        "haspool", "has2ndfloor", "hasgarage", "hasbsmt", "hasfireplace",
    ]
    # Bayesian optimization
    bay_opt = hyperopt_mlflow(run_name='opt_bayesian', max_evals=max_evals, alpha=alpha, verbose=0)
    bay_opt.fit(estimator, search_space=search_space, df_train=df_train, y_train=y_train, df_eval=df_valid, y_eval=y_valid)
    best_hyperparams = bay_opt.get_best_params()
    return best_hyperparams


def recreate_training(df_train: pd.DataFrame, df_valid: pd.DataFrame) -> pd.DataFrame:
    """Recreate the training dataset.
    Merge train and validation into one dataset to create the training dataset.

    Args:
        df_train: Train dataset
        df_valid: Validation dataset
    Returns:
        df_training: Training dataset
    """
    df_training = pd.concat([df_train, df_valid])
    return df_training


def train_model_mlflow(df_train: pd.DataFrame, df_test: pd.DataFrame, params_hgb: dict) -> Pipeline:
    """Train model using mlflow

    Args:
        df_train: Train set
        df_test: Test set
        params_hgb: Parameters of the HistGradientBoostingRegressor
    Returns:
        : Model trained on train set
    """
    estimator = pipe_estimator(feat_imp=feature_imputer(), col_transf=column_transformer(), **params_hgb)
    # Prepare dataset
    y_train = df_train["SalePrice"]
    df_train.drop(columns=["SalePrice"], inplace=True)
    y_test = df_test["SalePrice"]
    df_test.drop(["SalePrice"], inplace=True)
    # Train estimator using MLfloe
    est_mlflow = mlflow_run(model_name='house-prices', feat=df_train.columns, shap=False)
    est_mlflow.fit(
        model=estimator, df_train=df_train, y_train=y_train, df_eval=df_test, y_eval=y_test
    )
    return est_mlflow.model


def train_model(df_train: pd.DataFrame, params_hgb: dict) -> Pipeline:
    """Train the model

    Args:
        df_train: Train dataset
        params_hgb: Parameters for the HistGradientBoosting
    Returns:
        estimator: HistGradientBoosting trained
    """
    # Prepare data
    y_train = df_train["SalePrice"]
    df_train.drop(columns=["SalePrice"], inplace=True)
    # Create pipeline
    feat_imp = feature_imputer()
    col_transf = column_transformer()
    estimator = pipe_estimator(feat_imp=feat_imp, col_transf=col_transf, **params_hgb)
    # Train model
    estimator.fit(df_train, y_train)
    return estimator


def predict_model(estimator: Pipeline, df: pd.DataFrame) -> pd.DataFrame:
    """Predict using the estimator

    Args:
        estimator: HistGradientBoosting regressor trained
        df: DataFrame
    Returns:
        pred: Prediction on the dataset
    """
    # Get input features
    list_inputs = estimator.feature_names_in_
    # Predict
    pred = estimator.predict(df[list_inputs])
    df_pred = pd.DataFrame(data={'Id': df["Id"].values, 'SalePrice': pred})
    return df_pred
