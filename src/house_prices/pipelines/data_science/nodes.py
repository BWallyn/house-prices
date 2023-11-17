"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.18.14
"""
# =================
# ==== IMPORTS ====
# =================

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

# from .class_meanimputergroup import WithinClassMeanImputer

# ===================
# ==== FUNCTIONS ====
# ===================

class WithinClassMeanImputer(BaseEstimator, TransformerMixin):
    def __init__(self, replace_col_index, class_col_index: str=None, missing_values=np.nan):
        self.missing_values = missing_values
        self.replace_col_index = replace_col_index
        self.y = None
        self.class_col_index = class_col_index

    def fit(self, X, y = None):
        self.y = y
        return self

    def transform(self, X):
        y = self.y
        classes = np.unique(y)
        stacks = []

        if len(X) > 1 and len(self.y) == len(X):
            if( self.class_col_index == None ):
                # If we're using the dependent variable
                for aclass in classes:
                    with_missing = X[(y == aclass) & 
                                        (X[:, self.replace_col_index] == self.missing_values)]
                    without_missing = X[(y == aclass) & 
                                            (X[:, self.replace_col_index] != self.missing_values)]

                    column = without_missing[:, self.replace_col_index]
                    # Calculate mean from examples without missing values
                    mean = np.mean(column[without_missing[:, self.replace_col_index] != self.missing_values])

                    # Broadcast mean to all missing values
                    with_missing[:, self.replace_col_index] = mean

                    stacks.append(np.concatenate((with_missing, without_missing)))
            else:
                # If we're using nominal values within a binarised feature (i.e. the classes
                # are unique values within a nominal column - e.g. sex)
                for aclass in classes:
                    with_missing = X[(X[:, self.class_col_index] == aclass) & 
                                        (X[:, self.replace_col_index] == self.missing_values)]
                    without_missing = X[(X[:, self.class_col_index] == aclass) & 
                                            (X[:, self.replace_col_index] != self.missing_values)]

                    column = without_missing[:, self.replace_col_index]
                    # Calculate mean from examples without missing values
                    mean = np.mean(column[without_missing[:, self.replace_col_index] != self.missing_values])

                    # Broadcast mean to all missing values
                    with_missing[:, self.replace_col_index] = mean
                    stacks.append(np.concatenate((with_missing, without_missing)))

            if len(stacks) > 1 :
                # Reassemble our stacks of values
                X = np.concatenate(stacks)

        return X


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
            ('garage_count', SimpleImputer(strategy="constant", fill_value=0), ["GarageYrBlt", "GarageArea", "GarageCars"]),
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
#                 'garage__GarageType', 'garage__GarageFinish', 'garage__GarageQual', 'garage__GarageCond',
#                 'bsmt__BsmtQual', 'bsmt__BsmtCond', 'bsmt__BsmtExposure', 'bsmt__BsmtFinType1', 'bsmt__BsmtFinType2',
#                 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities',
#                 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
#                 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl',
#                 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
#                 'Heating', 'HeatingQC', 'CentralAir', 'KitchenQual',
#                 'FireplaceQu', 'PavedDrive', 'Fence', 'MiscFeature',
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
            ('regressor', TransformedTargetRegressor(
                regressor=HistGradientBoostingRegressor(**kwargs),
                func=np.log1p, inverse_func=np.expm1,
            )),
        ]
    ).set_output(transform="pandas")
    return estimator


def run_cross_val(estimator: Pipeline, df: pd.DataFrame, target: pd.Series, **kwargs) -> dict[np.array]:
    """Run the estimator on cross validation
    """
    scoring = ['neg_root_mean_squared_error']
    scores = cross_validate(estimator, X=df, y=target, scoring=scoring, **kwargs)
    return scores


def train_model(df_train: pd.DataFrame, params_hgb: dict) -> Pipeline:
    """
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


def predict_model(estimator: Pipeline, df: pd.DataFrame) -> pd.Series:
    """
    """
    # Predict
    pred = estimator.predict(df)
    return pred
