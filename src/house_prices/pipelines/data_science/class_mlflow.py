""""Class mlflow run"""
# =================
# ==== IMPORTS ====
# =================

# Essential
import numpy as np
import pandas as pd
from typing import Optional

# Machine Learning
import mlflow
import shap
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# Plots
import matplotlib.pyplot as plt


# ===================
# ==== FUNCTIONS ====
# ===================

class mlflow_run():
    """
    """
    def __init__(
        self, model_name: str, feat: list[str], run_name: Optional[str]=None,
        shap: bool=False, tags: Optional[dict]=None
    ):
        """Initialize the mlflow run

        Args:
            model_name: Name of the model that will be used for the file names
            feat: List of the features
            run_name: Name of the HistGradientBoosting run
            shap: Whether to compute and plot the SHAP values
            tags: Tags to add to the run for MLflow
        """
        self.model_name = model_name
        self.feat = feat
        self.run_name = run_name
        self.shap = shap
        self.tags = tags
    

    def compute_metrics(self, y_train: np.array, pred_train: np.array, y_eval: np.array, pred_eval: np.array) -> dict[str, float]:
        """Compute the metrics on the train and evaluation datasets

        Args:
            y_train: Target on the train set
            pred_train: Prediction on the train set
            y_eval: Target on the evaluation set
            pred_eval: Prediction on the evaluation set
        Returns:
            : metrics on the train and evaluation set
        """
        return {
            'rmse_train': mean_squared_error(y_true=y_train, y_pred=pred_train, squared=False),
            'rmse_eval': mean_squared_error(y_true=y_eval, y_pred=pred_eval, squared=False),
        }
    

    def _compute_shap_values(self, df: pd.DataFrame) -> None:
        """Compute the shap values on a HistgradientBoostingRegressor model using the TreeExplainer method.

        Args:
            df: DataFrame
        """
        explainer = shap.TreeExplainer(self.model)
        self.shap_values = explainer.shap_values(df)


    def _plot_shap_values(self, df: pd.DataFrame) -> None:
        """Plot the SHAP values and save it.

        Args:
            df: DataFrame
        """
        shap.summary_plot(self.shap_values, df, show=False)
        plt.savefig(f'reports/{self.model_name}_shap_beeswarm.png')


    def _store_dataframe_shap(self) -> None:
        """Store the average SHAP values of each feature in a DataFrame.

        Args:
            cols: List of the columns of the dataframe.
        """
        df_shap_val = pd.DataFrame(self.shap_values)
        df_shap_val.columns = self.feat
        imp_shap = pd.DataFrame(df_shap_val.abs().mean().sort_values(ascending=False))
        imp_shap.reset_index(inplace=True)
        imp_shap.columns = ["Features", "avg_SHAP"]
        self.imp_shap = imp_shap
        imp_shap.to_csv(f"reports/{self.model_name}_feat_importance_shap.csv")


    def _log_to_mlflow(self) -> None:
        """Log parameters, model and artifacts to MLflow
        """
        dict_params = self.model['model'].regressor_.get_params()
        dict_params.pop('categorical_features')
        mlflow.log_params(dict_params)
        mlflow.sklearn.log_model(self.model, 'HistGradientBoosting')
        mlflow.log_metrics(self.dict_metrics)
        if self.shap:
            mlflow.log_artifact()


    def fit(
        self, model: HistGradientBoostingRegressor,
        df_train: pd.DataFrame, y_train: np.array, df_eval: pd.DataFrame, y_eval: np.array
    ) -> None:
        """Fit a HistGradientBoosting regressor model and use MLflow to log parameters, model and metrics.

        Args:
            model: HistGradientBoostingRegressor model
            df_train: Train set
            y_train: Target of the train set
            df_eval: Evaluation set
            y_eval: Target of the evaluation set
        """
        self.model = model

        # Run mlflow
        with mlflow.start_run(tags=self.tags, run_name=self.run_name) as run:
            # Model
            print("\n*** Fit model ***")
            self.model.fit(df_train, y_train)

            # Predict
            pred_train = self.model.predict(df_train)
            pred_eval = self.model.predict(df_eval)

            # Metrics
            print("\n*** Metrics ***")
            self.dict_metrics = self.compute_metrics(y_train=y_train, pred_train=pred_train, y_eval=y_eval, pred_eval=pred_eval)
            print(self.dict_metrics)
            self.dict_metrics['n_iter'] = self.model['model'].regressor_.n_iter_

            # SHAP values
            if self.shap:
                print("\n*** Compute SHAP values ***")
                self._compute_shap_values(df_train)
                self._plot_shap_values(df_train)
                self._store_dataframe_shap(cols=df_train.columns.tolist())
            
            # Log parameters to MLflow
            self._log_to_mlflow()