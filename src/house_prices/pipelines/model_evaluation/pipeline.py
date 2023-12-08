"""
This is a boilerplate pipeline 'model_evaluation'
generated using Kedro 0.18.14
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import evaluate_model_from_df


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=evaluate_model_from_df,
            inputs=["training_data", "params:target_name", "pred_training"],
            outputs="metrics_training",
            name="node_metric_training",
        ),
    ])
