"""
This is a boilerplate pipeline 'data_engineering'
generated using Kedro 0.18.14
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import split_training


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=split_training,
            inputs=['train_raw_data', 'params:target_name'],
            outputs=['split_train', 'split_valid'],
            name='node_split_train_valid'
        ),
    ])
