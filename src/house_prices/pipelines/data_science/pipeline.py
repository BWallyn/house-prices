"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.18.14
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import feature_engineering, define_search_space, find_best_hyperparameters, recreate_training,\
    train_model_mlflow, train_model, predict_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=feature_engineering,
            inputs=["split_train", "params:remove_out_train"],
            outputs="train_data",
            name='node_feat_engineering_train',
        ),
        node(
            func=feature_engineering,
            inputs=["split_valid", "params:remove_out_valid"],
            outputs="valid_data",
            name='node_feat_engineering_valid',
        ),
        node(
            func=feature_engineering,
            inputs=["test_raw_data", "params:remove_out_test"],
            outputs="test_data",
            name='node_feat_engineering_test',
        ),
        node(
            func=define_search_space,
            inputs=None,
            outputs='search_space',
            name='define_search_space'
        ),
        node(
            func=find_best_hyperparameters,
            inputs=['search_space', 'train_data', 'valid_data', 'params:max_evals', 'params:alpha'],
            outputs='best_hyperparams',
            name='node_opt_bayes',
        ),
        node(
            func=recreate_training,
            inputs=['train_data', 'valid_data'],
            outputs='training_data',
            name='merge_train_valid',
        ),
        # node(
        #     func=train_model_mlflow,
        #     inputs=["training_data", "test_raw_data", "best_hyperparams"],
        #     outputs="ml_model",
        #     name='node_train_model'
        # ),
        node(
            func=train_model,
            inputs=["training_data", "best_hyperparams"],
            outputs="ml_model",
            name='node_train_model'
        ),
        node(
            func=predict_model,
            inputs=["ml_model", "training_data"],
            outputs="pred_training",
            name='node_pred_training',
        ),
        node(
            func=predict_model,
            inputs=["ml_model", "test_data"],
            outputs="pred_test",
            name="node_pred_test",
        )
    ])
