"""Classifiers compared under one split and one preprocessor."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from churn.config import RANDOM_STATE
from churn.preprocess import build_preprocessor


def positive_class_weight(y_train) -> float:
    """XGBoost scale_pos_weight from the training fold only (negative / positive)."""
    positives = int((y_train == 1).sum())
    negatives = int((y_train == 0).sum())
    if positives == 0:
        raise ValueError("Training labels contain no churn cases.")
    return negatives / positives


def build_classifiers(scale_pos_weight: float) -> dict:
    """Return unfitted classifiers.

    Logistic regression and the forest use class_weight so the minority churn
    class influences the loss. XGBoost gets the equivalent via scale_pos_weight.
    Hyperparameters are fixed, modest settings for a 7k-row table — not the
    result of a search on the holdout.
    """
    return {
        "logistic_regression": LogisticRegression(
            C=1.0,
            l1_ratio=0.0,
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            min_child_weight=5,
            reg_lambda=1.0,
            scale_pos_weight=scale_pos_weight,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            tree_method="hist",
        ),
    }


def make_pipeline(classifier) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            ("classifier", classifier),
        ]
    )
