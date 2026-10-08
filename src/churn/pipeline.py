"""Train three classifiers and write the holdout artifacts."""

from __future__ import annotations

import json
import logging
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from churn.config import (
    ARTIFACTS_DIR,
    DATA_PATH,
    DISPLAY_NAMES,
    N_SPLITS,
    RANDOM_STATE,
    TEST_SIZE,
)
from churn.data import load_raw
from churn.evaluate import (
    best_f1_threshold,
    binary_metrics,
    extract_importance,
    majority_baseline,
    outreach_comparison,
    plot_confusion_matrix,
    plot_importance,
    plot_metric_bars,
    plot_pr_curve,
    plot_roc_curves,
    positive_scores,
)
from churn.features import prepare_features
from churn.models import build_classifiers, make_pipeline, positive_class_weight

logger = logging.getLogger(__name__)

PACKAGE_VERSIONS = ("numpy", "pandas", "scikit-learn", "xgboost", "matplotlib", "joblib")


def _round_metrics(metrics: dict) -> dict:
    rounded = {}
    for key, value in metrics.items():
        if isinstance(value, float):
            rounded[key] = round(value, 6)
        else:
            rounded[key] = value
    return rounded


def train_and_evaluate(
    data_path: Path | None = None,
    artifacts_dir: Path | None = None,
) -> dict:
    """Fit, compare, and save the churn models.

    The stratified holdout is cut before any preprocessor is fit. Model choice
    uses out-of-fold ROC-AUC on the training split only. The F1 threshold is
    chosen on those same out-of-fold scores. The test set is scored once, after
    both decisions.
    """
    data_path = Path(data_path) if data_path else DATA_PATH
    artifacts_dir = Path(artifacts_dir) if artifacts_dir else ARTIFACTS_DIR
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    raw = load_raw(data_path)
    features, target = prepare_features(raw)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        stratify=target,
        random_state=RANDOM_STATE,
    )

    scale_pos_weight = positive_class_weight(y_train)
    classifiers = build_classifiers(scale_pos_weight)
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    logger.info(
        "Split %s train / %s test rows (churn rates %.3f / %.3f). scale_pos_weight=%.3f",
        len(x_train),
        len(x_test),
        float(y_train.mean()),
        float(y_test.mean()),
        scale_pos_weight,
    )

    report: dict = {
        "dataset": {
            "rows": int(len(features)),
            "positive_class": "Churn=Yes",
            "churn_rate": float(target.mean()),
            "source": "IBM telco-customer-churn-on-icp4d sample",
        },
        "split": {
            "test_size": TEST_SIZE,
            "stratified": True,
            "random_state": RANDOM_STATE,
            "n_train": int(len(x_train)),
            "n_test": int(len(x_test)),
            "train_churn_rate": float(y_train.mean()),
            "test_churn_rate": float(y_test.mean()),
        },
        "cv": {"n_splits": N_SPLITS, "shuffle": True, "random_state": RANDOM_STATE},
        "imbalance": {
            "logistic_regression": "class_weight=balanced",
            "random_forest": "class_weight=balanced_subsample",
            "xgboost": f"scale_pos_weight={scale_pos_weight:.6f} from training labels only",
        },
        "selection": {
            "rule": (
                "Highest ROC-AUC on concatenated out-of-fold training predictions. "
                "Ties break on out-of-fold F1 at threshold 0.50."
            ),
            "threshold_rule": "F1-maximizing threshold on out-of-fold training scores, applied unchanged to the test set.",
        },
        "versions": {name: version(name) for name in PACKAGE_VERSIONS},
        "models": {},
    }

    for name, classifier in classifiers.items():
        oof = np.zeros(len(y_train), dtype=float)
        fold_aucs = []
        for train_idx, valid_idx in cv.split(x_train, y_train):
            pipeline = make_pipeline(clone(classifier))
            pipeline.fit(x_train.iloc[train_idx], y_train.iloc[train_idx])
            fold_scores = positive_scores(pipeline, x_train.iloc[valid_idx])
            oof[valid_idx] = fold_scores
            fold_aucs.append(float(roc_auc_score(y_train.iloc[valid_idx], fold_scores)))
        threshold = best_f1_threshold(y_train, oof)
        report["models"][name] = {
            "display_name": DISPLAY_NAMES[name],
            "f1_threshold": threshold,
            "oof_roc_auc_folds": [round(score, 6) for score in fold_aucs],
            "oof_roc_auc_std": round(float(np.std(fold_aucs, ddof=1)), 6),
            "oof_at_0_5": _round_metrics(binary_metrics(y_train, oof, 0.5)),
            "oof_at_f1_threshold": _round_metrics(binary_metrics(y_train, oof, threshold)),
        }
        logger.info(
            "%s OOF ROC-AUC %.4f | F1@0.50 %.4f | F1 threshold %.2f",
            name,
            report["models"][name]["oof_at_0_5"]["roc_auc"],
            report["models"][name]["oof_at_0_5"]["f1"],
            threshold,
        )

    best_name = max(
        report["models"],
        key=lambda name: (
            report["models"][name]["oof_at_0_5"]["roc_auc"],
            report["models"][name]["oof_at_0_5"]["f1"],
        ),
    )
    report["best_model"] = best_name

    # Second set of estimators for the full-training fit. Fold models were clones.
    fitted = {}
    test_scores = {}
    final_classifiers = build_classifiers(scale_pos_weight)
    for name, classifier in final_classifiers.items():
        pipeline = make_pipeline(classifier)
        pipeline.fit(x_train, y_train)
        scores = positive_scores(pipeline, x_test)
        fitted[name] = pipeline
        test_scores[name] = scores
        threshold = report["models"][name]["f1_threshold"]
        report["models"][name]["test_at_0_5"] = _round_metrics(binary_metrics(y_test, scores, 0.5))
        report["models"][name]["test_at_f1_threshold"] = _round_metrics(
            binary_metrics(y_test, scores, threshold)
        )
        logger.info(
            "%s TEST ROC-AUC %.4f | F1@0.50 %.4f | F1@%.2f %.4f",
            name,
            report["models"][name]["test_at_0_5"]["roc_auc"],
            report["models"][name]["test_at_0_5"]["f1"],
            threshold,
            report["models"][name]["test_at_f1_threshold"]["f1"],
        )

    report["baseline_test"] = _round_metrics(majority_baseline(y_test))

    best_pipeline = fitted[best_name]
    best_threshold = report["models"][best_name]["f1_threshold"]
    best_scores = test_scores[best_name]
    importance = extract_importance(best_pipeline)
    importance.to_csv(artifacts_dir / "feature_importance.csv", index=False)
    logistic_coefficients = extract_importance(fitted["logistic_regression"])
    logistic_coefficients.to_csv(artifacts_dir / "logistic_coefficients.csv", index=False)

    comparison_rows = []
    for name, payload in report["models"].items():
        row = {"model": name, "selected": name == best_name}
        row.update({f"oof_{key}": value for key, value in payload["oof_at_0_5"].items()})
        row.update({f"test_{key}": value for key, value in payload["test_at_0_5"].items()})
        row["f1_threshold"] = payload["f1_threshold"]
        row.update(
            {f"test_tuned_{key}": value for key, value in payload["test_at_f1_threshold"].items()}
        )
        comparison_rows.append(row)
    pd.DataFrame(comparison_rows).to_csv(artifacts_dir / "model_comparison.csv", index=False)

    report["outreach_scenario"] = outreach_comparison(
        y_test,
        best_scores,
        x_test["MonthlyCharges"],
        threshold=best_threshold,
    )
    report["top_features"] = importance.head(15).to_dict(orient="records")

    plot_roc_curves(y_test, test_scores, artifacts_dir / "roc_curve.png")
    plot_pr_curve(y_test, best_scores, best_name, artifacts_dir / "pr_curve.png")
    plot_confusion_matrix(
        y_test,
        best_scores,
        best_threshold,
        title=f"{DISPLAY_NAMES[best_name]} holdout, threshold {best_threshold:.2f}",
        path=artifacts_dir / "confusion_matrix.png",
    )
    plot_importance(importance, best_name, artifacts_dir / "feature_importance.png")
    plot_importance(
        logistic_coefficients,
        "logistic_regression",
        artifacts_dir / "logistic_coefficients.png",
    )
    report["logistic_top_coefficients"] = logistic_coefficients.head(12).to_dict(orient="records")
    plot_metric_bars(
        {name: report["models"][name]["test_at_0_5"] for name in report["models"]},
        artifacts_dir / "model_comparison.png",
    )

    joblib.dump(
        {
            "pipeline": best_pipeline,
            "model_name": best_name,
            "threshold": best_threshold,
            "feature_columns": list(x_train.columns),
            "positive_class": 1,
        },
        artifacts_dir / "model.joblib",
    )

    metrics_path = artifacts_dir / "metrics.json"
    metrics_path.write_text(json.dumps(report, indent=2) + "\n")
    logger.info("Selected %s. Wrote %s", best_name, metrics_path)
    return report
