"""Metrics, plots, and a transparent outreach scenario."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from churn.config import CATEGORICAL_FEATURES, DISPLAY_NAMES
from churn.plotting import MODEL_COLORS, NAVY, SLATE, TERRACOTTA, TEAL, apply_style, save_figure, style_axes

apply_style()


def binary_metrics(y_true, y_prob, threshold: float) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in matrix.ravel())
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "average_precision": float(average_precision_score(y_true, y_prob)),
        "predicted_positive_rate": float(y_pred.mean()),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def majority_baseline(y_true) -> dict:
    """Predict the training majority class (stay) for every row.

    ROC-AUC is 0.5 because the score is constant. Precision and recall of the
    churn class are zero. Accuracy equals the stay rate and is the number a
    model has to beat before accuracy means anything on this set.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.zeros(len(y_true), dtype=float)
    metrics = binary_metrics(y_true, y_prob, threshold=0.5)
    metrics["roc_auc"] = 0.5
    metrics["note"] = "Constant score for the majority class (no churn)."
    return metrics


def best_f1_threshold(y_true, y_prob) -> float:
    """Threshold on out-of-fold training scores. Never call this with test labels."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.linspace(0.05, 0.95, 91):
        score = f1_score(y_true, (y_prob >= threshold).astype(int), zero_division=0)
        if score > best_f1:
            best_f1 = float(score)
            best_threshold = float(threshold)
    # The search grid is 0.01 wide; round off binary floating-point residue.
    return round(best_threshold, 2)


def positive_scores(pipeline, features: pd.DataFrame) -> np.ndarray:
    probabilities = pipeline.predict_proba(features)
    classes = list(pipeline.named_steps["classifier"].classes_)
    return probabilities[:, classes.index(1)]


def pretty_feature_name(raw_name: str) -> str:
    name = raw_name.split("__", 1)[1] if "__" in raw_name else raw_name
    for column in sorted(CATEGORICAL_FEATURES, key=len, reverse=True):
        prefix = column + "_"
        if name.startswith(prefix):
            return f"{column} = {name[len(prefix):]}"
    return name


def extract_importance(pipeline) -> pd.DataFrame:
    preprocessor = pipeline.named_steps["preprocess"]
    classifier = pipeline.named_steps["classifier"]
    names = [pretty_feature_name(name) for name in preprocessor.get_feature_names_out()]
    if hasattr(classifier, "coef_"):
        values = np.asarray(classifier.coef_).ravel()
        kind = "coefficient"
    elif hasattr(classifier, "feature_importances_"):
        values = np.asarray(classifier.feature_importances_).ravel()
        kind = "importance"
    else:
        raise TypeError(f"No coefficient or importance on {type(classifier).__name__}.")
    table = pd.DataFrame({"feature": names, "value": values, "kind": kind})
    table["abs_value"] = table["value"].abs()
    return table.sort_values("abs_value", ascending=False).reset_index(drop=True)


def campaign_scenario(
    y_true,
    y_pred,
    monthly_charges,
    save_rate: float,
    months_retained: float,
    offer_fraction: float,
) -> dict:
    """Gross revenue scenario for one contact policy.

    Assumptions, all explicit:
    - Contacting a customer costs offer_fraction of their current monthly charge, once.
    - A contacted churner is retained with probability save_rate and then contributes
      months_retained of their current monthly charge. That figure is gross charges,
      not margin, and it assumes the monthly price stays flat.
    - Contacted customers who would have stayed, and customers who are not contacted,
      contribute no retained revenue in this ledger.
    - Nothing here is a causal estimate of a real retention offer.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    charges = np.asarray(monthly_charges, dtype=float)
    contacted = y_pred == 1
    caught = contacted & (y_true == 1)
    spend = float((offer_fraction * charges[contacted]).sum())
    retained = float((save_rate * months_retained * charges[caught]).sum())
    return {
        "customers_contacted": int(contacted.sum()),
        "churners_caught": int(caught.sum()),
        "offer_spend": round(spend, 2),
        "assumed_retained_revenue": round(retained, 2),
        "net_value": round(retained - spend, 2),
    }


def outreach_comparison(y_true, y_prob, monthly_charges, threshold: float) -> dict:
    y_true = np.asarray(y_true).astype(int)
    charges = np.asarray(monthly_charges, dtype=float)
    model_pred = (np.asarray(y_prob) >= threshold).astype(int)
    assumptions = {
        "offer_fraction_of_monthly_charge": 0.20,
        "months_retained_if_saved": 3,
        "save_rates": [0.30, 1.00],
        "value_is": "gross monthly charges, not margin",
    }
    policies = {
        "contact_none": np.zeros(len(y_true), dtype=int),
        "contact_model_score": model_pred,
        "contact_everyone": np.ones(len(y_true), dtype=int),
    }
    by_save_rate = {}
    for save_rate in assumptions["save_rates"]:
        by_save_rate[f"{save_rate:.2f}"] = {
            name: campaign_scenario(
                y_true,
                pred,
                charges,
                save_rate=save_rate,
                months_retained=assumptions["months_retained_if_saved"],
                offer_fraction=assumptions["offer_fraction_of_monthly_charge"],
            )
            for name, pred in policies.items()
        }
    return {"assumptions": assumptions, "by_save_rate": by_save_rate}


def plot_confusion_matrix(y_true, y_prob, threshold: float, title: str, path: Path) -> None:
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    matrix = confusion_matrix(np.asarray(y_true).astype(int), y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks([0, 1], ["Predicted stay", "Predicted churn"])
    ax.set_yticks([0, 1], ["Stayed", "Churned"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for row in range(2):
        for col in range(2):
            value = int(matrix[row, col])
            color = "white" if value > matrix.max() / 2 else INK
            ax.text(col, row, f"{value:,}", ha="center", va="center", color=color, fontsize=13)
    ax.set_title(title)
    save_figure(fig, path)


# Local alias so the annotation color stays next to the plot code.
INK = "#1C2833"


def plot_roc_curves(y_true, score_map: dict[str, np.ndarray], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    y_true = np.asarray(y_true).astype(int)
    for name, scores in score_map.items():
        fpr, tpr, _ = roc_curve(y_true, scores)
        score = roc_auc_score(y_true, scores)
        ax.plot(
            fpr,
            tpr,
            color=MODEL_COLORS[name],
            lw=2.2,
            label=f"{DISPLAY_NAMES[name]}  {score:.3f}",
        )
    ax.plot([0, 1], [0, 1], color=SLATE, lw=1, ls="--", label="Chance  0.500")
    style_axes(ax, grid="both")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Holdout ROC curves")
    ax.legend(frameon=False, title="ROC-AUC", loc="lower right")
    save_figure(fig, path)


def plot_pr_curve(y_true, y_prob, model_name: str, path: Path) -> None:
    y_true = np.asarray(y_true).astype(int)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    average_precision = float(auc(recall, precision))
    baseline = float(np.mean(y_true))
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.plot(recall, precision, color=MODEL_COLORS[model_name], lw=2.2, label=f"Model  AP {average_precision:.3f}")
    ax.axhline(baseline, color=SLATE, lw=1, ls="--", label=f"Churn prevalence  {baseline:.3f}")
    style_axes(ax, grid="both")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Holdout precision–recall ({DISPLAY_NAMES[model_name]})")
    ax.legend(frameon=False, loc="upper right")
    save_figure(fig, path)


def plot_importance(table: pd.DataFrame, model_name: str, path: Path, top_n: int = 15) -> None:
    subset = table.head(top_n).iloc[::-1]
    kind = subset["kind"].iloc[0]
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    if kind == "coefficient":
        colors = [TEAL if value >= 0 else TERRACOTTA for value in subset["value"]]
        ax.barh(subset["feature"], subset["value"], color=colors, zorder=2)
        ax.axvline(0, color=SLATE, lw=1)
        ax.set_xlabel("Coefficient (numerics are standardized)")
        title = f"Largest coefficients ({DISPLAY_NAMES[model_name]})"
    else:
        ax.barh(subset["feature"], subset["value"], color=NAVY, zorder=2)
        ax.set_xlabel("Impurity importance")
        title = f"Top feature importances ({DISPLAY_NAMES[model_name]})"
    style_axes(ax, grid="x")
    ax.set_title(title)
    save_figure(fig, path)


def plot_metric_bars(test_metrics: dict[str, dict], path: Path) -> None:
    metric_keys = ["roc_auc", "f1", "recall", "precision"]
    metric_labels = ["ROC-AUC", "F1", "Recall", "Precision"]
    names = list(test_metrics)
    x = np.arange(len(metric_keys))
    width = 0.24
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for index, name in enumerate(names):
        offsets = x + (index - (len(names) - 1) / 2) * width
        values = [test_metrics[name][key] for key in metric_keys]
        ax.bar(offsets, values, width=width, color=MODEL_COLORS[name], label=DISPLAY_NAMES[name], zorder=2)
    style_axes(ax, grid="y")
    ax.set_xticks(x, metric_labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Holdout score at threshold 0.50")
    ax.set_title("Model comparison on the untouched test set")
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    save_figure(fig, path)
