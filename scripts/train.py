#!/usr/bin/env python3
"""Train the telco churn classifiers and write artifacts/."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from churn.config import ARTIFACTS_DIR, DATA_PATH  # noqa: E402
from churn.pipeline import train_and_evaluate  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train telco churn classifiers and write artifacts/.")
    parser.add_argument("--data", type=Path, default=DATA_PATH, help="CSV path. Downloaded if missing.")
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS_DIR, help="Directory for metrics, plots, and the model.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    report = train_and_evaluate(data_path=args.data, artifacts_dir=args.artifacts)
    best = report["best_model"]
    test_metrics = report["models"][best]["test_at_0_5"]
    tuned = report["models"][best]["test_at_f1_threshold"]
    print()
    print(f"Best model (by training OOF ROC-AUC): {best}")
    print(
        "Holdout at 0.50: "
        f"accuracy {test_metrics['accuracy']:.3f}  "
        f"precision {test_metrics['precision']:.3f}  "
        f"recall {test_metrics['recall']:.3f}  "
        f"f1 {test_metrics['f1']:.3f}  "
        f"roc_auc {test_metrics['roc_auc']:.3f}"
    )
    print(
        f"Holdout at F1 threshold {tuned['threshold']:.2f}: "
        f"accuracy {tuned['accuracy']:.3f}  "
        f"precision {tuned['precision']:.3f}  "
        f"recall {tuned['recall']:.3f}  "
        f"f1 {tuned['f1']:.3f}  "
        f"roc_auc {tuned['roc_auc']:.3f}"
    )
    print(f"Artifacts: {args.artifacts.resolve()}")


if __name__ == "__main__":
    main()
