"""Row-local cleaning and feature engineering.

Transforms in this module use only the current customer's fields. They do not
estimate means, frequencies, or target rates, so they are safe to run before
the train/test split. Statistics that must be learned (medians, scales, category
levels) live in the sklearn pipeline and are fit on training rows only.
"""

import pandas as pd

from churn.config import (
    ADDON_COLUMNS,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    POSITIVE_LABEL,
    TARGET_COLUMN,
)


def prepare_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return model matrix X and a 0/1 churn label.

    TotalCharges is stored as text in the IBM sample. Eleven customers with
    tenure 0 have a blank bill (a single space). Those accounts have not been
    invoiced yet, so the bill is set to 0 rather than imputed from other
    customers. avg_monthly_spend is the realized bill divided by tenure, and
    falls back to the listed monthly price when there is no billing history.
    """
    if TARGET_COLUMN not in frame.columns:
        raise ValueError(f"Missing target column {TARGET_COLUMN!r}.")

    data = frame.copy()
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    unbilled = data["TotalCharges"].isna() & (data["tenure"] == 0)
    data.loc[unbilled, "TotalCharges"] = 0.0

    data["addon_count"] = (data[ADDON_COLUMNS] == "Yes").sum(axis=1).astype(int)

    tenure = data["tenure"].astype(float)
    data["avg_monthly_spend"] = data["MonthlyCharges"].astype(float)
    billed = tenure > 0
    data.loc[billed, "avg_monthly_spend"] = data.loc[billed, "TotalCharges"] / tenure.loc[billed]

    labels = data[TARGET_COLUMN].map({POSITIVE_LABEL: 1, "No": 0})
    if labels.isna().any():
        unknown = sorted(data.loc[labels.isna(), TARGET_COLUMN].astype(str).unique())
        raise ValueError(f"Unexpected Churn labels: {unknown}")

    target = labels.astype(int)
    target.name = TARGET_COLUMN
    features = data[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    return features, target
