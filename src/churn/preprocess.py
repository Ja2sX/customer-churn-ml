"""Preprocessing that is fit inside each training fold."""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def build_preprocessor() -> ColumnTransformer:
    """Median-impute and scale numerics; one-hot encode categoricals.

    Imputation and scaling are inside the pipeline so fold statistics cannot
    leak into validation rows. Unknown categories at score time become zeros.
    """
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric, list(NUMERIC_FEATURES)),
            ("cat", categorical, list(CATEGORICAL_FEATURES)),
        ]
    )
