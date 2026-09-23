"""Checks for row-local cleaning. These do not download the dataset."""

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from churn.features import prepare_features
from churn.models import make_pipeline


def sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "customerID": "0001",
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 0,
                "PhoneService": "No",
                "MultipleLines": "No phone service",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 30.0,
                "TotalCharges": " ",
                "Churn": "No",
            },
            {
                "customerID": "0002",
                "gender": "Male",
                "SeniorCitizen": 1,
                "Partner": "No",
                "Dependents": "No",
                "tenure": 10,
                "PhoneService": "Yes",
                "MultipleLines": "Yes",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "Yes",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "Yes",
                "StreamingTV": "No",
                "StreamingMovies": "Yes",
                "Contract": "One year",
                "PaperlessBilling": "No",
                "PaymentMethod": "Credit card (automatic)",
                "MonthlyCharges": 80.0,
                "TotalCharges": "800.5",
                "Churn": "Yes",
            },
        ]
    )


class PrepareFeaturesTest(unittest.TestCase):
    def test_blank_bill_on_new_account_becomes_zero(self):
        features, target = prepare_features(sample_frame())
        self.assertEqual(float(features.loc[0, "TotalCharges"]), 0.0)
        self.assertEqual(float(features.loc[0, "avg_monthly_spend"]), 30.0)
        self.assertEqual(int(target.iloc[0]), 0)

    def test_addon_count_ignores_non_yes_levels(self):
        features, target = prepare_features(sample_frame())
        # Backup only. "No" and "No phone service" do not count.
        self.assertEqual(int(features.loc[0, "addon_count"]), 1)
        self.assertEqual(int(features.loc[1, "addon_count"]), 4)
        self.assertAlmostEqual(float(features.loc[1, "avg_monthly_spend"]), 80.05)
        self.assertEqual(int(target.iloc[1]), 1)

    def test_identifier_and_target_are_not_features(self):
        features, _ = prepare_features(sample_frame())
        self.assertNotIn("customerID", features.columns)
        self.assertNotIn("Churn", features.columns)

    def test_unknown_label_raises(self):
        frame = sample_frame()
        frame.loc[0, "Churn"] = "Maybe"
        with self.assertRaises(ValueError):
            prepare_features(frame)

    def test_pipeline_fits_on_the_tiny_frame(self):
        features, target = prepare_features(sample_frame())
        # Duplicate so both classes appear more than once for the tree splits.
        features = pd.concat([features, features, features], ignore_index=True)
        target = pd.concat([target, target, target], ignore_index=True)
        pipeline = make_pipeline(
            __import__("sklearn.linear_model", fromlist=["LogisticRegression"]).LogisticRegression(
                C=1.0, l1_ratio=0.0, max_iter=200
            )
        )
        pipeline.fit(features, target)
        scores = pipeline.predict_proba(features)
        self.assertEqual(scores.shape, (len(features), 2))


if __name__ == "__main__":
    unittest.main()
