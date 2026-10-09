"""Feature engineering pipelines and transformers for failure classification."""

import re
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


class FailureFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extracts and combines textual and numerical feature representations from test failure records."""

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """Converts raw records (DataFrame or list of dicts) into a structured DataFrame with combined text and numeric columns."""
        if isinstance(X, list):
            df = pd.DataFrame(X)
        elif isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            raise TypeError(f"Expected DataFrame or list of dicts, got {type(X)}")

        # Ensure all required fields exist
        for col in ["test_name", "error_type", "error_message", "stack_trace", "log_excerpt"]:
            if col not in df.columns:
                df[col] = ""
            else:
                df[col] = df[col].fillna("").astype(str)

        if "duration_seconds" not in df.columns:
            df["duration_seconds"] = 0.0
        else:
            df["duration_seconds"] = pd.to_numeric(df["duration_seconds"], errors="coerce").fillna(0.0)

        # Build clean consolidated textual representation
        combined_text = []
        for _, row in df.iterrows():
            parts = [
                f"[TEST] {row['test_name']}",
                f"[TYPE] {row['error_type']}",
                f"[MSG] {row['error_message']}",
                f"[TRACE] {row['stack_trace']}",
                f"[LOGS] {row['log_excerpt']}",
            ]
            text = " ".join([p for p in parts if p.strip()])
            # Normalize whitespace
            text = re.sub(r"\s+", " ", text).strip()
            combined_text.append(text)

        transformed_df = pd.DataFrame({
            "combined_text": combined_text,
            "duration_seconds": df["duration_seconds"].values,
        })
        return transformed_df


def build_ml_pipeline(
    model_type: str = "logistic_regression",
    random_state: int = 42,
    max_features: int = 1500,
) -> Pipeline:
    """Build a complete scikit-learn Pipeline with feature extraction, TF-IDF vectorization, and classifier.

    Supported model_type:
    - 'logistic_regression' (Baseline with calibrated probability estimates)
    - 'linear_svc' (Calibrated Linear Support Vector Classifier)
    - 'multinomial_nb' (Multinomial Naive Bayes)
    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "text",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_features=max_features,
                    sublinear_tf=True,
                    token_pattern=r"(?u)\b[a-zA-Z0-9_\-\.]{2,}\b",
                ),
                "combined_text",
            ),
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                ["duration_seconds"],
            ),
        ],
        remainder="drop",
    )

    if model_type == "logistic_regression":
        classifier = LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=random_state,
            class_weight="balanced",
        )
    elif model_type == "linear_svc":
        base_svc = LinearSVC(
            C=1.0,
            random_state=random_state,
            class_weight="balanced",
            dual=False,
            max_iter=2000,
        )
        classifier = CalibratedClassifierCV(estimator=base_svc, cv=3)
    elif model_type == "multinomial_nb":
        # MultinomialNB requires non-negative features; text only
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "text",
                    TfidfVectorizer(
                        ngram_range=(1, 2),
                        max_features=max_features,
                        sublinear_tf=True,
                    ),
                    "combined_text",
                ),
            ],
            remainder="drop",
        )
        classifier = MultinomialNB()
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose from 'logistic_regression', 'linear_svc', 'multinomial_nb'.")

    pipeline = Pipeline([
        ("extractor", FailureFeatureExtractor()),
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])

    return pipeline
