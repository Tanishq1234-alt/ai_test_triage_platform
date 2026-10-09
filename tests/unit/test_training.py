"""Unit tests for ML model training, candidate selection, and leakage prevention."""

import os
import tempfile
import pandas as pd
import pytest
from src.ml.dataset import SyntheticDatasetGenerator
from src.ml.feature_engineering import build_ml_pipeline
from src.ml.train import train_and_evaluate


@pytest.fixture
def sample_dataset():
    gen = SyntheticDatasetGenerator(seed=42)
    return gen.generate_dataset(samples_per_category=15)


def test_pipeline_trains_and_predicts(sample_dataset):
    pipeline = build_ml_pipeline(model_type="logistic_regression", random_state=42)
    X = sample_dataset[["test_name", "error_type", "error_message", "stack_trace", "log_excerpt", "duration_seconds"]]
    y = sample_dataset["canonical_category"]

    pipeline.fit(X, y)
    preds = pipeline.predict(X)
    assert len(preds) == len(y)
    assert set(preds).issubset(set(pipeline.classes_))


def test_leakage_prevention_on_unseen_vocabulary(sample_dataset):
    """Verifies that test features do not influence the trained TF-IDF vocabulary."""
    pipeline = build_ml_pipeline(model_type="logistic_regression", random_state=42)
    X_train = sample_dataset.iloc[:60][["test_name", "error_type", "error_message", "stack_trace", "log_excerpt", "duration_seconds"]]
    y_train = sample_dataset.iloc[:60]["canonical_category"]

    pipeline.fit(X_train, y_train)
    tfidf = pipeline.named_steps["preprocessor"].named_transformers_["text"]
    initial_vocab_size = len(tfidf.vocabulary_)

    # Create unseen test sample with novel out-of-vocabulary terms
    X_test_novel = pd.DataFrame([{
        "test_name": "test_unseen_special_case",
        "error_type": "SpecialNovelError",
        "error_message": "UnseenTokenAlpha UnseenTokenBeta novel failure",
        "stack_trace": "",
        "log_excerpt": "Novel telemetry stream",
        "duration_seconds": 1.0,
    }])

    # Predicting on test data must not alter fitted vocabulary size
    pipeline.predict(X_test_novel)
    assert len(tfidf.vocabulary_) == initial_vocab_size


def test_train_and_evaluate_end_to_end(sample_dataset):
    with tempfile.TemporaryDirectory() as tmp_dir:
        models_dir = os.path.join(tmp_dir, "models")
        reports_dir = os.path.join(tmp_dir, "reports")

        result = train_and_evaluate(
            df=sample_dataset,
            test_size=0.25,
            seed=42,
            model_version="test_v1",
            models_dir=models_dir,
            reports_dir=reports_dir,
        )

        assert result["best_model_name"] in ["logistic_regression", "linear_svc"]
        assert os.path.exists(result["model_paths"]["model_path"])
        assert os.path.exists(result["report_path"])
        assert result["metadata"]["macro_f1"] >= 0.80
