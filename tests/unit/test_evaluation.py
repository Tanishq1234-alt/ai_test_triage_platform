"""Unit tests for model evaluation, metrics generation, and model registry persistence."""

import os
import tempfile
import pytest
from src.ml.dataset import SyntheticDatasetGenerator
from src.ml.evaluate import evaluate_classifier, run_cross_validation
from src.ml.feature_engineering import build_ml_pipeline
from src.ml.model_registry import ModelRegistry


@pytest.fixture
def eval_dataset():
    gen = SyntheticDatasetGenerator(seed=42)
    return gen.generate_dataset(samples_per_category=15)


def test_evaluate_classifier_produces_complete_metrics(eval_dataset):
    X = eval_dataset[["test_name", "error_type", "error_message", "stack_trace", "log_excerpt", "duration_seconds"]]
    y = eval_dataset["canonical_category"]

    pipeline = build_ml_pipeline(model_type="logistic_regression", random_state=42)
    pipeline.fit(X.iloc[:60], y.iloc[:60])

    metrics = evaluate_classifier(pipeline, X.iloc[60:], y.iloc[60:])

    assert "ml_macro_f1" in metrics
    assert "ml_weighted_f1" in metrics
    assert "baseline_macro_f1" in metrics
    assert "baseline_weighted_f1" in metrics
    assert "ml_confusion_matrix" in metrics
    assert "comparison_table" in metrics
    assert len(metrics["comparison_table"]) == len(metrics["class_labels"])


def test_run_cross_validation(eval_dataset):
    X = eval_dataset[["test_name", "error_type", "error_message", "stack_trace", "log_excerpt", "duration_seconds"]]
    y = eval_dataset["canonical_category"]

    pipeline = build_ml_pipeline(model_type="logistic_regression", random_state=42)
    cv_res = run_cross_validation(pipeline, X, y, cv_splits=3)

    assert "mean_macro_f1" in cv_res
    assert "std_macro_f1" in cv_res
    assert cv_res["cv_splits"] == 3
    assert 0.0 <= cv_res["mean_macro_f1"] <= 1.0


def test_model_registry_save_and_safe_load():
    with tempfile.TemporaryDirectory() as tmp_dir:
        registry = ModelRegistry(models_dir=tmp_dir)
        pipeline = build_ml_pipeline(model_type="logistic_regression", random_state=42)
        metadata = {"test_run": True, "author": "triage_team"}

        save_paths = registry.save_model(pipeline, metadata, model_version="v_unit_test")
        assert os.path.exists(save_paths["model_path"])
        assert os.path.exists(save_paths["metadata_path"])

        loaded_pipeline, loaded_meta = registry.load_model(model_version="v_unit_test")
        assert loaded_pipeline is not None
        assert loaded_meta["author"] == "triage_team"
        assert loaded_meta["model_version"] == "v_unit_test"

        # Safe loading non-existent version raises FileNotFoundError
        with pytest.raises(FileNotFoundError):
            registry.load_model(model_version="non_existent_version_xyz")
