"""Unit tests for feature engineering transformers and pipelines."""

import pandas as pd
import pytest
from src.ml.feature_engineering import FailureFeatureExtractor, build_ml_pipeline


def test_failure_feature_extractor_combines_text_fields():
    extractor = FailureFeatureExtractor()
    df_raw = pd.DataFrame([{
        "test_name": "test_ping_failure",
        "error_type": "WirelessTimeoutError",
        "error_message": "ICMP probe timed out",
        "stack_trace": "line 10 in test",
        "log_excerpt": "Sending probe... no reply",
        "duration_seconds": 2.5,
    }])
    transformed = extractor.transform(df_raw)
    assert isinstance(transformed, pd.DataFrame)
    assert "combined_text" in transformed.columns
    assert "duration_seconds" in transformed.columns

    combined = transformed["combined_text"].iloc[0]
    assert "[TEST] test_ping_failure" in combined
    assert "[TYPE] WirelessTimeoutError" in combined
    assert "[MSG] ICMP probe timed out" in combined
    assert "[TRACE] line 10 in test" in combined
    assert "[LOGS] Sending probe... no reply" in combined
    assert transformed["duration_seconds"].iloc[0] == 2.5


def test_failure_feature_extractor_handles_missing_columns():
    extractor = FailureFeatureExtractor()
    df_bare = pd.DataFrame([{"error_message": "Single field only"}])
    transformed = extractor.transform(df_bare)
    assert "combined_text" in transformed.columns
    assert "duration_seconds" in transformed.columns
    assert transformed["duration_seconds"].iloc[0] == 0.0
    assert "Single field only" in transformed["combined_text"].iloc[0]


def test_build_ml_pipeline_construction():
    pipeline_lr = build_ml_pipeline(model_type="logistic_regression")
    assert "extractor" in pipeline_lr.named_steps
    assert "preprocessor" in pipeline_lr.named_steps
    assert "classifier" in pipeline_lr.named_steps

    pipeline_svc = build_ml_pipeline(model_type="linear_svc")
    assert "classifier" in pipeline_svc.named_steps

    with pytest.raises(ValueError):
        build_ml_pipeline(model_type="invalid_model_type")
