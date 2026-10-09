"""Unit tests for the FailureClassifierPredictor inference interface."""

import pytest
from src.ml.dataset import SyntheticDatasetGenerator
from src.ml.feature_engineering import build_ml_pipeline
from src.ml.predict import FailureClassifierPredictor


@pytest.fixture
def trained_predictor():
    gen = SyntheticDatasetGenerator(seed=42)
    df = gen.generate_dataset(samples_per_category=20)
    X = df[["test_name", "error_type", "error_message", "stack_trace", "log_excerpt", "duration_seconds"]]
    y = df["canonical_category"]

    pipeline = build_ml_pipeline(model_type="logistic_regression", random_state=42)
    pipeline.fit(X, y)

    return FailureClassifierPredictor(
        pipeline=pipeline,
        metadata={"model_version": "v_test"},
        confidence_threshold=0.50,
    )


def test_predict_output_schema(trained_predictor):
    sample_rec = {
        "test_id": "TC-101",
        "test_name": "test_latency_sla",
        "error_type": "WirelessAssertionError",
        "error_message": "Latency SLA violation: measured RTT 240.5 ms exceeds maximum allowed 50.0 ms",
        "stack_trace": "line 42 in ping",
        "log_excerpt": "telemetry: latency_ms=240.5",
        "duration_seconds": 0.5,
    }
    pred = trained_predictor.predict(sample_rec)

    assert "predicted_category" in pred
    assert "confidence" in pred
    assert "probabilities" in pred
    assert "model_version" in pred
    assert "needs_review" in pred
    assert "top_features" in pred
    assert "rule_based_category" in pred
    assert "is_agreement" in pred

    assert pred["predicted_category"] == "assertion_failure"
    assert pred["confidence"] >= 0.50
    assert pred["needs_review"] is False
    assert isinstance(pred["top_features"], list)
    assert len(pred["top_features"]) > 0


def test_predict_low_confidence_triggers_review(trained_predictor):
    trained_predictor.confidence_threshold = 0.9999  # artificially high threshold
    sample_rec = {
        "test_id": "TC-102",
        "test_name": "test_general",
        "error_type": "WirelessTimeoutError",
        "error_message": "Connection handshake timed out after 3.0s",
        "log_excerpt": "timeout error",
    }
    pred = trained_predictor.predict(sample_rec)

    assert pred["needs_review"] is True
    assert pred["predicted_category"] == "unknown"
    assert "Low prediction confidence" in (pred["review_reason"] or "")


def test_predict_completely_empty_record_handles_gracefully(trained_predictor):
    empty_rec = {}
    pred = trained_predictor.predict(empty_rec)

    assert pred["predicted_category"] == "unknown"
    assert pred["confidence"] == 0.0
    assert pred["needs_review"] is True
    assert "Insufficient log content" in pred["review_reason"]


def test_predict_unknown_anomaly_triggers_review(trained_predictor):
    anomaly_rec = {
        "test_id": "TC-103",
        "test_name": "test_system_anomaly",
        "error_type": "RuntimeError",
        "error_message": "Unrecognized system anomaly in thread: diagnostic code 0x8F91",
        "log_excerpt": "General exception raised with unmapped diagnostic code",
    }
    pred = trained_predictor.predict(anomaly_rec)

    assert pred["predicted_category"] == "unknown"
    assert pred["needs_review"] is True
