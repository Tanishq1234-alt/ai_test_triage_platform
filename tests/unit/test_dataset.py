"""Unit tests for dataset loading, label normalization, and synthetic generation."""

import os
import tempfile
import pandas as pd
import pytest
from src.ml.dataset import (
    CANONICAL_CATEGORIES,
    DatasetLoader,
    SyntheticDatasetGenerator,
    normalize_failure_category,
)


def test_normalize_failure_category_mappings():
    assert normalize_failure_category("timeout") == "timeout"
    assert normalize_failure_category("assertion") == "assertion_failure"
    assert normalize_failure_category("assertion_failure") == "assertion_failure"
    assert normalize_failure_category("connection") == "connection_error"
    assert normalize_failure_category("configuration") == "configuration_error"
    assert normalize_failure_category("dependency") == "dependency_error"
    assert normalize_failure_category("unknown") == "unknown"
    assert normalize_failure_category(None) == "unknown"
    assert normalize_failure_category("unexpected_random") == "unknown"


def test_dataset_loader_filters_passed_and_skipped():
    loader = DatasetLoader()
    raw_records = [
        {"test_id": "T1", "status": "passed", "failure_category": None},
        {"test_id": "T2", "status": "skipped", "failure_category": None},
        {"test_id": "T3", "status": "failed", "failure_category": "timeout", "error_message": "timeout occurred"},
        {"test_id": "T4", "status": "error", "failure_category": "dependency", "error_message": "driver died"},
    ]
    df = loader.process_records(raw_records)
    assert len(df) == 2
    assert set(df["test_id"]) == {"T3", "T4"}
    assert list(df["canonical_category"]) == ["timeout", "dependency_error"]


def test_dataset_loader_deduplication():
    loader = DatasetLoader()
    raw_records = [
        {"test_name": "test_dup", "error_message": "same error", "stack_trace": "same trace", "status": "failed", "failure_category": "timeout"},
        {"test_name": "test_dup", "error_message": "same error", "stack_trace": "same trace", "status": "failed", "failure_category": "timeout"},
        {"test_name": "test_diff", "error_message": "diff error", "stack_trace": "diff trace", "status": "failed", "failure_category": "timeout"},
    ]
    df = loader.process_records(raw_records)
    assert len(df) == 2


def test_dataset_loader_missing_fields_handled_cleanly():
    loader = DatasetLoader()
    raw_records = [
        {"status": "failed"}  # completely bare record
    ]
    df = loader.process_records(raw_records)
    assert len(df) == 1
    assert df.loc[0, "test_name"] == ""
    assert df.loc[0, "error_type"] == ""
    assert df.loc[0, "duration_seconds"] == 0.0
    assert df.loc[0, "canonical_category"] == "unknown"


def test_check_class_distribution_detects_underrepresented():
    loader = DatasetLoader(min_samples_per_class=5)
    df = pd.DataFrame([
        {"canonical_category": "timeout"},
        {"canonical_category": "timeout"},
        {"canonical_category": "timeout"},
    ])
    info = loader.check_class_distribution(df)
    assert info["total"] == 3
    assert info["counts"]["timeout"] == 3
    assert "timeout" in info["underrepresented"]
    assert "assertion_failure" in info["underrepresented"]


def test_synthetic_dataset_generator_reproducibility():
    gen1 = SyntheticDatasetGenerator(seed=123)
    df1 = gen1.generate_dataset(samples_per_category=5)

    gen2 = SyntheticDatasetGenerator(seed=123)
    df2 = gen2.generate_dataset(samples_per_category=5)

    assert len(df1) == len(df2)
    assert list(df1["error_message"]) == list(df2["error_message"])
    assert list(df1["canonical_category"]) == list(df2["canonical_category"])
    assert all(cat in df1["canonical_category"].values for cat in CANONICAL_CATEGORIES)
