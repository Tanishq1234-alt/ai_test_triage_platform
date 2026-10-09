"""Unit tests for schema normalization, correlation, and rule-based categorization."""

import os
import pytest
from src.log_parser import correlate_and_normalize
from src.log_parser.normalizer import NormalizedTestRecord, RuleBasedCategorizer, TestResultNormalizer


@pytest.fixture
def normalizer():
    return TestResultNormalizer()


@pytest.fixture
def fixtures_dir():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "fixtures"))


def test_rule_based_categorization_categories(normalizer):
    cat = RuleBasedCategorizer()

    # Timeout
    assert cat.categorize("failed", error_type="WirelessTimeoutError", error_message="Probe response timer expired") == "timeout"
    assert cat.categorize("failed", error_message="Ping deadline exceeded after 5.0s") == "timeout"

    # Assertion
    assert cat.categorize("failed", error_type="AssertionError", error_message="assert 185.0 < 50.0") == "assertion"
    assert cat.categorize("failed", error_message="Latency SLA violation: measured RTT 120 ms") == "assertion"
    assert cat.categorize("failed", error_message="Packet loss threshold breach: 12% dropped") == "assertion"

    # Connection
    assert cat.categorize("failed", error_type="WirelessConnectionError", error_message="Beacon frame loss detected") == "connection"
    assert cat.categorize("failed", error_message="Radio link lost: carrier drop across 3 attempts") == "connection"

    # Configuration
    assert cat.categorize("failed", error_type="WirelessConfigError", error_message="Invalid SSID profile") == "configuration"
    assert cat.categorize("failed", error_message="Authentication key must be at least 8 characters") == "configuration"

    # Dependency
    assert cat.categorize("error", error_type="WirelessDependencyError", error_message="HAL driver failed: kernel ring buffer corrupted") == "dependency"
    assert cat.categorize("error", error_type="ModuleNotFoundError", error_message="No module named 'hardware_hal'") == "dependency"

    # Unknown
    assert cat.categorize("failed", error_type="CustomUnmappedError", error_message="Something unexpected happened in subroutine beta") == "unknown"

    # Passed / Skipped
    assert cat.categorize("passed") is None
    assert cat.categorize("skipped") is None


def test_normalizer_preserves_none_without_inventing_info(normalizer):
    rec = normalizer.normalize(
        test_id="TC-999",
        test_name="test_minimal",
        status="passed",
    )
    assert rec.test_id == "TC-999"
    assert rec.test_name == "test_minimal"
    assert rec.status == "passed"
    assert rec.failure_category is None
    assert rec.duration_seconds is None
    assert rec.timestamp is None
    assert rec.error_type is None
    assert rec.error_message is None
    assert rec.stack_trace is None
    assert rec.log_excerpt is None
    assert rec.source_file is None


def test_correlate_and_normalize_enriches_junit_with_logs(fixtures_dir):
    xml_path = os.path.join(fixtures_dir, "sample_junit.xml")
    log_path = os.path.join(fixtures_dir, "sample_app.log")

    records = correlate_and_normalize(junit_path=xml_path, log_path=log_path)
    assert len(records) == 5

    # test_connectivity_nominal in XML matches test_connectivity_nominal in log
    nominal = next(r for r in records if r.test_name == "test_connectivity_nominal")
    assert nominal.status == "passed"
    # Was enriched with precise timestamp from log
    assert nominal.timestamp == "2026-10-10 12:00:01"
