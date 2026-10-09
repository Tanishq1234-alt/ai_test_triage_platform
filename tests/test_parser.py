"""Unit tests for the LogParser."""

import pytest
import pandas as pd
from src.models.schemas import FailureCategory, TestStatus
from src.parser.log_parser import LogParser


@pytest.fixture
def parser():
    return LogParser()


def test_parse_valid_log_line(parser):
    line = "2026-10-10 12:00:00 [INFO] [RADIO] Associated successfully with BSSID 00:11:22:33:44:55 RSSI=-55.0 dBm"
    parsed = parser.parse_log_line(line)
    assert parsed["level"] == "INFO"
    assert parsed["subsystem"] == "RADIO"
    assert "Associated successfully" in parsed["message"]


def test_extract_telemetry_metrics(parser):
    logs = [
        "2026-10-10 12:00:01 [INFO] [METRIC] telemetry: latency_ms=18.5 target=8.8.8.8",
        "2026-10-10 12:00:02 [INFO] [METRIC] telemetry: packet_loss_pct=0.5 transmitted=100",
        "2026-10-10 12:00:03 [INFO] [RADIO] Measured RSSI=-62.5 dBm snr_db=33.0",
    ]
    metrics = parser.extract_metrics(logs)
    assert metrics["latency_ms"] == 18.5
    assert metrics["packet_loss_pct"] == 0.5
    assert metrics["rssi"] == -62.5 or metrics["rssi_dbm"] == -62.5 or -62.5 in metrics.values()
    assert metrics["snr_db"] == 33.0


def test_extract_error_info_timeout(parser):
    logs = [
        "2026-10-10 12:00:01 [ERROR] [WIFI-STACK] Association timeout: Connection handshake timed out after 5.0s",
    ]
    trace = "WirelessTimeoutError: Connection handshake timed out after 5.0s"
    info = parser.extract_error_info(logs, trace)
    assert info["error_type"] == "WirelessTimeoutError"
    assert info["category"] == FailureCategory.TIMEOUT
    assert "handshake timed out" in info["error_message"]


def test_extract_error_info_dependency(parser):
    logs = [
        "2026-10-10 12:00:01 [CRITICAL] [HAL] Fatal hardware driver failure: HAL driver failed to respond: err_code=0xDEAD8002",
    ]
    trace = "WirelessDependencyError: HAL driver failed to respond: err_code=0xDEAD8002"
    info = parser.extract_error_info(logs, trace)
    assert info["error_type"] == "WirelessDependencyError"
    assert info["category"] == FailureCategory.DEPENDENCY


def test_parse_passed_test_result_resets_failure_category(parser):
    logs = [
        "2026-10-10 12:00:00 [INFO] [TEST] Starting test",
        "2026-10-10 12:00:01 [INFO] [TEST] Test PASSED successfully",
    ]
    res = parser.parse_test_result(
        test_id="T-001",
        test_name="test_nominal",
        status="PASSED",
        duration_ms=12.5,
        raw_logs=logs,
    )
    assert res.status == TestStatus.PASSED
    assert res.failure_category == FailureCategory.NONE
    assert res.error_type is None
    assert res.error_message is None


def test_results_to_dataframe(parser):
    logs_pass = ["2026-10-10 12:00:00 [INFO] [RADIO] telemetry: latency_ms=12.0"]
    logs_fail = ["2026-10-10 12:00:01 [ERROR] [WIFI-STACK] Handshake timeout"]

    res1 = parser.parse_test_result(
        test_id="T-01",
        test_name="test_pass",
        status="PASSED",
        duration_ms=15.0,
        raw_logs=logs_pass,
    )
    res2 = parser.parse_test_result(
        test_id="T-02",
        test_name="test_fail",
        status="FAILED",
        duration_ms=45.0,
        raw_logs=logs_fail,
        explicit_category=FailureCategory.TIMEOUT,
        explicit_error_type="WirelessTimeoutError",
        explicit_error_message="Handshake timeout",
    )

    df = parser.results_to_dataframe([res1, res2])
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "clean_log_text" in df.columns
    assert "failure_category" in df.columns
    assert df.loc[df["test_id"] == "T-01", "failure_category"].values[0] == "none"
    assert df.loc[df["test_id"] == "T-02", "failure_category"].values[0] == "timeout"
    assert df.loc[df["test_id"] == "T-01", "latency_ms"].values[0] == 12.0
