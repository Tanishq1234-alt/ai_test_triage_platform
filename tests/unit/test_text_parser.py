"""Unit tests for PlainTextLogParser."""

import os
import pytest
from src.log_parser.text_parser import PlainTextLogParser


@pytest.fixture
def parser():
    return PlainTextLogParser(context_lines_before=2, context_lines_after=2)


@pytest.fixture
def fixtures_dir():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "fixtures"))


def test_parse_log_line_components(parser):
    line = "2026-10-10 12:00:01 [ERROR] [WIFI-STACK] Association handshake timed out after 3.00s"
    parsed = parser.parse_log_line(line)
    assert parsed["timestamp"] == "2026-10-10 12:00:01"
    assert parsed["level"] == "ERROR"
    assert parsed["subsystem"] == "WIFI-STACK"


def test_parse_sample_app_log(parser, fixtures_dir):
    log_path = os.path.join(fixtures_dir, "sample_app.log")
    records = parser.parse_file(log_path)

    assert len(records) == 3

    nominal = next(r for r in records if r.test_name == "test_connectivity_nominal")
    assert nominal.status == "passed"
    assert nominal.failure_category is None
    assert nominal.error_message is None

    timeout_rec = next(r for r in records if r.test_name == "test_connection_timeout_failure")
    assert timeout_rec.status == "failed"
    assert timeout_rec.failure_category == "timeout"
    assert "Association timeout" in (timeout_rec.error_message or "")
    assert timeout_rec.log_excerpt is not None
    assert "Initiating Wi-Fi association" in timeout_rec.log_excerpt

    assert_rec = next(r for r in records if r.test_name == "test_latency_assertion_failure")
    assert assert_rec.status == "failed"
    assert assert_rec.failure_category == "assertion"
    assert "Latency SLA violation" in (assert_rec.error_message or "")


def test_noisy_logs_do_not_flag_false_positive_failure(parser, fixtures_dir):
    """Verifies that lines containing 'error' in normal info contexts do not mark tests failed."""
    noisy_path = os.path.join(fixtures_dir, "noisy_logs.log")
    records = parser.parse_file(noisy_path)

    assert len(records) == 1
    rec = records[0]
    assert rec.status == "passed"
    assert rec.failure_category is None
    assert rec.error_message is None


def test_context_window_preserves_lines_before_and_after(parser):
    lines = [
        "Line 0: setup starting",
        "Line 1: configuring hardware",
        "Line 2: pre-check passed",
        "2026-10-10 12:00:00 [ERROR] [RADIO] Radio transmitter crashed",
        "Line 4: post-error cleanup attempt",
        "Line 5: teardown starting",
        "Line 6: teardown complete",
    ]
    error_idx = [3]
    excerpt = parser.extract_context_window(lines, error_idx)

    assert "Line 1: configuring hardware" in excerpt
    assert "Line 2: pre-check passed" in excerpt
    assert "Radio transmitter crashed" in excerpt
    assert "Line 4: post-error cleanup attempt" in excerpt
    assert "Line 5: teardown starting" in excerpt
    # Line 0 and Line 6 are outside the 2-line window
    assert "Line 0: setup starting" not in excerpt
    assert "Line 6: teardown complete" not in excerpt


def test_missing_and_empty_log_files(parser):
    assert parser.parse_file("non_existent_file.log") == []
    assert parser.parse_text("   \n\t  ") == []
