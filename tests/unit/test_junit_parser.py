"""Unit tests for JUnit XML parsing."""

import os
import pytest
from src.log_parser.junit_parser import JUnitParser


@pytest.fixture
def parser():
    return JUnitParser()


@pytest.fixture
def fixtures_dir():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "fixtures"))


def test_parse_sample_junit_xml(parser, fixtures_dir):
    xml_path = os.path.join(fixtures_dir, "sample_junit.xml")
    records = parser.parse_file(xml_path)

    assert len(records) == 5

    # Check nominal test
    nominal = next(r for r in records if r.test_name == "test_connectivity_nominal")
    assert nominal.status == "passed"
    assert nominal.failure_category is None
    assert nominal.duration_seconds == 0.045
    assert nominal.error_type is None
    assert nominal.error_message is None
    assert "Associated successfully" in (nominal.log_excerpt or "")

    # Check timeout failure
    timeout_rec = next(r for r in records if r.test_name == "test_connectivity_handshake_timeout")
    assert timeout_rec.status == "failed"
    assert timeout_rec.failure_category == "timeout"
    assert timeout_rec.error_type == "WirelessTimeoutError"
    assert "handshake timed out" in timeout_rec.error_message
    assert "Traceback" in timeout_rec.stack_trace

    # Check assertion failure
    assert_rec = next(r for r in records if r.test_name == "test_latency_sla_breach")
    assert assert_rec.status == "failed"
    assert assert_rec.failure_category == "assertion"
    assert assert_rec.error_type == "WirelessAssertionError"
    assert "Latency SLA violation" in assert_rec.error_message

    # Check error test
    error_rec = next(r for r in records if r.test_name == "test_hal_driver_crash")
    assert error_rec.status == "error"
    assert error_rec.failure_category == "dependency"
    assert error_rec.error_type == "WirelessDependencyError"
    assert "HAL driver failed" in error_rec.error_message

    # Check skipped test
    skipped_rec = next(r for r in records if r.test_name == "test_6ghz_band_experimental")
    assert skipped_rec.status == "skipped"
    assert skipped_rec.failure_category is None
    assert "unavailable" in skipped_rec.error_message


def test_parse_missing_file_handled_gracefully(parser):
    records = parser.parse_file("non_existent_path.xml")
    assert records == []


def test_parse_empty_file_handled_gracefully(parser, fixtures_dir):
    xml_path = os.path.join(fixtures_dir, "empty.xml")
    records = parser.parse_file(xml_path)
    assert records == []


def test_parse_malformed_xml_handled_gracefully(parser, fixtures_dir):
    xml_path = os.path.join(fixtures_dir, "malformed.xml")
    records = parser.parse_file(xml_path)
    assert records == []


def test_parse_xml_missing_attributes(parser):
    xml_snippet = """<testsuite name="bare_suite">
        <testcase name="test_no_time" />
    </testsuite>"""
    records = parser.parse_string(xml_snippet)
    assert len(records) == 1
    assert records[0].test_name == "test_no_time"
    assert records[0].status == "passed"
    assert records[0].duration_seconds is None
    assert records[0].timestamp is None


def test_parse_stability_across_repeated_runs(parser, fixtures_dir):
    xml_path = os.path.join(fixtures_dir, "sample_junit.xml")
    run1 = parser.parse_file(xml_path)
    run2 = parser.parse_file(xml_path)
    assert len(run1) == len(run2)
    for r1, r2 in zip(run1, run2):
        assert r1.test_id == r2.test_id
        assert r1.status == r2.status
        assert r1.failure_category == r2.failure_category
        assert r1.duration_seconds == r2.duration_seconds
