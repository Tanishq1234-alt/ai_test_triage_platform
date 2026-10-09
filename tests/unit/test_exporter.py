"""Unit tests for RecordExporter."""

import json
import os
import tempfile
import pandas as pd
import pytest
from src.log_parser.exporter import RecordExporter
from src.log_parser.normalizer import NormalizedTestRecord


@pytest.fixture
def exporter():
    return RecordExporter()


@pytest.fixture
def sample_records():
    return [
        NormalizedTestRecord(
            test_id="TC-01",
            test_name="test_passed",
            test_suite="suite_1",
            timestamp="2026-10-10 12:00:00",
            status="passed",
            duration_seconds=0.05,
            source_file="tests/test_demo.py",
        ),
        NormalizedTestRecord(
            test_id="TC-02",
            test_name="test_failed",
            test_suite="suite_1",
            timestamp="2026-10-10 12:00:01",
            status="failed",
            duration_seconds=1.23,
            error_type="WirelessTimeoutError",
            error_message="Handshake timed out",
            stack_trace="Traceback...",
            log_excerpt="2026-10-10 [ERROR] Timeout",
            failure_category="timeout",
            source_file="tests/test_demo.py",
        ),
    ]


def test_export_to_json_and_load(exporter, sample_records):
    with tempfile.TemporaryDirectory() as tmpdir:
        json_file = os.path.join(tmpdir, "out.json")
        exporter.export_to_json(sample_records, json_file)

        assert os.path.exists(json_file)
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert len(data) == 2
        assert data[0]["test_id"] == "TC-01"
        assert data[0]["status"] == "passed"
        assert data[0]["failure_category"] is None
        assert data[1]["test_id"] == "TC-02"
        assert data[1]["failure_category"] == "timeout"


def test_export_to_csv_and_load(exporter, sample_records):
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_file = os.path.join(tmpdir, "out.csv")
        exporter.export_to_csv(sample_records, csv_file)

        assert os.path.exists(csv_file)
        df = pd.read_csv(csv_file)
        assert len(df) == 2
        assert list(df.columns) == exporter.COLUMNS
        assert df.loc[0, "test_id"] == "TC-01"
        assert df.loc[1, "failure_category"] == "timeout"


def test_export_preserves_history_in_append_mode(exporter, sample_records):
    with tempfile.TemporaryDirectory() as tmpdir:
        json_file = os.path.join(tmpdir, "history.json")
        csv_file = os.path.join(tmpdir, "history.csv")

        # Run 1: 2 records
        exporter.export_to_json(sample_records, json_file, append=True)
        exporter.export_to_csv(sample_records, csv_file, append=True)

        # Run 2: append 2 more records
        exporter.export_to_json(sample_records, json_file, append=True)
        exporter.export_to_csv(sample_records, csv_file, append=True)

        # Verify JSON accumulated 4 records
        with open(json_file, "r", encoding="utf-8") as f:
            json_data = json.load(f)
        assert len(json_data) == 4

        # Verify CSV accumulated 4 records
        df = pd.read_csv(csv_file)
        assert len(df) == 4


def test_to_dataframe(exporter, sample_records):
    df = exporter.to_dataframe(sample_records)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "log_excerpt" in df.columns
    assert "failure_category" in df.columns
