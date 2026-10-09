"""Structured log parser converting raw test execution logs into structured schemas and DataFrames."""

import datetime
import json
import re
from typing import Any, Dict, List, Optional
import pandas as pd

from src.models.schemas import FailureCategory, TestCaseResult, TestSessionSummary, TestStatus


class LogParser:
    """Parses raw text and session logs from wireless device test runs into structured records."""

    # Regex patterns for parsing log components
    LOG_LINE_PATTERN = re.compile(
        r"^(?P<timestamp>\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2})\s+\[(?P<level>[A-Z]+)\]\s+\[(?P<subsystem>[^\]]+)\]\s+(?P<message>.*)$"
    )
    METRIC_PATTERN = re.compile(r"(\b[a-zA-Z_][a-zA-Z0-9_]*)\s*[:=]\s*([+-]?\d+(?:\.\d+)?)\s*(?:dBm|dB|ms|%|s)?\b")
    EXCEPTION_PATTERN = re.compile(r"(?:([A-Za-z0-9_]+Error|[A-Za-z0-9_]+Exception))\s*:\s*(.*)")

    EXCEPTION_CATEGORY_MAP = {
        "WirelessTimeoutError": FailureCategory.TIMEOUT,
        "TimeoutError": FailureCategory.TIMEOUT,
        "WirelessAssertionError": FailureCategory.ASSERTION,
        "AssertionError": FailureCategory.ASSERTION,
        "WirelessConfigError": FailureCategory.CONFIGURATION,
        "KeyError": FailureCategory.CONFIGURATION,
        "ValueError": FailureCategory.CONFIGURATION,
        "WirelessConnectionError": FailureCategory.CONNECTION,
        "ConnectionResetError": FailureCategory.CONNECTION,
        "ConnectionRefusedError": FailureCategory.CONNECTION,
        "WirelessDependencyError": FailureCategory.DEPENDENCY,
        "FileNotFoundError": FailureCategory.DEPENDENCY,
        "ModuleNotFoundError": FailureCategory.DEPENDENCY,
        "OSError": FailureCategory.DEPENDENCY,
    }

    def parse_log_line(self, line: str) -> Dict[str, Any]:
        """Parse a single formatted log line into its components."""
        match = self.LOG_LINE_PATTERN.match(line.strip())
        if match:
            return {
                "timestamp": match.group("timestamp"),
                "level": match.group("level"),
                "subsystem": match.group("subsystem").strip(),
                "message": match.group("message").strip(),
                "raw": line.strip(),
            }
        return {
            "timestamp": "",
            "level": "INFO",
            "subsystem": "GENERAL",
            "message": line.strip(),
            "raw": line.strip(),
        }

    def extract_metrics(self, text_or_lines: str | List[str]) -> Dict[str, float]:
        """Extract numerical metrics (latency_ms, rssi_dbm, packet_loss_pct, etc.) from log text."""
        combined_text = "\n".join(text_or_lines) if isinstance(text_or_lines, list) else text_or_lines
        metrics: Dict[str, float] = {}

        for match in self.METRIC_PATTERN.finditer(combined_text):
            k, v = match.group(1).lower(), match.group(2)
            # Filter standard telemetry keys
            if any(metric_key in k for metric_key in ["latency", "rtt", "rssi", "snr", "loss", "packet", "retry", "count", "elapsed"]):
                try:
                    metrics[k] = float(v)
                except ValueError:
                    continue

        return metrics

    def extract_error_info(self, raw_logs: List[str], error_trace: Optional[str] = None) -> Dict[str, Optional[str]]:
        """Identify exception type, error message, and probable failure category."""
        target_text = (error_trace or "") + "\n" + "\n".join(raw_logs)

        error_type: Optional[str] = None
        error_message: Optional[str] = None

        # Look for explicit exception matches
        for line in reversed(target_text.splitlines()):
            line_strip = line.strip()
            match = self.EXCEPTION_PATTERN.search(line_strip)
            if match:
                error_type = match.group(1).strip()
                error_message = match.group(2).strip()
                break

        # Fallback search in logs if not formatted as Exception: message
        if not error_type:
            for line in raw_logs:
                line_lower = line.lower()
                if "error" in line_lower or "critical" in line_lower or "failure" in line_lower:
                    error_message = line.strip()
                    break

        category = FailureCategory.NONE
        if error_type and error_type in self.EXCEPTION_CATEGORY_MAP:
            category = self.EXCEPTION_CATEGORY_MAP[error_type]

        return {
            "error_type": error_type,
            "error_message": error_message,
            "category": category,
        }

    def parse_test_result(
        self,
        test_id: str,
        test_name: str,
        status: str,
        duration_ms: float = 0.0,
        raw_logs: Optional[List[str]] = None,
        stack_trace: Optional[str] = None,
        explicit_category: Optional[FailureCategory] = None,
        explicit_error_type: Optional[str] = None,
        explicit_error_message: Optional[str] = None,
    ) -> TestCaseResult:
        """Construct a structured TestCaseResult from execution artifacts."""
        logs = raw_logs or []
        metrics = self.extract_metrics(logs)
        error_info = self.extract_error_info(logs, stack_trace)

        final_category = explicit_category or error_info["category"]
        final_error_type = explicit_error_type or error_info["error_type"]
        final_error_message = explicit_error_message or error_info["error_message"]

        # Ensure passing tests do not carry failure categories
        parsed_status = TestStatus(status.upper())
        if parsed_status == TestStatus.PASSED:
            final_category = FailureCategory.NONE
            final_error_type = None
            final_error_message = None

        return TestCaseResult(
            test_id=test_id,
            test_name=test_name,
            status=parsed_status,
            failure_category=final_category,
            duration_ms=round(duration_ms, 2),
            error_type=final_error_type,
            error_message=final_error_message,
            stack_trace=stack_trace,
            metrics=metrics,
            raw_logs=logs,
            timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

    def results_to_dataframe(self, results: List[TestCaseResult]) -> pd.DataFrame:
        """Convert a list of TestCaseResult into a clean DataFrame ready for analysis and ML."""
        records = []
        for r in results:
            # Combine logs and error messages into a clean consolidated textual feature
            text_body = " ".join([self.parse_log_line(log)["message"] for log in r.raw_logs])
            if r.error_message:
                text_body += f" [ERROR] {r.error_message}"
            if r.error_type:
                text_body += f" [TYPE] {r.error_type}"

            record = {
                "test_id": r.test_id,
                "test_name": r.test_name,
                "status": r.status.value,
                "failure_category": r.failure_category.value if r.failure_category else "none",
                "duration_ms": r.duration_ms,
                "error_type": r.error_type or "",
                "error_message": r.error_message or "",
                "latency_ms": r.metrics.get("latency_ms", None),
                "rssi_dbm": r.metrics.get("rssi_dbm", None),
                "packet_loss_pct": r.metrics.get("packet_loss_pct", None),
                "snr_db": r.metrics.get("snr_db", None),
                "log_count": len(r.raw_logs),
                "clean_log_text": text_body.strip(),
                "timestamp": r.timestamp,
            }
            records.append(record)

        return pd.DataFrame(records)

    def parse_pytest_json_or_output(self, pytest_lines: List[str]) -> List[TestCaseResult]:
        """Extract test outcomes and traceback blocks from raw pytest CLI output."""
        results: List[TestCaseResult] = []
        current_test_name = None
        current_logs: List[str] = []
        status = "PASSED"

        for line in pytest_lines:
            # Detect pytest progress line e.g. tests/test_wireless_suite.py::test_basic_connection PASSED
            pass_fail_match = re.search(r"::(test_\w+)\s+(PASSED|FAILED|SKIPPED|ERROR)", line)
            if pass_fail_match:
                test_name = pass_fail_match.group(1)
                outcome = pass_fail_match.group(2)
                results.append(
                    self.parse_test_result(
                        test_id=f"RUN-{test_name}",
                        test_name=test_name,
                        status=outcome,
                        duration_ms=10.0,
                        raw_logs=[line],
                    )
                )

        return results
