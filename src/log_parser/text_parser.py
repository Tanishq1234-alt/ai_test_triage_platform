"""Plain-text log parser extracting timestamps, log levels, exceptions, and context windows."""

import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from src.log_parser.normalizer import NormalizedTestRecord, TestResultNormalizer

logger = logging.getLogger(__name__)


class PlainTextLogParser:
    """Parses plain-text test logs, extracting structured events and preserving contextual error excerpts."""

    TIMESTAMP_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:[+-]\d{2}:?\d{2}|Z)?)")
    LEVEL_PATTERN = re.compile(r"\[(DEBUG|INFO|NOTICE|WARN|WARNING|ERROR|CRITICAL|FATAL)\]")
    SUBSYSTEM_PATTERN = re.compile(r"\[([A-Z0-9_\-]+)\]")
    TEST_START_PATTERN = re.compile(r"(?:=== TEST:\s*(\w+)|Starting test case:\s*(\w+)|Running test:\s*(\w+)|::(test_\w+))", re.IGNORECASE)
    TEST_END_BANNER = re.compile(r"=== TEST:\s*(\w+)\s*\[(\w+)\](?:\s*\(category:\s*(\w+)\))?\s*===", re.IGNORECASE)
    EXCEPTION_LINE_PATTERN = re.compile(r"^([A-Za-z0-9_]+(?:Error|Exception))\s*:\s*(.+)$")
    TRACEBACK_START = re.compile(r"^Traceback \(most recent call last\):", re.IGNORECASE)

    def __init__(
        self,
        context_lines_before: int = 3,
        context_lines_after: int = 3,
        normalizer: Optional[TestResultNormalizer] = None,
    ):
        self.context_lines_before = context_lines_before
        self.context_lines_after = context_lines_after
        self.normalizer = normalizer or TestResultNormalizer()

    def parse_file(self, file_path: str) -> List[NormalizedTestRecord]:
        """Parse a plain-text log file from disk into NormalizedTestRecords."""
        if not os.path.exists(file_path):
            logger.warning(f"Log file not found at path: {file_path}")
            return []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            logger.warning(f"Failed to read log file '{file_path}': {e}")
            return []

        if not content.strip():
            logger.warning(f"Log file is empty: {file_path}")
            return []

        return self.parse_text(content, source_file=file_path)

    def parse_log_line(self, line: str) -> Dict[str, Any]:
        """Extract timestamp, severity level, subsystem, and clean message from a log line."""
        clean = line.strip()
        ts_match = self.TIMESTAMP_PATTERN.search(clean)
        timestamp = ts_match.group(1) if ts_match else None

        lvl_match = self.LEVEL_PATTERN.search(clean)
        level = lvl_match.group(1).upper() if lvl_match else "INFO"
        if level == "WARN":
            level = "WARNING"

        subsystems = self.SUBSYSTEM_PATTERN.findall(clean)
        # Avoid treating level as subsystem if both matched brackets
        subsystem = [s for s in subsystems if s not in ["DEBUG", "INFO", "WARN", "WARNING", "ERROR", "CRITICAL", "FATAL", "TEST"]]
        subsystem_str = subsystem[0] if subsystem else "GENERAL"

        return {
            "timestamp": timestamp,
            "level": level,
            "subsystem": subsystem_str,
            "raw": clean,
        }

    def is_actual_error_line(self, line: str) -> bool:
        """Determines if a line is an actual error event.

        CRITICAL REQUIREMENT: Does not treat every line containing the word 'error' as a failure.
        Only lines with level ERROR/CRITICAL/FATAL, exception class signatures, or traceback indicators are counted.
        """
        line_strip = line.strip()
        parsed = self.parse_log_line(line_strip)

        # High severity level is an explicit error
        if parsed["level"] in ["ERROR", "CRITICAL", "FATAL"]:
            return True

        # Traceback starts
        if self.TRACEBACK_START.match(line_strip):
            return True

        # Explicit exception signature e.g. "WirelessTimeoutError: Connection handshake..."
        if self.EXCEPTION_LINE_PATTERN.match(line_strip):
            return True

        # Explicit test failure statement
        if line_strip.startswith("ERROR:") or line_strip.startswith("Test FAILED"):
            return True

        return False

    def extract_context_window(self, lines: List[str], error_indices: List[int]) -> Optional[str]:
        """Extracts a consolidated contextual window of lines around detected error events."""
        if not error_indices:
            return None

        total = len(lines)
        selected_line_indices = set()

        for idx in error_indices:
            start = max(0, idx - self.context_lines_before)
            end = min(total, idx + self.context_lines_after + 1)
            for i in range(start, end):
                selected_line_indices.add(i)

        sorted_indices = sorted(selected_line_indices)
        excerpt_lines = []
        last_idx = None

        for idx in sorted_indices:
            if last_idx is not None and idx > last_idx + 1:
                excerpt_lines.append("--- [...] ---")
            excerpt_lines.append(lines[idx].rstrip())
            last_idx = idx

        return "\n".join(excerpt_lines) if excerpt_lines else None

    def partition_logs_by_test(self, content: str) -> Dict[str, Dict[str, Any]]:
        """Splits log contents into per-test segments associating logs with test cases."""
        if not content or not content.strip():
            return {}

        lines = content.splitlines()
        test_blocks: Dict[str, Dict[str, Any]] = {}
        current_test: Optional[str] = None
        current_lines: List[str] = []
        explicit_category: Optional[str] = None
        explicit_status: Optional[str] = None

        def finalize_current():
            nonlocal current_test, current_lines, explicit_category, explicit_status
            if current_test:
                test_blocks[current_test] = {
                    "lines": list(current_lines),
                    "category": explicit_category,
                    "status": explicit_status,
                }
            current_lines = []
            explicit_category = None
            explicit_status = None

        for line in lines:
            line_str = line.strip()
            # Check for header banner: === TEST: <name> [<status>] (category: <cat>) ===
            banner_match = self.TEST_END_BANNER.match(line_str)
            if banner_match:
                finalize_current()
                current_test = banner_match.group(1)
                explicit_status = banner_match.group(2).lower()
                explicit_category = banner_match.group(3)
                continue

            # Check for test start log: Starting test case: <name>
            start_match = self.TEST_START_PATTERN.search(line_str)
            if start_match:
                matched_name = next(g for g in start_match.groups() if g is not None)
                if current_test != matched_name:
                    finalize_current()
                    current_test = matched_name

            if current_test:
                current_lines.append(line)

        finalize_current()

        # If no explicit banners were found, treat whole log as a single execution
        if not test_blocks and lines:
            test_blocks["test_execution"] = {
                "lines": lines,
                "category": None,
                "status": None,
            }

        return test_blocks

    def parse_text(self, content: str, source_file: Optional[str] = None) -> List[NormalizedTestRecord]:
        """Convert full text log stream into NormalizedTestRecords."""
        if not content or not content.strip():
            return []

        test_blocks = self.partition_logs_by_test(content)
        records: List[NormalizedTestRecord] = []

        for test_name, block_info in test_blocks.items():
            lines = block_info["lines"]
            explicit_status = block_info.get("status")
            explicit_cat = block_info.get("category")

            # Discover timestamps and error indices
            timestamps: List[str] = []
            error_indices: List[int] = []
            error_type: Optional[str] = None
            error_message: Optional[str] = None
            stack_trace_lines: List[str] = []
            in_traceback = False

            for idx, line in enumerate(lines):
                parsed = self.parse_log_line(line)
                if parsed["timestamp"]:
                    timestamps.append(parsed["timestamp"])

                if self.TRACEBACK_START.match(line.strip()):
                    in_traceback = True
                    error_indices.append(idx)
                    stack_trace_lines.append(line)
                    continue

                if in_traceback:
                    stack_trace_lines.append(line)
                    error_indices.append(idx)
                    exc_match = self.EXCEPTION_LINE_PATTERN.match(line.strip())
                    if exc_match:
                        error_type = exc_match.group(1)
                        error_message = exc_match.group(2)
                        in_traceback = False
                    continue

                if self.is_actual_error_line(line):
                    error_indices.append(idx)
                    exc_match = self.EXCEPTION_LINE_PATTERN.search(line.strip())
                    if exc_match and not error_type:
                        error_type = exc_match.group(1)
                        error_message = exc_match.group(2)
                    elif not error_message and ("ERROR:" in line or "[ERROR]" in line or "[CRITICAL]" in line):
                        # Extract message portion without timestamp colons
                        if "ERROR:" in line:
                            error_message = line.split("ERROR:", 1)[1].strip()
                        else:
                            last_bracket = line.rfind("]")
                            if last_bracket != -1:
                                error_message = line[last_bracket + 1:].strip()
                            else:
                                error_message = line.strip()

            # Determine status
            status = explicit_status or ("failed" if error_indices else "passed")
            stack_trace = "\n".join(stack_trace_lines) if stack_trace_lines else None
            log_excerpt = self.extract_context_window(lines, error_indices)
            first_timestamp = timestamps[0] if timestamps else None

            rec = self.normalizer.normalize(
                test_id=f"LOG::{test_name}",
                test_name=test_name,
                test_suite="plain_text_suite",
                timestamp=first_timestamp,
                status=status,
                duration_seconds=None,
                error_type=error_type,
                error_message=error_message,
                stack_trace=stack_trace,
                log_excerpt=log_excerpt,
                source_file=source_file,
                explicit_category=explicit_cat if explicit_cat != "none" else None,
            )
            records.append(rec)

        return records
