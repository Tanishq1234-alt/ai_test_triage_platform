"""JUnit XML parser extracting test cases, durations, failure details, and traces."""

import logging
import os
import xml.etree.ElementTree as ET
from typing import List, Optional

from src.log_parser.normalizer import NormalizedTestRecord, TestResultNormalizer

logger = logging.getLogger(__name__)


class JUnitParser:
    """Robust parser for Pytest and standard JUnit XML test report files."""

    def __init__(self, normalizer: Optional[TestResultNormalizer] = None):
        self.normalizer = normalizer or TestResultNormalizer()

    def parse_file(self, file_path: str) -> List[NormalizedTestRecord]:
        """Parse a JUnit XML file on disk. Gracefully handles missing, empty, or malformed files."""
        if not os.path.exists(file_path):
            logger.warning(f"JUnit XML file not found at path: {file_path}")
            return []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
        except Exception as e:
            logger.warning(f"Failed to read JUnit XML file '{file_path}': {e}")
            return []

        if not content:
            logger.warning(f"JUnit XML file is empty: {file_path}")
            return []

        return self.parse_string(content, source_file=file_path)

    def parse_string(self, xml_content: str, source_file: Optional[str] = None) -> List[NormalizedTestRecord]:
        """Parse XML string content into NormalizedTestRecord instances."""
        if not xml_content or not xml_content.strip():
            return []

        try:
            root = ET.fromstring(xml_content.strip())
        except ET.ParseError as e:
            logger.warning(f"Malformed JUnit XML content: {e}")
            return []

        records: List[NormalizedTestRecord] = []

        # Find all testsuite nodes
        if root.tag == "testsuite":
            suites = [root]
        elif root.tag == "testsuites":
            suites = root.findall("testsuite")
            if not suites:
                suites = [root]
        else:
            suites = root.findall(".//testsuite")
            if not suites:
                suites = [root]

        for suite in suites:
            suite_name = suite.get("name") or "default_suite"
            suite_timestamp = suite.get("timestamp")

            for case in suite.findall(".//testcase") if suite.tag == "testsuites" else suite.findall("testcase"):
                case_name = case.get("name") or "unnamed_test"
                classname = case.get("classname")
                case_file = case.get("file") or (classname.replace(".", "/") + ".py" if classname else source_file)

                # Duration
                duration_val = case.get("time")
                duration_s: Optional[float] = None
                if duration_val is not None:
                    try:
                        duration_s = float(duration_val)
                    except ValueError:
                        duration_s = None

                status = "passed"
                error_type: Optional[str] = None
                error_message: Optional[str] = None
                stack_trace: Optional[str] = None
                log_excerpt: Optional[str] = None

                # Check for failure, error, or skipped children
                failure_node = case.find("failure")
                error_node = case.find("error")
                skipped_node = case.find("skipped")

                if failure_node is not None:
                    status = "failed"
                    error_message = failure_node.get("message")
                    error_type = failure_node.get("type")
                    stack_trace = failure_node.text.strip() if failure_node.text else None
                    if not error_message and stack_trace:
                        first_line = stack_trace.splitlines()[-1] if stack_trace else ""
                        error_message = first_line.strip()
                elif error_node is not None:
                    status = "error"
                    error_message = error_node.get("message")
                    error_type = error_node.get("type")
                    stack_trace = error_node.text.strip() if error_node.text else None
                    if not error_message and stack_trace:
                        first_line = stack_trace.splitlines()[-1] if stack_trace else ""
                        error_message = first_line.strip()
                elif skipped_node is not None:
                    status = "skipped"
                    error_message = skipped_node.get("message") or (skipped_node.text.strip() if skipped_node.text else None)

                # Check for stdout / stderr inside testcase
                sys_out = case.find("system-out")
                sys_err = case.find("system-err")
                output_parts = []
                if sys_out is not None and sys_out.text:
                    output_parts.append(sys_out.text.strip())
                if sys_err is not None and sys_err.text:
                    output_parts.append(sys_err.text.strip())
                if output_parts:
                    log_excerpt = "\n".join(output_parts)

                # Construct test_id
                test_id = f"{classname}::{case_name}" if classname else f"{suite_name}::{case_name}"

                record = self.normalizer.normalize(
                    test_id=test_id,
                    test_name=case_name,
                    test_suite=suite_name,
                    timestamp=suite_timestamp,
                    status=status,
                    duration_seconds=duration_s,
                    error_type=error_type,
                    error_message=error_message,
                    stack_trace=stack_trace,
                    log_excerpt=log_excerpt,
                    source_file=case_file,
                )
                records.append(record)

        return records
