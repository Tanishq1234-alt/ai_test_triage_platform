"""Log parser and structured failure normalization package."""

import os
from typing import List, Optional

from src.log_parser.exporter import RecordExporter
from src.log_parser.junit_parser import JUnitParser
from src.log_parser.normalizer import NormalizedTestRecord, RuleBasedCategorizer, TestResultNormalizer
from src.log_parser.text_parser import PlainTextLogParser


def correlate_and_normalize(
    junit_path: Optional[str] = None,
    log_path: Optional[str] = None,
    context_lines_before: int = 3,
    context_lines_after: int = 3,
) -> List[NormalizedTestRecord]:
    """Correlates JUnit XML reports and plain-text application logs into a single enriched set of NormalizedTestRecords."""
    normalizer = TestResultNormalizer()
    junit_parser = JUnitParser(normalizer=normalizer)
    text_parser = PlainTextLogParser(
        context_lines_before=context_lines_before,
        context_lines_after=context_lines_after,
        normalizer=normalizer,
    )

    junit_records: List[NormalizedTestRecord] = []
    if junit_path and os.path.exists(junit_path):
        junit_records = junit_parser.parse_file(junit_path)

    log_blocks = {}
    if log_path and os.path.exists(log_path):
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                log_content = f.read()
            log_blocks = text_parser.partition_logs_by_test(log_content)
        except Exception:
            log_blocks = {}

    if not junit_records and log_path and os.path.exists(log_path):
        # Fall back to pure text log parsing if no JUnit XML is provided
        return text_parser.parse_file(log_path)

    if not log_blocks:
        return junit_records

    # Enrich JUnit records with contextual logs and precise timestamps from text logs
    enriched_records: List[NormalizedTestRecord] = []
    for rec in junit_records:
        matched_block = None
        for block_name, block_info in log_blocks.items():
            if block_name == rec.test_name or block_name in rec.test_id or rec.test_name in block_name:
                matched_block = block_info
                break

        if matched_block:
            lines = matched_block["lines"]
            # Extract error indices
            err_indices = [
                idx for idx, line in enumerate(lines)
                if text_parser.is_actual_error_line(line)
            ]
            contextual_excerpt = text_parser.extract_context_window(lines, err_indices)

            # Discover first timestamp from log
            first_ts = rec.timestamp
            for line in lines:
                parsed_line = text_parser.parse_log_line(line)
                if parsed_line["timestamp"]:
                    first_ts = parsed_line["timestamp"]
                    break

            # If log had explicit failure category banner
            explicit_cat = matched_block.get("category")
            if explicit_cat == "none":
                explicit_cat = None

            # Re-normalize with enriched log excerpt and timestamp
            enriched = normalizer.normalize(
                test_id=rec.test_id,
                test_name=rec.test_name,
                test_suite=rec.test_suite,
                timestamp=first_ts,
                status=rec.status,
                duration_seconds=rec.duration_seconds,
                error_type=rec.error_type,
                error_message=rec.error_message,
                stack_trace=rec.stack_trace,
                log_excerpt=contextual_excerpt or rec.log_excerpt,
                source_file=rec.source_file,
                explicit_category=explicit_cat or rec.failure_category,
            )
            enriched_records.append(enriched)
        else:
            enriched_records.append(rec)

    return enriched_records


__all__ = [
    "NormalizedTestRecord",
    "RuleBasedCategorizer",
    "TestResultNormalizer",
    "JUnitParser",
    "PlainTextLogParser",
    "RecordExporter",
    "correlate_and_normalize",
]
