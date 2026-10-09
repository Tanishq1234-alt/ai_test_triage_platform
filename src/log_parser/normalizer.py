"""Common schema, normalization, and deterministic rule-based failure categorization."""

import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class NormalizedTestRecord(BaseModel):
    """Normalized test execution record conforming to Phase 2 failure triage schema."""

    model_config = ConfigDict(populate_by_name=True)

    test_id: str
    test_name: str
    test_suite: Optional[str] = None
    timestamp: Optional[str] = None
    status: str  # "passed", "failed", "error", "skipped"
    duration_seconds: Optional[float] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    stack_trace: Optional[str] = None
    log_excerpt: Optional[str] = None
    failure_category: Optional[str] = None  # timeout, assertion, connection, configuration, dependency, unknown, or null
    source_file: Optional[str] = None


class RuleBasedCategorizer:
    """Deterministic rule-based failure classifier for obvious error patterns."""

    TIMEOUT_PATTERNS = [
        re.compile(r"timeout", re.IGNORECASE),
        re.compile(r"timed\s+out", re.IGNORECASE),
        re.compile(r"deadline\s+exceeded", re.IGNORECASE),
        re.compile(r"timer\s+expired", re.IGNORECASE),
        re.compile(r"wait\s+timed\s+out", re.IGNORECASE),
    ]

    ASSERTION_PATTERNS = [
        re.compile(r"assertionerror", re.IGNORECASE),
        re.compile(r"wirelessassertionerror", re.IGNORECASE),
        re.compile(r"\bassert\b", re.IGNORECASE),
        re.compile(r"sla\s+violation", re.IGNORECASE),
        re.compile(r"threshold\s+breach", re.IGNORECASE),
        re.compile(r"exceeds\s+maximum", re.IGNORECASE),
        re.compile(r"exceeds\s+limit", re.IGNORECASE),
        re.compile(r"below\s+minimum", re.IGNORECASE),
    ]

    CONNECTION_PATTERNS = [
        re.compile(r"connectionerror", re.IGNORECASE),
        re.compile(r"wirelessconnectionerror", re.IGNORECASE),
        re.compile(r"connectionreset", re.IGNORECASE),
        re.compile(r"connectionrefused", re.IGNORECASE),
        re.compile(r"link\s+lost", re.IGNORECASE),
        re.compile(r"link\s+failure", re.IGNORECASE),
        re.compile(r"beacon\s+frame\s+loss", re.IGNORECASE),
        re.compile(r"carrier\s+drop", re.IGNORECASE),
        re.compile(r"carrier\s+wave\s+degraded", re.IGNORECASE),
        re.compile(r"disassociated", re.IGNORECASE),
        re.compile(r"radio\s+link", re.IGNORECASE),
    ]

    CONFIGURATION_PATTERNS = [
        re.compile(r"configerror", re.IGNORECASE),
        re.compile(r"wirelessconfigerror", re.IGNORECASE),
        re.compile(r"configuration\s+reject", re.IGNORECASE),
        re.compile(r"invalid\s+ssid", re.IGNORECASE),
        re.compile(r"authentication\s+key", re.IGNORECASE),
        re.compile(r"invalid\s+wpa", re.IGNORECASE),
        re.compile(r"mismatched\s+cipher", re.IGNORECASE),
        re.compile(r"keyerror", re.IGNORECASE),
        re.compile(r"valueerror", re.IGNORECASE),
    ]

    DEPENDENCY_PATTERNS = [
        re.compile(r"dependencyerror", re.IGNORECASE),
        re.compile(r"wirelessdependencyerror", re.IGNORECASE),
        re.compile(r"hal\s+driver", re.IGNORECASE),
        re.compile(r"hardware\s+driver", re.IGNORECASE),
        re.compile(r"kernel\s+ring", re.IGNORECASE),
        re.compile(r"ring\s+buffer\s+corrupted", re.IGNORECASE),
        re.compile(r"dma\s+tx\s+queue", re.IGNORECASE),
        re.compile(r"driver\s+hang", re.IGNORECASE),
        re.compile(r"sensor\s+bus\s+error", re.IGNORECASE),
        re.compile(r"modulenotfound", re.IGNORECASE),
        re.compile(r"filenotfound", re.IGNORECASE),
        re.compile(r"oserror", re.IGNORECASE),
        re.compile(r"unresponsive", re.IGNORECASE),
    ]

    @classmethod
    def categorize(
        cls,
        status: str,
        error_type: Optional[str] = None,
        error_message: Optional[str] = None,
        stack_trace: Optional[str] = None,
        log_excerpt: Optional[str] = None,
    ) -> Optional[str]:
        """Categorize failure deterministically into timeout, assertion, connection, configuration, dependency, or unknown.

        Returns None (null) for passed or skipped tests.
        """
        normalized_status = status.lower().strip()
        if normalized_status in ["passed", "skipped"]:
            return None

        # Build composite inspection text, weighting explicit error_type and error_message first
        type_and_msg = f"{error_type or ''} {error_message or ''}".strip()
        full_text = f"{type_and_msg}\n{stack_trace or ''}\n{log_excerpt or ''}".strip()

        if not full_text:
            return "unknown"

        # Check explicit error type matches first for precision
        if error_type:
            e_lower = error_type.lower()
            if "timeout" in e_lower:
                return "timeout"
            if "assertion" in e_lower:
                return "assertion"
            if "config" in e_lower:
                return "configuration"
            if "connection" in e_lower:
                return "connection"
            if "dependency" in e_lower:
                return "dependency"

        # Evaluate pattern sets on type_and_msg first, then full_text
        for category, pattern_list in [
            ("timeout", cls.TIMEOUT_PATTERNS),
            ("configuration", cls.CONFIGURATION_PATTERNS),
            ("connection", cls.CONNECTION_PATTERNS),
            ("dependency", cls.DEPENDENCY_PATTERNS),
            ("assertion", cls.ASSERTION_PATTERNS),
        ]:
            if any(p.search(type_and_msg) for p in pattern_list):
                return category

        for category, pattern_list in [
            ("timeout", cls.TIMEOUT_PATTERNS),
            ("configuration", cls.CONFIGURATION_PATTERNS),
            ("connection", cls.CONNECTION_PATTERNS),
            ("dependency", cls.DEPENDENCY_PATTERNS),
            ("assertion", cls.ASSERTION_PATTERNS),
        ]:
            if any(p.search(full_text) for p in pattern_list):
                return category

        return "unknown"


class TestResultNormalizer:
    """Normalizes raw parsed data from JUnit XML and plain-text logs into NormalizedTestRecord objects."""

    def __init__(self, categorizer: Optional[RuleBasedCategorizer] = None):
        self.categorizer = categorizer or RuleBasedCategorizer()

    def normalize(
        self,
        test_id: str,
        test_name: str,
        status: str,
        test_suite: Optional[str] = None,
        timestamp: Optional[str] = None,
        duration_seconds: Optional[float] = None,
        error_type: Optional[str] = None,
        error_message: Optional[str] = None,
        stack_trace: Optional[str] = None,
        log_excerpt: Optional[str] = None,
        source_file: Optional[str] = None,
        explicit_category: Optional[str] = None,
    ) -> NormalizedTestRecord:
        """Create a validated NormalizedTestRecord, determining failure_category rule-based when appropriate."""
        norm_status = status.lower().strip()
        if norm_status not in ["passed", "failed", "error", "skipped"]:
            # Fallback normalization
            if "pass" in norm_status:
                norm_status = "passed"
            elif "fail" in norm_status:
                norm_status = "failed"
            elif "err" in norm_status:
                norm_status = "error"
            elif "skip" in norm_status:
                norm_status = "skipped"
            else:
                norm_status = "error"

        # Round duration if present
        norm_duration = round(float(duration_seconds), 4) if duration_seconds is not None else None

        # Clean strings, mapping empty strings to None (null)
        clean_err_type = error_type.strip() if error_type and error_type.strip() else None
        clean_err_msg = error_message.strip() if error_message and error_message.strip() else None
        clean_trace = stack_trace.strip() if stack_trace and stack_trace.strip() else None
        clean_excerpt = log_excerpt.strip() if log_excerpt and log_excerpt.strip() else None
        clean_source = source_file.strip() if source_file and source_file.strip() else None
        clean_suite = test_suite.strip() if test_suite and test_suite.strip() else None
        clean_timestamp = timestamp.strip() if timestamp and timestamp.strip() else None

        # Determine failure category
        if norm_status == "passed":
            category = None
            clean_err_type = None
            clean_err_msg = None
            clean_trace = None
        elif norm_status == "skipped":
            category = None
            clean_err_type = None
            clean_trace = None
            # Preserve clean_err_msg as the skip rationale
        elif explicit_category:
            category = explicit_category.lower().strip()
        else:
            category = self.categorizer.categorize(
                status=norm_status,
                error_type=clean_err_type,
                error_message=clean_err_msg,
                stack_trace=clean_trace,
                log_excerpt=clean_excerpt,
            )

        return NormalizedTestRecord(
            test_id=test_id,
            test_name=test_name,
            test_suite=clean_suite,
            timestamp=clean_timestamp,
            status=norm_status,
            duration_seconds=norm_duration,
            error_type=clean_err_type,
            error_message=clean_err_msg,
            stack_trace=clean_trace,
            log_excerpt=clean_excerpt,
            failure_category=category,
            source_file=clean_source,
        )
