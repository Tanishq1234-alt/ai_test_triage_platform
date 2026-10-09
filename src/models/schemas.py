"""Data models and schemas for test results, logs, and triage classification."""

from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class TestStatus(str, Enum):
    PASSED = "PASSED"
    FAILED = "FAILED"
    ERROR = "ERROR"
    SKIPPED = "SKIPPED"


class FailureCategory(str, Enum):
    NONE = "none"
    TIMEOUT = "timeout"
    ASSERTION = "assertion"
    CONFIGURATION = "configuration"
    CONNECTION = "connection"
    DEPENDENCY = "dependency"


class TestCaseResult(BaseModel):
    test_id: str
    test_name: str
    status: TestStatus
    failure_category: Optional[FailureCategory] = None
    duration_ms: float = 0.0
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    stack_trace: Optional[str] = None
    metrics: Dict[str, float] = Field(default_factory=dict)
    raw_logs: List[str] = Field(default_factory=list)
    timestamp: str


class TestSessionSummary(BaseModel):
    session_id: str
    timestamp: str
    total_tests: int
    passed_count: int
    failed_count: int
    duration_seconds: float
    results: List[TestCaseResult] = Field(default_factory=list)


class FailureClassificationResult(BaseModel):
    test_id: str
    test_name: str
    predicted_category: FailureCategory
    confidence: float
    probabilities: Dict[str, float] = Field(default_factory=dict)
    model_type: str = "ml"  # "ml" or "baseline_rule"
    log_evidence: List[str] = Field(default_factory=list)
    likely_cause: str = ""
    suggested_steps: List[str] = Field(default_factory=list)
