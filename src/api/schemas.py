"""Pydantic schemas and data transfer objects for the FastAPI REST interface."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    """System health check and component status response."""
    status: str = Field(..., description="Overall service status (healthy/degraded)")
    version: str = Field(..., description="Application version")
    database_connected: bool = Field(..., description="SQLite connection state")
    model_loaded: bool = Field(..., description="Whether ML classifier pipeline is active")
    timestamp: str = Field(..., description="Current server UTC timestamp")


class ModelStatusResponse(BaseModel):
    """Machine learning model status, parameters, and evaluation metadata."""
    model_name: str
    model_version: str
    classes: List[str]
    confidence_threshold: float
    is_loaded: bool
    evaluation_metrics: Optional[Dict[str, Any]] = None


class TestRecordResponse(BaseModel):
    """Normalized test execution record summary."""
    model_config = ConfigDict(extra="ignore")

    test_id: str
    test_name: str
    test_suite: Optional[str] = None
    timestamp: Optional[str] = None
    status: str
    duration_seconds: Optional[float] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    stack_trace: Optional[str] = None
    log_excerpt: Optional[str] = None
    failure_category: Optional[str] = None
    source_file: Optional[str] = None
    created_at: Optional[str] = None


class MLPredictionDetail(BaseModel):
    """Details of ML inference classification."""
    id: Optional[int] = None
    test_id: str
    predicted_category: str
    raw_predicted_category: Optional[str] = None
    confidence: float
    probabilities: Dict[str, float] = Field(default_factory=dict)
    model_version: str
    needs_review: bool
    review_reason: Optional[str] = None
    top_features: List[str] = Field(default_factory=list)
    rule_based_category: Optional[str] = None
    is_agreement: bool = True
    created_at: Optional[str] = None


class HumanReviewDetail(BaseModel):
    """Human review override record."""
    id: int
    test_id: str
    original_predicted_category: str
    corrected_category: str
    reviewer_name: Optional[str] = None
    notes: Optional[str] = None
    created_at: Optional[str] = None


class TestRecordDetailResponse(TestRecordResponse):
    """Complete test record with associated ML prediction and review audit logs."""
    latest_prediction: Optional[Dict[str, Any]] = None
    human_reviews: List[Dict[str, Any]] = Field(default_factory=list)


class PaginatedTestRecordsResponse(BaseModel):
    """Paginated collection of test records."""
    items: List[TestRecordResponse]
    total: int
    limit: int
    offset: int


class SummaryStatisticsResponse(BaseModel):
    """Historical test metrics and category breakdown."""
    total_tests: int
    passed_tests: int
    failed_tests: int
    skipped_tests: int
    pass_rate_pct: float
    avg_duration_seconds: float
    max_duration_seconds: float
    category_distribution: Dict[str, int]
    human_reviews_count: int


class DailyTrendItem(BaseModel):
    """Daily test counts and breakdown."""
    date: str
    total: int
    passed: int
    failed: int
    skipped: int


class DailyTrendsResponse(BaseModel):
    """Historical timeline trends."""
    trends: List[DailyTrendItem]


class FailureAnalyzeRequest(BaseModel):
    """Input payload for root cause triage analysis."""
    test_id: Optional[str] = None
    test_name: Optional[str] = "unnamed_test"
    test_suite: Optional[str] = "wireless_validation_suite"
    timestamp: Optional[str] = None
    status: Optional[str] = "failed"
    duration_seconds: Optional[float] = 0.0
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    stack_trace: Optional[str] = None
    log_excerpt: Optional[str] = None
    source_file: Optional[str] = None
    log_text: Optional[str] = Field(None, description="Optional raw log excerpt to parse")
    save_to_db: bool = Field(True, description="Persist the record and triage result to SQLite")


class FailureAnalyzeResponse(BaseModel):
    """Comprehensive triage analysis and recommendation response."""
    test_id: str
    test_name: str
    normalized_record: Dict[str, Any]
    rule_based_category: str
    ml_prediction: Dict[str, Any]
    predicted_category: str
    confidence: float
    model_version: str
    needs_review: bool
    review_reason: Optional[str] = None
    top_features: List[str] = Field(default_factory=list)
    suggested_actions: List[str] = Field(default_factory=list)
    evidence: List[str] = Field(default_factory=list)


class HumanReviewRequest(BaseModel):
    """Human-in-the-loop review correction payload."""
    corrected_category: str = Field(..., description="Canonical category override")
    reviewer_name: str = Field("test_engineer", description="Name or identifier of reviewer")
    notes: Optional[str] = Field("", description="Triage investigation notes")


class HumanReviewResponse(BaseModel):
    """Response confirming human review persistence."""
    success: bool
    review_id: int
    test_id: str
    original_predicted_category: str
    corrected_category: str
    reviewer_name: str
    message: str


class BatchImportRequest(BaseModel):
    """Batch import of normalized test records."""
    records: List[Dict[str, Any]]


class BatchImportResponse(BaseModel):
    """Summary of batch import outcome."""
    inserted: int
    skipped: int
    total: int
