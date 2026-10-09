"""Health check and ML model status routes."""

import datetime
from typing import Any, Dict
from fastapi import APIRouter, Depends

from src.api.schemas import HealthResponse, ModelStatusResponse
from src.ml.dataset import CANONICAL_CATEGORIES
from src.services.analysis_service import AnalysisService
from src.services.report_service import ReportService
from src.storage.database import Database

router = APIRouter(tags=["Health & Status"])


def get_analysis_service() -> AnalysisService:
    return AnalysisService()


def get_report_service() -> ReportService:
    return ReportService()


def get_database() -> Database:
    return Database()


@router.get("/health", response_model=HealthResponse)
def health_check(
    db: Database = Depends(get_database),
    analysis: AnalysisService = Depends(get_analysis_service),
) -> HealthResponse:
    """Verify backend health, database connectivity, and ML model readiness."""
    db_connected = False
    try:
        with db.transaction() as conn:
            cur = conn.execute("SELECT 1;")
            db_connected = cur.fetchone()[0] == 1
    except Exception:
        db_connected = False

    model_loaded = analysis.predictor is not None
    status = "healthy" if (db_connected and model_loaded) else ("degraded" if db_connected else "unhealthy")

    return HealthResponse(
        status=status,
        version="1.0.0",
        database_connected=db_connected,
        model_loaded=model_loaded,
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )


@router.get("/api/model/status", response_model=ModelStatusResponse)
def model_status(
    analysis: AnalysisService = Depends(get_analysis_service),
    report: ReportService = Depends(get_report_service),
) -> ModelStatusResponse:
    """Return loaded ML model metadata, supported classes, thresholds, and evaluation report."""
    predictor = analysis.predictor
    eval_metrics = report.get_model_evaluation_metrics()

    if predictor is not None:
        return ModelStatusResponse(
            model_name=predictor.metadata.get("model_name", "LogisticRegression"),
            model_version=predictor.model_version,
            classes=predictor.classes_,
            confidence_threshold=predictor.confidence_threshold,
            is_loaded=True,
            evaluation_metrics=eval_metrics,
        )

    return ModelStatusResponse(
        model_name="rule_based_fallback",
        model_version="fallback",
        classes=CANONICAL_CATEGORIES,
        confidence_threshold=0.50,
        is_loaded=False,
        evaluation_metrics=eval_metrics,
    )
