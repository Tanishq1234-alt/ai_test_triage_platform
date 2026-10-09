"""Historical reporting, aggregations, trend analytics, and data export routes."""

import logging
from typing import Any, Dict, Optional
from fastapi import APIRouter, Depends, Query, Response

from src.api.schemas import PaginatedTestRecordsResponse, SummaryStatisticsResponse, TestRecordResponse
from src.services.report_service import ReportService

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Historical Reports"])


def get_report_service() -> ReportService:
    return ReportService()


@router.get("/api/summary", response_model=SummaryStatisticsResponse)
def get_summary_statistics(
    report: ReportService = Depends(get_report_service),
) -> SummaryStatisticsResponse:
    """Return total test executions, pass/fail counts, pass rate, and category distributions."""
    stats = report.get_summary()
    return SummaryStatisticsResponse(**stats)


@router.get("/api/reports/summary")
def get_reports_summary_with_trends(
    report: ReportService = Depends(get_report_service),
) -> Dict[str, Any]:
    """Return consolidated summary metrics and daily execution trends for charts."""
    summary = report.get_summary()
    trends = report.get_trends()
    return {
        "summary": summary,
        "trends": trends,
    }


@router.get("/api/reports/failures", response_model=PaginatedTestRecordsResponse)
def get_failure_reports(
    limit: int = Query(50, ge=1, le=500, description="Page limit"),
    offset: int = Query(0, ge=0, description="Offset index"),
    failure_category: Optional[str] = Query(None, description="Filter by failure category"),
    search: Optional[str] = Query(None, description="Search term in test name, error message, or test_id"),
    report: ReportService = Depends(get_report_service),
) -> PaginatedTestRecordsResponse:
    """Return historical failure records with their failure categorization."""
    result = report.get_failures(
        limit=limit,
        offset=offset,
        failure_category=failure_category,
        search=search,
    )
    items = [TestRecordResponse(**r) for r in result["items"]]
    return PaginatedTestRecordsResponse(
        items=items,
        total=result["total"],
        limit=limit,
        offset=offset,
    )


@router.get("/api/reports/download")
def download_test_report(
    format: str = Query("json", pattern="^(json|csv)$", description="Format to download ('json' or 'csv')"),
    report: ReportService = Depends(get_report_service),
) -> Response:
    """Download historical test execution and failure data in CSV or JSON format."""
    content, media_type = report.export_records(export_format=format)
    filename = f"test_triage_report.{format}"
    headers = {
        "Content-Disposition": f'attachment; filename="{filename}"'
    }
    return Response(content=content, media_type=media_type, headers=headers)
