"""Routes for test record exploration, single test inspection, and human review feedback."""

import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status

from src.api.schemas import (
    BatchImportRequest,
    BatchImportResponse,
    HumanReviewRequest,
    HumanReviewResponse,
    PaginatedTestRecordsResponse,
    TestRecordDetailResponse,
    TestRecordResponse,
)
from src.ml.dataset import CANONICAL_CATEGORIES, normalize_failure_category
from src.storage.repository import TestRepository

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/tests", tags=["Test Executions"])


def get_repository() -> TestRepository:
    return TestRepository()


@router.get("", response_model=PaginatedTestRecordsResponse)
def list_tests(
    limit: int = Query(50, ge=1, le=500, description="Page size limit"),
    offset: int = Query(0, ge=0, description="Offset index"),
    status: Optional[str] = Query(None, description="Filter by status (passed, failed, error, skipped)"),
    failure_category: Optional[str] = Query(None, description="Filter by root cause failure category"),
    start_date: Optional[str] = Query(None, description="ISO start date filter"),
    end_date: Optional[str] = Query(None, description="ISO end date filter"),
    search: Optional[str] = Query(None, description="Search keyword in test name, id, or error message"),
    repo: TestRepository = Depends(get_repository),
) -> PaginatedTestRecordsResponse:
    """Retrieve historical test executions with filtering, search, and pagination."""
    result = repo.list_test_records(
        limit=limit,
        offset=offset,
        status=status,
        failure_category=failure_category,
        start_date=start_date,
        end_date=end_date,
        search=search,
    )
    items = [TestRecordResponse(**item) for item in result["items"]]
    return PaginatedTestRecordsResponse(
        items=items,
        total=result["total"],
        limit=limit,
        offset=offset,
    )


@router.get("/{test_id}", response_model=TestRecordDetailResponse)
def get_test_by_id(
    test_id: str,
    repo: TestRepository = Depends(get_repository),
) -> TestRecordDetailResponse:
    """Fetch details of a single test execution, including ML classification and review history."""
    record = repo.get_test_record(test_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Test record with id '{test_id}' not found.",
        )
    return TestRecordDetailResponse(**record)


@router.post("/{test_id}/review", response_model=HumanReviewResponse)
def review_test_prediction(
    test_id: str,
    payload: HumanReviewRequest,
    repo: TestRepository = Depends(get_repository),
) -> HumanReviewResponse:
    """Submit human review correction to update ground truth while preserving the original ML audit log."""
    record = repo.get_test_record(test_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Test record with id '{test_id}' not found.",
        )

    # Validate and canonicalize corrected category
    canonical_cat = normalize_failure_category(payload.corrected_category)
    if canonical_cat not in CANONICAL_CATEGORIES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid category '{payload.corrected_category}'. Must be one of {CANONICAL_CATEGORIES}.",
        )

    latest_pred = record.get("latest_prediction")
    orig_cat = latest_pred.get("predicted_category", record.get("failure_category", "unknown")) if latest_pred else record.get("failure_category", "unknown")

    review_data = {
        "original_predicted_category": orig_cat,
        "corrected_category": canonical_cat,
        "reviewer_name": payload.reviewer_name.strip() or "test_engineer",
        "notes": payload.notes or "",
    }
    review_id = repo.save_human_review(test_id, review_data)

    return HumanReviewResponse(
        success=True,
        review_id=review_id,
        test_id=test_id,
        original_predicted_category=orig_cat,
        corrected_category=canonical_cat,
        reviewer_name=review_data["reviewer_name"],
        message="Human review successfully recorded and canonical failure category updated.",
    )


@router.post("/import", response_model=BatchImportResponse)
def batch_import_tests(
    payload: BatchImportRequest,
    repo: TestRepository = Depends(get_repository),
) -> BatchImportResponse:
    """Batch import test records into SQLite with deduplication."""
    result = repo.save_test_records_batch(payload.records)
    return BatchImportResponse(
        inserted=result["inserted"],
        skipped=result["skipped"],
        total=result["total"],
    )
