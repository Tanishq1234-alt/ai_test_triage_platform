from src.api.routes.tests import review_test_prediction
from src.api.schemas import HumanReviewRequest
from src.services.report_service import ReportService
from src.storage.database import Database
from src.storage.repository import TestRepository


class EmptyRepository:
    def list_test_records(self, **kwargs):
        return {"items": [], "total": 0}


def _record(test_id: str, status: str, timestamp: str) -> dict:
    return {
        "test_id": test_id,
        "test_name": f"test_{test_id}",
        "timestamp": timestamp,
        "status": status,
        "error_message": "link handshake timed out" if status != "passed" else None,
    }


def test_failure_reports_filter_before_pagination(tmp_path):
    repository = TestRepository(Database(str(tmp_path / "reports.db")))
    repository.save_test_records_batch(
        [
            _record("pass-1", "passed", "2026-01-03T00:00:00Z"),
            _record("fail-1", "failed", "2026-01-02T00:00:00Z"),
            _record("error-1", "error", "2026-01-01T00:00:00Z"),
        ]
    )
    service = ReportService(repository=repository)

    first_page = service.get_failures(limit=1, offset=0)
    second_page = service.get_failures(limit=1, offset=1)

    assert first_page["total"] == 2
    assert second_page["total"] == 2
    assert first_page["items"][0]["test_id"] == "fail-1"
    assert second_page["items"][0]["test_id"] == "error-1"


def test_human_review_is_persisted_with_original_prediction(tmp_path):
    db_path = tmp_path / "review.db"
    repository = TestRepository(Database(str(db_path)))
    repository.save_test_record(_record("review-1", "failed", "2026-01-01T00:00:00Z"))
    repository.save_ml_prediction(
        "review-1",
        {
            "predicted_category": "timeout",
            "raw_predicted_category": "timeout",
            "confidence": 0.72,
            "model_version": "test-model",
            "needs_review": True,
        },
    )
    response = review_test_prediction(
        "review-1",
        HumanReviewRequest(
            corrected_category="connection_error",
            reviewer_name="engineer-1",
            notes="Access point dropped the link.",
        ),
        repository,
    )
    assert response.success is True
    assert response.original_predicted_category == "timeout"

    reopened_repository = TestRepository(Database(str(db_path)))
    saved = reopened_repository.get_test_record("review-1")
    assert saved["failure_category"] == "connection_error"
    assert saved["latest_prediction"]["predicted_category"] == "timeout"
    assert saved["human_reviews"][0]["corrected_category"] == "connection_error"


def test_missing_evaluation_report_does_not_supply_metrics(tmp_path):
    service = ReportService(
        repository=EmptyRepository(),
        evaluation_report_path=str(tmp_path / "missing-evaluation.json")
    )

    report = service.get_model_evaluation_metrics()

    assert report["metadata"]["model_name"] == "unknown"
    assert "macro_f1" not in report["metadata"]
    assert "unavailable" in report["evaluation_summary"]["status"]
