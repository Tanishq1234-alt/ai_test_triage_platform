"""Service layer for historical reporting, trend aggregations, and data exports."""

import csv
import io
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from src.storage.repository import TestRepository

logger = logging.getLogger(__name__)


class ReportService:
    """Provides reporting metrics, failure digests, data exports, and model evaluations."""

    def __init__(
        self,
        repository: Optional[TestRepository] = None,
        evaluation_report_path: Optional[str] = None,
    ):
        self.repository = repository or TestRepository()
        self.evaluation_report_path = evaluation_report_path or os.path.join(
            "reports", "evaluation_report.json"
        )

    def get_summary(self) -> Dict[str, Any]:
        """Aggregate test execution metrics, pass/fail counts, and category distributions."""
        return self.repository.get_summary_statistics()

    def get_trends(self) -> List[Dict[str, Any]]:
        """Retrieve daily test execution counts and status breakdown."""
        return self.repository.get_trends_by_date()

    def get_failures(
        self,
        limit: int = 50,
        offset: int = 0,
        failure_category: Optional[str] = None,
        search: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Retrieve paginated test failure records."""
        # Query specifically for failed and error statuses
        results = self.repository.list_test_records(
            limit=limit,
            offset=offset,
            statuses=["failed", "error"],
            failure_category=failure_category,
            search=search,
        )
        return {
            "items": results["items"],
            "total": results["total"],
            "limit": limit,
            "offset": offset,
        }

    def get_model_evaluation_metrics(self) -> Dict[str, Any]:
        """Loads Phase 3 evaluation report comparing ML classifier against rule-based baseline."""
        if os.path.exists(self.evaluation_report_path):
            try:
                with open(self.evaluation_report_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to read evaluation report '{self.evaluation_report_path}': {e}")

        # Return fallback template if report file is not yet generated
        return {
            "metadata": {
                "model_name": "unknown",
                "model_version": "unknown",
                "classes": [
                    "assertion_failure",
                    "configuration_error",
                    "connection_error",
                    "dependency_error",
                    "timeout",
                    "unknown",
                ],
            },
            "evaluation_summary": {
                "status": "Evaluation report not found on disk; metrics are unavailable.",
            },
        }

    def export_records(self, export_format: str = "json") -> Tuple[str, str]:
        """Exports all test records stored in SQLite as a JSON or CSV formatted string."""
        # Query all records
        all_records = self.repository.list_test_records(limit=100000, offset=0)["items"]

        fmt = export_format.lower().strip()
        if fmt == "csv":
            if not all_records:
                return "test_id,test_name,status,failure_category\n", "text/csv"

            output = io.StringIO()
            headers = [
                "test_id",
                "test_name",
                "test_suite",
                "timestamp",
                "status",
                "duration_seconds",
                "error_type",
                "error_message",
                "stack_trace",
                "log_excerpt",
                "failure_category",
                "source_file",
                "created_at",
            ]
            writer = csv.DictWriter(output, fieldnames=headers, extrasaction="ignore")
            writer.writeheader()
            for r in all_records:
                writer.writerow(r)
            return output.getvalue(), "text/csv"

        # Default: JSON format
        return json.dumps(all_records, indent=2, default=str), "application/json"
