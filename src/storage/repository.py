"""Repository performing CRUD queries, deduplication, and analytics on SQLite storage."""

import hashlib
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from src.storage.database import Database, DatabaseError

logger = logging.getLogger(__name__)


class TestRepository:
    """Provides high-level repository access to test executions, ML predictions, and reviews."""

    def __init__(self, db: Optional[Database] = None):
        self.db = db or Database()

    @staticmethod
    def compute_record_hash(record: Dict[str, Any]) -> str:
        """Computes a deterministic SHA256 hash across stable fields for duplicate prevention."""
        name = str(record.get("test_name", "")).strip()
        msg = str(record.get("error_message", "")).strip()
        trace = str(record.get("stack_trace", "")).strip()
        ts = str(record.get("timestamp", "")).strip()
        content = f"{name}::{msg}::{trace}::{ts}".encode("utf-8")
        return hashlib.sha256(content).hexdigest()

    def save_test_record(self, record: Dict[str, Any]) -> Tuple[str, bool]:
        """Save a single test record. Returns (test_id, is_new)."""
        rec_hash = self.compute_record_hash(record)
        test_id = record.get("test_id") or f"TEST-{rec_hash[:12]}"

        with self.db.transaction() as conn:
            # Check for existing record via record_hash or test_id
            cursor = conn.execute(
                "SELECT test_id FROM test_records WHERE record_hash = ? OR test_id = ? LIMIT 1;",
                (rec_hash, test_id)
            )
            existing = cursor.fetchone()
            if existing:
                return existing["test_id"], False

            conn.execute(
                """
                INSERT INTO test_records (
                    test_id, test_name, test_suite, timestamp, status, duration_seconds,
                    error_type, error_message, stack_trace, log_excerpt, failure_category,
                    source_file, record_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    test_id,
                    record.get("test_name", "unnamed_test"),
                    record.get("test_suite"),
                    record.get("timestamp"),
                    record.get("status", "unknown").lower(),
                    record.get("duration_seconds"),
                    record.get("error_type"),
                    record.get("error_message"),
                    record.get("stack_trace"),
                    record.get("log_excerpt"),
                    record.get("failure_category"),
                    record.get("source_file"),
                    rec_hash,
                )
            )
            return test_id, True

    def save_test_records_batch(self, records: List[Dict[str, Any]]) -> Dict[str, int]:
        """Batch insert test records with deduplication. Returns count of inserted vs skipped."""
        inserted = 0
        skipped = 0
        for rec in records:
            _, is_new = self.save_test_record(rec)
            if is_new:
                inserted += 1
            else:
                skipped += 1
        return {"inserted": inserted, "skipped": skipped, "total": len(records)}

    def get_test_record(self, test_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve single test record with associated ML prediction and review history."""
        with self.db.transaction() as conn:
            cursor = conn.execute("SELECT * FROM test_records WHERE test_id = ? LIMIT 1;", (test_id,))
            row = cursor.fetchone()
            if not row:
                return None
            result = dict(row)

            # Fetch latest ML prediction
            pred_cursor = conn.execute(
                "SELECT * FROM ml_predictions WHERE test_id = ? ORDER BY id DESC LIMIT 1;",
                (test_id,)
            )
            pred_row = pred_cursor.fetchone()
            if pred_row:
                pred_dict = dict(pred_row)
                pred_dict["probabilities"] = json.loads(pred_dict.pop("probabilities_json") or "{}")
                pred_dict["top_features"] = json.loads(pred_dict.pop("top_features_json") or "[]")
                pred_dict["needs_review"] = bool(pred_dict["needs_review"])
                pred_dict["is_agreement"] = bool(pred_dict["is_agreement"])
                result["latest_prediction"] = pred_dict
            else:
                result["latest_prediction"] = None

            # Fetch human review history
            rev_cursor = conn.execute(
                "SELECT * FROM human_reviews WHERE test_id = ? ORDER BY id ASC;",
                (test_id,)
            )
            result["human_reviews"] = [dict(r) for r in rev_cursor.fetchall()]
            return result

    def list_test_records(
        self,
        limit: int = 50,
        offset: int = 0,
        status: Optional[str] = None,
        statuses: Optional[List[str]] = None,
        failure_category: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        search: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Query test records with pagination, status/category filters, and keyword search."""
        query_parts = ["SELECT * FROM test_records WHERE 1=1"]
        count_parts = ["SELECT COUNT(*) as total FROM test_records WHERE 1=1"]
        params: List[Any] = []

        if status:
            query_parts.append("AND status = ?")
            count_parts.append("AND status = ?")
            params.append(status.lower().strip())

        if statuses:
            normalized_statuses = list(dict.fromkeys(
                item.lower().strip() for item in statuses if item and item.strip()
            ))
            if normalized_statuses:
                placeholders = ", ".join("?" for _ in normalized_statuses)
                clause = f"AND status IN ({placeholders})"
                query_parts.append(clause)
                count_parts.append(clause)
                params.extend(normalized_statuses)

        if failure_category:
            query_parts.append("AND failure_category = ?")
            count_parts.append("AND failure_category = ?")
            params.append(failure_category.strip())

        if start_date:
            query_parts.append("AND timestamp >= ?")
            count_parts.append("AND timestamp >= ?")
            params.append(start_date)

        if end_date:
            query_parts.append("AND timestamp <= ?")
            count_parts.append("AND timestamp <= ?")
            params.append(end_date)

        if search:
            query_parts.append("AND (test_name LIKE ? OR error_message LIKE ? OR test_id LIKE ?)")
            count_parts.append("AND (test_name LIKE ? OR error_message LIKE ? OR test_id LIKE ?)")
            term = f"%{search.strip()}%"
            params.extend([term, term, term])

        # Get total count
        with self.db.transaction() as conn:
            count_cursor = conn.execute(" ".join(count_parts), params)
            total_records = count_cursor.fetchone()["total"]

            # Query paginated rows
            query_parts.append("ORDER BY timestamp DESC, created_at DESC LIMIT ? OFFSET ?;")
            query_params = list(params) + [max(1, limit), max(0, offset)]
            cursor = conn.execute(" ".join(query_parts), query_params)
            rows = [dict(r) for r in cursor.fetchall()]

            return {
                "items": rows,
                "total": total_records,
                "limit": limit,
                "offset": offset,
            }

    def save_ml_prediction(self, test_id: str, prediction: Dict[str, Any]) -> int:
        """Store an ML triage prediction record associated with a test case."""
        with self.db.transaction() as conn:
            cursor = conn.execute(
                """
                INSERT INTO ml_predictions (
                    test_id, predicted_category, raw_predicted_category, confidence,
                    probabilities_json, model_version, needs_review, review_reason,
                    top_features_json, rule_based_category, is_agreement
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    test_id,
                    prediction.get("predicted_category", "unknown"),
                    prediction.get("raw_predicted_category"),
                    prediction.get("confidence", 0.0),
                    json.dumps(prediction.get("probabilities", {})),
                    prediction.get("model_version", "unknown"),
                    1 if prediction.get("needs_review") else 0,
                    prediction.get("review_reason"),
                    json.dumps(prediction.get("top_features", [])),
                    prediction.get("rule_based_category"),
                    1 if prediction.get("is_agreement", True) else 0,
                )
            )
            return cursor.lastrowid

    def save_human_review(self, test_id: str, review: Dict[str, Any]) -> int:
        """Save reviewer feedback and update the canonical failure category while preserving original ML prediction."""
        with self.db.transaction() as conn:
            cursor = conn.execute(
                """
                INSERT INTO human_reviews (
                    test_id, original_predicted_category, corrected_category, reviewer_name, notes
                ) VALUES (?, ?, ?, ?, ?);
                """,
                (
                    test_id,
                    review.get("original_predicted_category", "unknown"),
                    review.get("corrected_category", "unknown"),
                    review.get("reviewer_name", "anonymous_reviewer"),
                    review.get("notes", ""),
                )
            )
            # Update canonical failure category on the test record for reporting
            conn.execute(
                "UPDATE test_records SET failure_category = ? WHERE test_id = ?;",
                (review.get("corrected_category"), test_id)
            )
            return cursor.lastrowid

    def get_summary_statistics(self) -> Dict[str, Any]:
        """Aggregate total counts, pass/fail rates, category distribution, and duration metrics."""
        with self.db.transaction() as conn:
            # Overall status counts
            status_cur = conn.execute("SELECT status, COUNT(*) as cnt FROM test_records GROUP BY status;")
            status_counts = {r["status"]: r["cnt"] for r in status_cur.fetchall()}

            total = sum(status_counts.values())
            passed = status_counts.get("passed", 0)
            failed = status_counts.get("failed", 0) + status_counts.get("error", 0)
            skipped = status_counts.get("skipped", 0)
            pass_rate = round((passed / total * 100.0), 2) if total > 0 else 0.0

            # Failure category distribution
            cat_cur = conn.execute(
                """
                SELECT failure_category, COUNT(*) as cnt 
                FROM test_records 
                WHERE status IN ('failed', 'error') AND failure_category IS NOT NULL
                GROUP BY failure_category;
                """
            )
            category_counts = {r["failure_category"]: r["cnt"] for r in cat_cur.fetchall()}

            # Duration metrics
            dur_cur = conn.execute(
                "SELECT AVG(duration_seconds) as avg_d, MAX(duration_seconds) as max_d FROM test_records WHERE duration_seconds IS NOT NULL;"
            )
            dur_row = dur_cur.fetchone()
            avg_duration = round(dur_row["avg_d"] or 0.0, 4) if dur_row else 0.0
            max_duration = round(dur_row["max_d"] or 0.0, 4) if dur_row else 0.0

            # Reviews count
            rev_cnt = conn.execute("SELECT COUNT(*) as c FROM human_reviews;").fetchone()["c"]

            return {
                "total_tests": total,
                "passed_tests": passed,
                "failed_tests": failed,
                "skipped_tests": skipped,
                "pass_rate_pct": pass_rate,
                "avg_duration_seconds": avg_duration,
                "max_duration_seconds": max_duration,
                "category_distribution": category_counts,
                "human_reviews_count": rev_cnt,
            }

    def get_trends_by_date(self) -> List[Dict[str, Any]]:
        """Group test executions by date for trend visualization."""
        with self.db.transaction() as conn:
            cur = conn.execute(
                """
                SELECT SUBSTR(timestamp, 1, 10) as run_date,
                       status,
                       COUNT(*) as cnt
                FROM test_records
                WHERE timestamp IS NOT NULL AND timestamp != ''
                GROUP BY run_date, status
                ORDER BY run_date ASC;
                """
            )
            date_map: Dict[str, Dict[str, Any]] = {}
            for r in cur.fetchall():
                d = r["run_date"]
                if d not in date_map:
                    date_map[d] = {"date": d, "total": 0, "passed": 0, "failed": 0, "skipped": 0}
                st = r["status"]
                c = r["cnt"]
                date_map[d]["total"] += c
                if st == "passed":
                    date_map[d]["passed"] += c
                elif st in ["failed", "error"]:
                    date_map[d]["failed"] += c
                elif st == "skipped":
                    date_map[d]["skipped"] += c
            return list(date_map.values())
