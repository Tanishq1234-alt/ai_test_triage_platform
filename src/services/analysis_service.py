"""Service layer orchestrating failure log analysis, ML prediction, and actionable triage."""

import datetime
import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple

from src.log_parser.normalizer import NormalizedTestRecord, RuleBasedCategorizer
from src.log_parser.text_parser import PlainTextLogParser
from src.ml.dataset import normalize_failure_category
from src.ml.predict import FailureClassifierPredictor
from src.storage.repository import TestRepository

logger = logging.getLogger(__name__)


SUGGESTED_DEBUGGING_ACTIONS: Dict[str, List[str]] = {
    "timeout": [
        "Check device responsiveness and verify if keepalive heartbeats are arriving within the configured deadline.",
        "Inspect network latency and packet jitter between the test harness and device under test.",
        "Verify whether the test execution timeout deadline is sufficient under peak DUT processing loads.",
        "Check device firmware logs for thread starvation, deadlock, or blocked I/O during state transitions.",
    ],
    "connection_error": [
        "Verify the device's physical and RF radio link state (e.g., RSSI thresholds and antenna coupling).",
        "Inspect connection handshake negotiation logs for dropped beacon frames or disassociation requests.",
        "Check wireless access point status, DHCP address leases, and gateway connectivity.",
        "Inspect device network driver logs for socket resets, carrier drops, or unexpected teardowns.",
    ],
    "assertion_failure": [
        "Compare actual measurement telemetry with required SLA threshold specifications.",
        "Inspect telemetry samples immediately preceding the failure to detect degradation trends.",
        "Verify if firmware calibration parameters or test tolerance limits were updated recently.",
        "Confirm test harness expectations align with current specification revision.",
    ],
    "configuration_error": [
        "Verify configuration payloads, SSID settings, security keys, and channel assignments.",
        "Check for schema validation mismatches or missing required keys in configuration profile.",
        "Ensure environment variables, credentials, and test profile definitions exist in test environment.",
        "Inspect firmware provisioning logs for rejected or unsupported parameter values.",
    ],
    "dependency_error": [
        "Check if simulated hardware mock or peripheral subsystem service was started before test execution.",
        "Verify Python environment dependencies and required external driver libraries.",
        "Inspect system event log for crash reports or unavailable external service daemons.",
        "Confirm fixture teardown and setup lifecycle hooks executed successfully without dangling locks.",
    ],
    "unknown": [
        "Request human triage review to classify this unrecognized error pattern.",
        "Examine full raw logs and device console output around the failure timestamp for unhandled conditions.",
        "Check if a new failure mode was introduced in recent firmware or test suite changes.",
        "Once verified, submit human review correction to incorporate this case into future model retraining.",
    ],
}


class AnalysisService:
    """Orchestrates test failure parsing, ML classification, and triage recommendations."""

    def __init__(
        self,
        predictor: Optional[FailureClassifierPredictor] = None,
        repository: Optional[TestRepository] = None,
    ):
        self.repository = repository or TestRepository()
        try:
            self.predictor = predictor or FailureClassifierPredictor()
        except Exception as e:
            logger.warning(f"Could not load ML predictor from disk; running in degraded/rule-based mode: {e}")
            self.predictor = None

        self.rule_categorizer = RuleBasedCategorizer()
        self.text_parser = PlainTextLogParser()

    @staticmethod
    def get_suggested_actions(category: str) -> List[str]:
        """Returns targeted troubleshooting steps for a categorized failure."""
        normalized = normalize_failure_category(category)
        return SUGGESTED_DEBUGGING_ACTIONS.get(normalized, SUGGESTED_DEBUGGING_ACTIONS["unknown"])

    @staticmethod
    def extract_evidence(
        error_message: Optional[str] = None,
        stack_trace: Optional[str] = None,
        log_excerpt: Optional[str] = None,
    ) -> List[str]:
        """Extracts key evidence snippets from error message, stack trace, and log excerpt."""
        evidence: List[str] = []
        if error_message and error_message.strip():
            evidence.append(f"Primary error: {error_message.strip()}")

        if stack_trace and stack_trace.strip():
            trace_lines = [line.strip() for line in stack_trace.strip().splitlines() if line.strip()]
            if trace_lines:
                # Capture the top of traceback and final exception line
                evidence.append(f"Exception origin: {trace_lines[-1]}")
                if len(trace_lines) > 2:
                    evidence.append(f"Traceback frame: {trace_lines[1]}")

        if log_excerpt and log_excerpt.strip():
            log_lines = [
                line.strip() for line in log_excerpt.strip().splitlines()
                if any(kw in line.upper() for kw in ["ERROR", "FAIL", "WARN", "EXCEPTION", "TIMEOUT"])
            ]
            for line in log_lines[:3]:
                if line not in evidence:
                    evidence.append(f"Log marker: {line}")

        return evidence

    def analyze_failure(
        self,
        record_data: Dict[str, Any],
        save_to_db: bool = True,
    ) -> Dict[str, Any]:
        """Analyzes a failure record or raw log text, runs ML triage, and optionally persists results."""
        # 1. Check if raw log text was submitted instead of structured fields
        raw_log = record_data.get("log_text")
        if raw_log and not record_data.get("error_message") and not record_data.get("stack_trace"):
            parsed_records = self.text_parser.parse_text(raw_log, source_file="manual_input")
            if parsed_records:
                # Pick the first failing record or first record
                failing = [r for r in parsed_records if r.status in ("failed", "error")]
                selected = failing[0] if failing else parsed_records[0]
                norm_dict = selected.model_dump()
            else:
                # Fallback: treat raw log text as error message / excerpt
                norm_dict = {
                    "test_id": record_data.get("test_id") or f"TEST-{uuid.uuid4().hex[:8].upper()}",
                    "test_name": record_data.get("test_name") or "manual_analyzed_test",
                    "test_suite": record_data.get("test_suite") or "manual_suite",
                    "timestamp": record_data.get("timestamp") or datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "status": "failed",
                    "duration_seconds": record_data.get("duration_seconds", 0.0),
                    "error_type": "UnknownError",
                    "error_message": raw_log[:300].strip(),
                    "stack_trace": None,
                    "log_excerpt": raw_log.strip(),
                    "failure_category": "unknown",
                    "source_file": "manual_input",
                }
        else:
            test_id = record_data.get("test_id") or f"TEST-{uuid.uuid4().hex[:8].upper()}"
            norm_dict = {
                "test_id": test_id,
                "test_name": record_data.get("test_name", "unnamed_test"),
                "test_suite": record_data.get("test_suite"),
                "timestamp": record_data.get("timestamp") or datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "status": record_data.get("status", "failed"),
                "duration_seconds": record_data.get("duration_seconds"),
                "error_type": record_data.get("error_type"),
                "error_message": record_data.get("error_message"),
                "stack_trace": record_data.get("stack_trace"),
                "log_excerpt": record_data.get("log_excerpt"),
                "failure_category": record_data.get("failure_category"),
                "source_file": record_data.get("source_file"),
            }

        # 2. Compute Rule-Based Classification
        raw_rule = self.rule_categorizer.categorize(
            status=norm_dict.get("status", "failed"),
            error_type=norm_dict.get("error_type"),
            error_message=norm_dict.get("error_message"),
            stack_trace=norm_dict.get("stack_trace"),
            log_excerpt=norm_dict.get("log_excerpt"),
        )
        rule_based_category = normalize_failure_category(raw_rule)

        # 3. Compute ML Prediction (or fallback to rule-based)
        if self.predictor is not None:
            pred_res = self.predictor.predict(norm_dict)
            predicted_category = pred_res["predicted_category"]
            raw_predicted_category = pred_res["raw_predicted_category"]
            confidence = pred_res["confidence"]
            probabilities = pred_res["probabilities"]
            model_version = pred_res["model_version"]
            needs_review = pred_res["needs_review"]
            review_reason = pred_res["review_reason"]
            top_features = pred_res["top_features"]
            is_agreement = pred_res["is_agreement"]
        else:
            predicted_category = rule_based_category
            raw_predicted_category = rule_based_category
            confidence = 0.50 if rule_based_category != "unknown" else 0.0
            probabilities = {rule_based_category: 1.0}
            model_version = "rule_based_fallback"
            needs_review = rule_based_category == "unknown"
            review_reason = "ML model unavailable; using rule-based fallback."
            top_features = []
            is_agreement = True

        # Update failure_category on normalized record to best known label
        effective_category = predicted_category if predicted_category != "unknown" else rule_based_category
        norm_dict["failure_category"] = effective_category

        # 4. Generate recommendations and evidence
        suggested_actions = self.get_suggested_actions(effective_category)
        evidence = self.extract_evidence(
            error_message=norm_dict.get("error_message"),
            stack_trace=norm_dict.get("stack_trace"),
            log_excerpt=norm_dict.get("log_excerpt"),
        )

        prediction_payload = {
            "predicted_category": predicted_category,
            "raw_predicted_category": raw_predicted_category,
            "confidence": confidence,
            "probabilities": probabilities,
            "model_version": model_version,
            "needs_review": needs_review,
            "review_reason": review_reason,
            "top_features": top_features,
            "rule_based_category": rule_based_category,
            "is_agreement": is_agreement,
        }

        # 5. Persist to SQLite if requested
        test_id = norm_dict["test_id"]
        saved_record_id = test_id
        if save_to_db:
            saved_id, _ = self.repository.save_test_record(norm_dict)
            norm_dict["test_id"] = saved_id
            saved_record_id = saved_id
            self.repository.save_ml_prediction(saved_id, prediction_payload)

        return {
            "test_id": saved_record_id,
            "test_name": norm_dict.get("test_name"),
            "normalized_record": norm_dict,
            "rule_based_category": rule_based_category,
            "ml_prediction": prediction_payload,
            "predicted_category": predicted_category,
            "confidence": confidence,
            "model_version": model_version,
            "needs_review": needs_review,
            "review_reason": review_reason,
            "top_features": top_features,
            "suggested_actions": suggested_actions,
            "evidence": evidence,
        }
