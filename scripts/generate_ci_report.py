"""Parse a JUnit run, classify failures, persist history, and write JSON/Markdown reports."""

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.log_parser import correlate_and_normalize
from src.ml.predict import FailureClassifierPredictor
from src.notifications.webhook import deliver_report
from src.storage.repository import TestRepository


def _markdown(report):
    summary = report["summary"]
    lines = [
        f"# CI Test Report: {report['run_id']}", "",
        f"- Mode: {report['mode']}",
        f"- Tests: {summary['total']} total; {summary['passed']} passed, {summary['failed']} failed, {summary['skipped']} skipped",
        f"- Aggregate test duration: {report['execution_time_seconds']:.3f}s",
        f"- Model: {report['model']['version'] or 'unavailable'}",
        "", "## Failures", "",
    ]
    if not report["failures"]:
        lines.append("No failed tests were reported.")
    for failure in report["failures"]:
        lines += [
            f"### {failure['test_name']}",
            f"- Duration: {failure.get('duration_seconds') if failure.get('duration_seconds') is not None else 'unknown'}s",
            f"- Rule category: {failure.get('rule_based_category') or 'unknown'}",
            f"- ML category: {failure.get('ml_category') or 'unavailable'}",
            f"- Confidence: {failure.get('confidence') if failure.get('confidence') is not None else 'unavailable'}",
            f"- Human review: {failure.get('needs_review') if failure.get('needs_review') is not None else 'unavailable'}",
        ]
        excerpt = failure.get("log_excerpt") or failure.get("stack_trace") or failure.get("error_message")
        if excerpt:
            lines += ["", "```text", str(excerpt)[:3000], "```"]
        lines.append("")
    lines += ["## Prioritization", ""]
    priority = report.get("prioritization")
    if priority:
        lines.append(
            f"Selected {priority.get('selected_count', 'unknown')} of "
            f"{priority.get('total_collected', 'unknown')} collected tests; "
            f"subset execution: {priority.get('selection_is_subset', 'unknown')}."
        )
    else:
        lines.append("No prioritization metadata was supplied.")
    comparison = report.get("comparison") or {}
    if comparison.get("reference_run_id"):
        lines += [
            "", "## Run comparison", "",
            f"Compared with {comparison['reference_run_id']} ({comparison.get('reference_mode', 'unknown')}): "
            f"{comparison.get('reference_test_count')} tests / {comparison.get('reference_time_seconds')}s versus "
            f"{comparison.get('current_test_count')} tests / {comparison.get('current_time_seconds')}s.",
            f"Execution time change: {comparison.get('time_change_pct')}%.",
            "Failure recall was not measured because no independent labeled defect set is available.",
        ]
    lines += ["", f"Notification: {report.get('notification', {}).get('status', 'not_attempted')}.", ""]
    if report["warnings"]:
        lines += ["## Warnings", ""] + [f"- {item}" for item in report["warnings"]] + [""]
    return "\n".join(lines)


def build_report(args):
    junit_path = Path(args.junit)
    warnings = []
    if not junit_path.exists():
        raise FileNotFoundError(f"JUnit report does not exist: {junit_path}")
    records = correlate_and_normalize(junit_path=str(junit_path), log_path=args.log)
    if not records:
        warnings.append("The JUnit parser returned no test records.")
    counts = Counter(record.status for record in records)
    run_id = args.run_id or datetime.now(timezone.utc).strftime("local-%Y%m%dT%H%M%SZ")
    run_timestamp = datetime.now(timezone.utc).isoformat()
    model_version = None
    predictor = None
    try:
        predictor = FailureClassifierPredictor(
            model_version=os.getenv("MODEL_VERSION", "latest"),
            models_dir=os.getenv("MODELS_DIR", "models"),
            confidence_threshold=float(os.getenv("CONFIDENCE_THRESHOLD", "0.50")),
        )
        model_version = predictor.model_version
    except Exception as exc:
        warnings.append(f"ML classifier unavailable; failed tests have rule-based categories only ({type(exc).__name__}).")

    repository = TestRepository()
    failures = []
    persisted = {"inserted": 0, "skipped": 0}
    for record in records:
        values = record.model_dump()
        values["test_id"] = f"{run_id}::{values['test_id']}"
        # The repository's deduplication hash includes timestamp. Stamp missing
        # JUnit suite times with the ingestion time so repeated runs remain history.
        values["timestamp"] = values.get("timestamp") or run_timestamp
        prediction = None
        if record.status in ("failed", "error") and predictor is not None:
            try:
                prediction = predictor.predict(values)
            except Exception as exc:
                warnings.append(f"ML prediction failed for a test ({type(exc).__name__}); rule-based data was retained.")
                predictor = None
        try:
            _, inserted = repository.save_test_record(values)
            persisted["inserted" if inserted else "skipped"] += 1
            if prediction and inserted:
                repository.save_ml_prediction(values["test_id"], prediction)
        except Exception as exc:
            warnings.append(f"Could not persist a test record ({type(exc).__name__}).")
        if record.status in ("failed", "error"):
            failures.append({
                "test_id": values["test_id"],
                "test_name": record.test_name,
                "status": record.status,
                "duration_seconds": record.duration_seconds,
                "error_type": record.error_type,
                "error_message": record.error_message,
                "stack_trace": record.stack_trace,
                "log_excerpt": record.log_excerpt,
                "rule_based_category": record.failure_category,
                "ml_category": prediction.get("predicted_category") if prediction else None,
                "confidence": prediction.get("confidence") if prediction else None,
                "needs_review": prediction.get("needs_review") if prediction else None,
                "review_status": "needs_review" if prediction and prediction.get("needs_review") else "not_reviewed",
                "model_version": prediction.get("model_version") if prediction else None,
            })

    eval_path = Path("reports/evaluation_report.json")
    try:
        evaluation = json.loads(eval_path.read_text(encoding="utf-8")) if eval_path.exists() else None
    except (OSError, json.JSONDecodeError):
        evaluation = None
    if evaluation is None:
        warnings.append("Phase 3 evaluation metrics are unavailable; no metrics were included.")
    priority = None
    if args.prioritization_json:
        try:
            priority = json.loads(Path(args.prioritization_json).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            warnings.append("Prioritization metadata could not be read.")
    durations = [record.duration_seconds for record in records if record.duration_seconds is not None]
    comparison = {"failure_recall": None, "reason": "No independent labeled defect dataset is available."}
    comparison_report = getattr(args, "comparison_report", None)
    if comparison_report:
        try:
            reference = json.loads(Path(comparison_report).read_text(encoding="utf-8"))
            current_time = round(sum(durations), 4)
            reference_time = float(reference.get("execution_time_seconds") or 0.0)
            comparison = {
                "reference_run_id": reference.get("run_id"),
                "reference_mode": reference.get("mode", "unknown"),
                "reference_test_count": reference.get("summary", {}).get("total"),
                "current_test_count": len(records),
                "reference_time_seconds": reference_time,
                "current_time_seconds": current_time,
                "time_change_pct": round((current_time - reference_time) / reference_time * 100, 2) if reference_time else None,
                "failure_recall": None,
                "reason": "A test-count and measured-duration comparison only; defect recall remains unmeasured.",
            }
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            warnings.append("Comparison report could not be read; run comparison is unavailable.")
    report = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": priority.get("mode", "full") if priority else "full",
        "summary": {
            "total": len(records), "passed": counts.get("passed", 0),
            "failed": counts.get("failed", 0) + counts.get("error", 0),
            "skipped": counts.get("skipped", 0),
        },
        "execution_time_seconds": round(sum(durations), 4),
        "failures": failures,
        "prioritization": priority,
        "comparison": comparison,
        "model": {"version": model_version, "evaluation_metrics": evaluation},
        "persistence": persisted,
        "warnings": warnings,
        "notification": {"status": "not_attempted"},
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path, markdown_path = output_dir / "ci_report.json", output_dir / "ci_report.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    markdown_path.write_text(_markdown(report), encoding="utf-8")
    report["notification"] = deliver_report(report)
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    markdown_path.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "markdown": str(markdown_path), "summary": report["summary"], "warnings": warnings}, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--junit", default="reports/junit.xml")
    parser.add_argument("--log")
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--run-id")
    parser.add_argument("--prioritization-json")
    parser.add_argument("--comparison-report", help="A report from another actual run for count/time comparison")
    args = parser.parse_args()
    build_report(args)


if __name__ == "__main__":
    main()
