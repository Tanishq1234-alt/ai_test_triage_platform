"""Executes wireless test automation suite and automatically parses results into normalized failure records."""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.log_parser import RecordExporter, correlate_and_normalize
from src.simulator.runner import TestRunner


def run_and_parse():
    print("Executing simulated wireless test suite...")
    runner = TestRunner(seed=42)
    summary = runner.run_suite("automated_wireless_run")

    os.makedirs("data/logs", exist_ok=True)
    os.makedirs("data/normalized", exist_ok=True)

    log_path = os.path.join("data", "logs", "current_test_run.log")
    with open(log_path, "w", encoding="utf-8") as f:
        for res in summary.results:
            cat = res.failure_category.value if res.failure_category else "none"
            f.write(f"=== TEST: {res.test_name} [{res.status.value}] (category: {cat}) ===\n")
            for log_line in res.raw_logs:
                f.write(log_line + "\n")
            if res.error_message:
                f.write(f"ERROR: {res.error_message}\n")
            f.write("\n")

    print(f"Test suite finished ({summary.total_tests} tests: {summary.passed_count} passed, {summary.failed_count} failed).")
    print(f"Log saved to: {log_path}")

    print("Parsing and normalizing test execution logs...")
    records = correlate_and_normalize(log_path=log_path)

    exporter = RecordExporter()
    outputs = exporter.export_all(
        records=records,
        output_dir="data/normalized",
        base_name="normalized_results",
        append=True,
    )

    print("\nNormalized Output Generated:")
    print(f"  - Total records: {len(records)}")
    print(f"  - JSON: {outputs['json']}")
    print(f"  - CSV:  {outputs['csv']}")


if __name__ == "__main__":
    run_and_parse()
