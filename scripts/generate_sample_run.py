"""Generate sample test run and save structured JSON and raw logs."""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.simulator.runner import TestRunner


def generate_sample_data():
    runner = TestRunner(seed=42)
    summary = runner.run_suite()

    os.makedirs("data/logs", exist_ok=True)
    json_path = os.path.join("data", "sample_test_run.json")
    log_path = os.path.join("data", "logs", "sample_wireless_run.log")

    with open(json_path, "w", encoding="utf-8") as f:
        f.write(summary.model_dump_json(indent=2))

    with open(log_path, "w", encoding="utf-8") as f:
        for res in summary.results:
            cat = res.failure_category.value if res.failure_category else "none"
            f.write(f"=== TEST: {res.test_name} [{res.status.value}] (category: {cat}) ===\n")
            for log_line in res.raw_logs:
                f.write(log_line + "\n")
            if res.error_message:
                f.write(f"ERROR: {res.error_message}\n")
            f.write("\n")

    print(f"Sample run generated successfully:")
    print(f"  - Total tests: {summary.total_tests}")
    print(f"  - Passed: {summary.passed_count}")
    print(f"  - Failed: {summary.failed_count}")
    print(f"  - JSON saved to: {json_path}")
    print(f"  - Logs saved to: {log_path}")


if __name__ == "__main__":
    generate_sample_data()
