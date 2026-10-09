"""Rank collected pytest tests and optionally execute a bounded prioritized subset."""

import argparse
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.orchestration.prioritizer import rank_tests
from src.storage.repository import TestRepository


class _SelectionPlugin:
    def __init__(self, selected):
        self.selected = list(selected)
        self.selected_set = set(selected)

    def pytest_collection_modifyitems(self, session, config, items):
        by_nodeid = {item.nodeid: item for item in items}
        deselected = [item for item in items if item.nodeid not in self.selected_set]
        items[:] = [by_nodeid[nodeid] for nodeid in self.selected if nodeid in by_nodeid]
        if deselected:
            config.hook.pytest_deselected(items=deselected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("full", "prioritized", "rank-only"), default="full")
    parser.add_argument("--limit", type=int, default=20, help="Maximum selected tests in prioritized mode")
    parser.add_argument("--full-suite-every", type=int, default=7, help="Run full suite every N prioritized invocations")
    parser.add_argument("--state-file", default=".triage_prioritization_state.json")
    parser.add_argument("--report-json", default=None, help="Write ranking details to this JSON path")
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.limit < 1 or args.full_suite_every < 1:
        parser.error("--limit and --full-suite-every must be positive")
    pytest_args = args.pytest_args[1:] if args.pytest_args[:1] == ["--"] else args.pytest_args
    pytest_args = pytest_args or ["tests"]

    collected = []
    class _CollectPlugin:
        def pytest_collection_finish(self, session):
            collected.extend(item.nodeid for item in session.items)

    collect_args = [arg for arg in pytest_args if not arg.startswith("--junitxml")]
    collect_code = pytest.main([*collect_args, "--collect-only", "-q"], plugins=[_CollectPlugin()])
    if collect_code not in (pytest.ExitCode.OK, pytest.ExitCode.NO_TESTS_COLLECTED):
        return int(collect_code)

    history = TestRepository().list_test_records(limit=100000, offset=0)["items"]
    ranking = rank_tests(collected, history)
    selected = ranking
    state_path = Path(args.state_file)
    state = {"prioritized_runs_since_full": 0}
    if state_path.exists():
        try:
            state.update(json.loads(state_path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            pass
    run_full = args.mode == "full"
    if run_full:
        state["prioritized_runs_since_full"] = 0
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    if args.mode == "prioritized":
        run_full = int(state.get("prioritized_runs_since_full", 0)) + 1 >= args.full_suite_every
        state["prioritized_runs_since_full"] = 0 if run_full else int(state.get("prioritized_runs_since_full", 0)) + 1
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        if not run_full:
            selected = ranking[: args.limit]

    payload = {
        "mode": "full" if run_full else args.mode,
        "total_collected": len(collected),
        "selected_count": len(collected) if run_full else len(selected),
        "selection_is_subset": not run_full and len(selected) < len(collected),
        "full_suite_safety_interval": args.full_suite_every,
        "ranking": ranking,
    }
    print(json.dumps({k: v for k, v in payload.items() if k != "ranking"}, indent=2))
    if args.report_json:
        Path(args.report_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.mode == "rank-only":
        print(json.dumps(ranking, indent=2))
        return 0
    plugins = [] if run_full else [_SelectionPlugin([item["nodeid"] for item in selected])]
    return int(pytest.main(pytest_args, plugins=plugins))


if __name__ == "__main__":
    raise SystemExit(main())
