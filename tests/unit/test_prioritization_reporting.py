import json
from types import SimpleNamespace

from src.orchestration.prioritizer import rank_tests
from src.storage.repository import TestRepository
from scripts.generate_ci_report import build_report
from scripts.run_prioritized_tests import _SelectionPlugin


def test_rank_tests_ignores_future_outcomes_and_uses_failure_history():
    ranking = rank_tests(
        ["tests/test_b.py::test_slow", "tests/test_a.py::test_flaky"],
        [
            {"test_name": "test_flaky", "status": "failed", "timestamp": "2026-05-01T00:00:00+00:00", "duration_seconds": 2},
            {"test_name": "test_flaky", "status": "passed", "timestamp": "2026-05-02T00:00:00+00:00", "duration_seconds": 2},
            {"test_name": "test_slow", "status": "failed", "timestamp": "2030-01-01T00:00:00+00:00", "duration_seconds": 30},
        ],
        as_of=__import__("datetime").datetime.fromisoformat("2026-05-03T00:00:00+00:00"),
    )
    assert ranking[0]["nodeid"].endswith("test_flaky")
    assert "failure rate 50%" in ranking[0]["reason"]
    assert ranking[1]["score"] == 0


def test_rank_tests_fallback_is_deterministic():
    ranking = rank_tests(["z::test_b", "a::test_a"], [])
    assert [row["nodeid"] for row in ranking] == ["a::test_a", "z::test_b"]
    assert all("No historical records" in row["reason"] for row in ranking)


def test_selection_plugin_runs_items_in_priority_order():
    deselected = []
    config = SimpleNamespace(hook=SimpleNamespace(pytest_deselected=lambda items: deselected.extend(items)))
    items = [SimpleNamespace(nodeid=nodeid) for nodeid in ("a", "b", "c")]

    _SelectionPlugin(["b", "a"]).pytest_collection_modifyitems(None, config, items)

    assert [item.nodeid for item in items] == ["b", "a"]
    assert [item.nodeid for item in deselected] == ["c"]


def test_ci_report_parses_classifies_and_persists_junit(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SQLITE_DB_PATH", str(tmp_path / "history.sqlite"))
    monkeypatch.setenv("MODELS_DIR", str(tmp_path / "missing-model"))
    (tmp_path / "reports").mkdir()
    junit = tmp_path / "junit.xml"
    junit.write_text(
        '<testsuites tests="2" failures="1"><testsuite name="unit" tests="2">'
        '<testcase classname="unit.test_a" name="test_ok" time="0.2" />'
        '<testcase classname="unit.test_a" name="test_bad" time="0.4">'
        '<failure type="AssertionError" message="expected true">trace frame</failure>'
        '</testcase></testsuite></testsuites>', encoding="utf-8"
    )
    args = type("Args", (), {
        "junit": str(junit), "log": None, "run_id": "verify-1",
        "output_dir": "reports", "prioritization_json": None,
    })()

    report = build_report(args)

    assert report["summary"] == {"total": 2, "passed": 1, "failed": 1, "skipped": 0}
    assert report["failures"][0]["test_name"] == "test_bad"
    assert report["failures"][0]["rule_based_category"] == "assertion"
    assert report["comparison"]["failure_recall"] is None
    assert report["persistence"]["inserted"] == 2
    assert TestRepository().get_test_record("verify-1::unit.test_a::test_bad") is not None
    assert (tmp_path / "reports" / "ci_report.md").exists()
    assert json.loads((tmp_path / "reports" / "ci_report.json").read_text())["run_id"] == "verify-1"
    args.run_id = "verify-2"
    second = build_report(args)
    assert second["persistence"]["inserted"] == 2


def test_webhook_defaults_to_skipped(monkeypatch):
    from src.notifications.webhook import deliver_report

    monkeypatch.delenv("TRIAGE_NOTIFICATION_ENABLED", raising=False)
    assert deliver_report({"summary": {}})["status"] == "skipped"


def test_webhook_dry_run_never_delivers(monkeypatch):
    from src.notifications.webhook import deliver_report

    monkeypatch.setenv("TRIAGE_NOTIFICATION_ENABLED", "true")
    monkeypatch.setenv("TRIAGE_NOTIFICATION_DRY_RUN", "true")
    monkeypatch.setenv("TRIAGE_WEBHOOK_URL", "https://example.invalid/secret")
    assert deliver_report({"summary": {}})["status"] == "dry_run"
