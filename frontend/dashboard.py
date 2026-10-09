"""Streamlit dashboard for historical test triage and human review."""

import json
import os

import pandas as pd
import requests
import streamlit as st


st.set_page_config(page_title="Failure Triage", page_icon="🧪", layout="wide")
st.title("AI Test Failure Triage")
st.caption("Review test history, inspect model classifications, and record corrections.")

api_url = st.sidebar.text_input(
    "FastAPI server",
    value=os.getenv("TRIAGE_API_URL", "http://127.0.0.1:8000").rstrip("/"),
).rstrip("/")


def api_get(path: str, **params):
    response = requests.get(f"{api_url}{path}", params=params, timeout=10)
    response.raise_for_status()
    return response.json()


def api_post(path: str, payload: dict):
    response = requests.post(f"{api_url}{path}", json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


try:
    health = api_get("/health")
    st.sidebar.success(f"API {health['status']} · database {'connected' if health['database_connected'] else 'offline'}")
except requests.RequestException as exc:
    st.sidebar.error(f"Cannot reach the API: {exc}")
    st.info("Start the FastAPI service, then refresh this page.")
    st.stop()


overview_tab, failures_tab, analyze_tab = st.tabs(["Overview", "Failures and review", "Analyze a log"])

with overview_tab:
    try:
        report = api_get("/api/reports/summary")
        summary = report["summary"]
        metrics = st.columns(5)
        metrics[0].metric("Tests", summary["total_tests"])
        metrics[1].metric("Passed", summary["passed_tests"])
        metrics[2].metric("Failed or errored", summary["failed_tests"])
        metrics[3].metric("Pass rate", f"{summary['pass_rate_pct']:.1f}%")
        metrics[4].metric("Human reviews", summary["human_reviews_count"])

        st.subheader("Daily test outcomes")
        trends = report.get("trends", [])
        if trends:
            trend_df = pd.DataFrame(trends).set_index("date")
            st.bar_chart(trend_df[["passed", "failed", "skipped"]])
        else:
            st.info("No historical test runs are stored yet.")

        categories = summary.get("category_distribution", {})
        if categories:
            st.subheader("Failure categories")
            st.bar_chart(pd.Series(categories, name="Failures"))

        ci_json = os.path.join("reports", "ci_report.json")
        ci_markdown = os.path.join("reports", "ci_report.md")
        if os.path.exists(ci_json):
            st.subheader("Recent CI execution")
            try:
                with open(ci_json, "r", encoding="utf-8") as report_file:
                    ci_report = json.load(report_file)
                ci_summary = ci_report.get("summary", {})
                st.caption(f"Run {ci_report.get('run_id', 'unknown')} · {ci_report.get('mode', 'unknown')} mode")
                ci_metrics = st.columns(4)
                ci_metrics[0].metric("Selected tests", ci_summary.get("total", 0))
                ci_metrics[1].metric("Passed", ci_summary.get("passed", 0))
                ci_metrics[2].metric("Failed", ci_summary.get("failed", 0))
                ci_metrics[3].metric("Test duration", f"{ci_report.get('execution_time_seconds', 0):.2f}s")
                compare = ci_report.get("comparison", {})
                if compare.get("reference_run_id"):
                    st.write(
                        f"Compared with {compare.get('reference_run_id')}: "
                        f"{compare.get('reference_test_count')} tests / {compare.get('reference_time_seconds')}s; "
                        f"time change {compare.get('time_change_pct')}%. Failure recall is not measured."
                    )
                prioritization = ci_report.get("prioritization")
                if prioritization and prioritization.get("ranking"):
                    st.dataframe(pd.DataFrame(prioritization["ranking"]), use_container_width=True, hide_index=True)
                st.caption(f"Notification: {ci_report.get('notification', {}).get('status', 'not configured')}")
                for warning in ci_report.get("warnings", []):
                    st.warning(warning)
                if os.path.exists(ci_markdown):
                    with open(ci_markdown, "rb") as report_file:
                        st.download_button("Download CI Markdown report", report_file, "ci_report.md", "text/markdown")
                st.download_button("Download CI JSON report", json.dumps(ci_report, indent=2), "ci_report.json", "application/json")
            except (OSError, ValueError, TypeError) as exc:
                st.warning(f"Could not read the local CI report: {exc}")
    except requests.RequestException as exc:
        st.error(f"Could not load historical summary: {exc}")

with failures_tab:
    st.subheader("Failure records")
    filters = st.columns([2, 1])
    search = filters[0].text_input("Search tests or error messages")
    category = filters[1].selectbox(
        "Category",
        ["All", "assertion_failure", "configuration_error", "connection_error", "dependency_error", "timeout", "unknown"],
    )
    query = {"limit": 100}
    if search.strip():
        query["search"] = search.strip()
    if category != "All":
        query["failure_category"] = category

    try:
        page = api_get("/api/reports/failures", **query)
        records = page["items"]
        st.caption(f"Showing {len(records)} of {page['total']} matching failures")
        if records:
            st.dataframe(
                pd.DataFrame(records)[["test_id", "test_name", "timestamp", "status", "failure_category", "error_message"]],
                use_container_width=True,
                hide_index=True,
            )
            selected_id = st.selectbox("Inspect a failure", [record["test_id"] for record in records])
            detail = api_get(f"/api/tests/{selected_id}")
            st.json(detail.get("latest_prediction") or {"prediction": "No prediction stored"})
            with st.expander("Test details and review history"):
                st.json(detail)

            st.markdown("**Record a human correction**")
            with st.form("human_review"):
                corrected = st.selectbox(
                    "Correct category",
                    ["assertion_failure", "configuration_error", "connection_error", "dependency_error", "timeout", "unknown"],
                )
                reviewer = st.text_input("Reviewer", value="test_engineer")
                notes = st.text_area("Review notes")
                submitted = st.form_submit_button("Save review")
            if submitted:
                try:
                    saved = api_post(
                        f"/api/tests/{selected_id}/review",
                        {"corrected_category": corrected, "reviewer_name": reviewer, "notes": notes},
                    )
                    st.success(saved["message"])
                    st.rerun()
                except requests.RequestException as exc:
                    st.error(f"Could not save review: {exc}")
        else:
            st.info("No matching failure records.")
    except requests.RequestException as exc:
        st.error(f"Could not load failure records: {exc}")

with analyze_tab:
    st.subheader("Analyze a failure log")
    with st.form("analyze_log"):
        test_name = st.text_input("Test name", value="manual_log_analysis")
        log_text = st.text_area("Log text", height=240, placeholder="Paste a failing test log here")
        analyze_submitted = st.form_submit_button("Analyze and save")
    if analyze_submitted:
        if not log_text.strip():
            st.warning("Paste log text before analyzing.")
        else:
            try:
                result = api_post(
                    "/api/analyze",
                    {"test_name": test_name, "status": "failed", "log_text": log_text, "save_to_db": True},
                )
                st.success(f"Classified as {result['predicted_category']} ({result['confidence']:.0%} confidence)")
                if result["needs_review"]:
                    st.warning(result.get("review_reason") or "This result needs human review.")
                st.json(result)
            except requests.RequestException as exc:
                st.error(f"Analysis request failed: {exc}")
