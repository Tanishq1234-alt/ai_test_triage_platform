"""Opt-in JSON webhook notification with a safe local dry-run mode."""

import json
import os
from urllib.request import Request, urlopen


def deliver_report(report: dict) -> dict:
    """Send a concise report summary when explicitly enabled; never expose its URL."""
    if os.getenv("TRIAGE_NOTIFICATION_ENABLED", "false").lower() not in {"1", "true", "yes"}:
        return {"status": "skipped", "reason": "Notifications are disabled."}
    webhook_url = os.getenv("TRIAGE_WEBHOOK_URL", "").strip()
    if not webhook_url:
        return {"status": "skipped", "reason": "No webhook is configured."}
    summary = report.get("summary", {})
    body = {
        "text": (
            f"Test run {report.get('run_id', 'unknown')}: "
            f"{summary.get('passed', 0)} passed, {summary.get('failed', 0)} failed, "
            f"{summary.get('skipped', 0)} skipped."
        ),
        "report_url": os.getenv("TRIAGE_REPORT_URL", "").strip() or None,
    }
    if os.getenv("TRIAGE_NOTIFICATION_DRY_RUN", "true").lower() in {"1", "true", "yes"}:
        return {"status": "dry_run", "reason": "Notification prepared; delivery disabled by dry-run mode."}
    try:
        request = Request(
            webhook_url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=10) as response:
            if 200 <= response.status < 300:
                return {"status": "delivered", "reason": "Webhook accepted the report summary."}
            return {"status": "failed", "reason": f"Webhook returned HTTP {response.status}."}
    except Exception:
        # Do not include exception text: URL-based webhook credentials can appear there.
        return {"status": "failed", "reason": "Webhook delivery failed; destination details were suppressed."}
