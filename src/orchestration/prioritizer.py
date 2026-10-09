"""Transparent, deterministic prioritization from already-observed test history."""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional


def _parse_time(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return result.replace(tzinfo=timezone.utc) if result.tzinfo is None else result
    except (TypeError, ValueError):
        return None


def rank_tests(
    candidates: Iterable[str],
    history: Iterable[Dict[str, Any]],
    *,
    as_of: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    """Rank pytest node IDs using past failure rate, failure recency and duration.

    History newer than ``as_of`` is ignored. If no usable history exists, node IDs
    are returned in lexical order with an explicit deterministic-fallback reason.
    """
    now = as_of or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    by_name: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in history:
        observed = _parse_time(record.get("timestamp") or record.get("created_at"))
        if observed and observed > now:
            continue
        name = str(record.get("test_name") or "").strip()
        if name:
            by_name[name].append(record)

    candidates = list(dict.fromkeys(str(item) for item in candidates))
    if not any(by_name.values()):
        return [
            {"nodeid": nodeid, "score": 0.0, "reason": "No historical records; deterministic node ID order."}
            for nodeid in sorted(candidates)
        ]

    rows = []
    for nodeid in candidates:
        name = nodeid.rsplit("::", 1)[-1]
        records = by_name.get(name, [])
        if not records:
            rows.append({"nodeid": nodeid, "score": 0.0, "reason": "No history for this test; kept after known-risk tests."})
            continue
        failures = [r for r in records if str(r.get("status", "")).lower() in {"failed", "error"}]
        rate = len(failures) / len(records)
        newest_failure = max((_parse_time(r.get("timestamp") or r.get("created_at")) for r in failures), default=None)
        age_days = max(0.0, (now - newest_failure).total_seconds() / 86400) if newest_failure else None
        recency = math.exp(-age_days / 30.0) if age_days is not None else 0.0
        durations = [float(r["duration_seconds"]) for r in records if r.get("duration_seconds") is not None]
        duration = math.log1p(max(durations, default=0.0))
        # Clamp duration contribution so duration cannot dominate observed failures.
        duration_score = min(1.0, duration / math.log1p(300.0))
        score = 0.65 * rate + 0.25 * recency + 0.10 * duration_score
        reasons = [f"historical failure rate {rate:.0%}"]
        if newest_failure:
            reasons.append(f"last failure {age_days:.1f} days ago")
        if durations:
            reasons.append(f"mean duration {sum(durations) / len(durations):.2f}s")
        rows.append({"nodeid": nodeid, "score": round(score, 6), "reason": "; ".join(reasons)})
    return sorted(rows, key=lambda item: (-item["score"], item["nodeid"]))
