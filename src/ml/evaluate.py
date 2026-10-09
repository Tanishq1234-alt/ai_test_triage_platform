"""Evaluation metrics, rule-based baseline comparison, and leakage-safe validation."""

import json
import logging
import os
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from src.log_parser.normalizer import RuleBasedCategorizer
from src.ml.dataset import CANONICAL_CATEGORIES, normalize_failure_category

logger = logging.getLogger(__name__)


def evaluate_classifier(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    class_labels: Optional[List[str]] = None,
    rule_based_categorizer: Optional[RuleBasedCategorizer] = None,
) -> Dict[str, Any]:
    """Evaluate trained ML pipeline on unseen test data and compare against Phase 2 rule-based baseline.

    Leakage prevention: X_test must NOT have been used during pipeline.fit().
    """
    labels = class_labels or sorted(list(set(y_test.unique())))
    y_pred_ml = pipeline.predict(X_test)

    # Compute ML performance metrics
    ml_report = classification_report(
        y_test,
        y_pred_ml,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    ml_cm = confusion_matrix(y_test, y_pred_ml, labels=labels)

    # Compute Phase 2 Rule-Based Baseline predictions on identical test records
    categorizer = rule_based_categorizer or RuleBasedCategorizer()
    baseline_raw_preds = []
    for _, row in X_test.iterrows():
        raw_pred = categorizer.categorize(
            status=row.get("status", "failed"),
            error_type=row.get("error_type"),
            error_message=row.get("error_message"),
            stack_trace=row.get("stack_trace"),
            log_excerpt=row.get("log_excerpt"),
        )
        canonical_pred = normalize_failure_category(raw_pred)
        baseline_raw_preds.append(canonical_pred)

    baseline_report = classification_report(
        y_test,
        baseline_raw_preds,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    baseline_cm = confusion_matrix(y_test, baseline_raw_preds, labels=labels)

    # Build per-class comparison summary
    comparison_table = []
    for lbl in labels:
        ml_stats = ml_report.get(lbl, {})
        base_stats = baseline_report.get(lbl, {})
        comparison_table.append({
            "category": lbl,
            "support": int(ml_stats.get("support", 0)),
            "ml_precision": round(ml_stats.get("precision", 0.0), 4),
            "ml_recall": round(ml_stats.get("recall", 0.0), 4),
            "ml_f1": round(ml_stats.get("f1-score", 0.0), 4),
            "baseline_precision": round(base_stats.get("precision", 0.0), 4),
            "baseline_recall": round(base_stats.get("recall", 0.0), 4),
            "baseline_f1": round(base_stats.get("f1-score", 0.0), 4),
        })

    # Summary overall metrics
    metrics_summary = {
        "test_sample_count": len(y_test),
        "class_labels": labels,
        "ml_macro_f1": round(ml_report.get("macro avg", {}).get("f1-score", 0.0), 4),
        "ml_weighted_f1": round(ml_report.get("weighted avg", {}).get("f1-score", 0.0), 4),
        "baseline_macro_f1": round(baseline_report.get("macro avg", {}).get("f1-score", 0.0), 4),
        "baseline_weighted_f1": round(baseline_report.get("weighted avg", {}).get("f1-score", 0.0), 4),
        "ml_classification_report": ml_report,
        "baseline_classification_report": baseline_report,
        "ml_confusion_matrix": ml_cm.tolist(),
        "baseline_confusion_matrix": baseline_cm.tolist(),
        "comparison_table": comparison_table,
    }

    return metrics_summary


def run_cross_validation(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cv_splits: int = 5,
) -> Dict[str, float]:
    """Execute stratified k-fold cross validation reporting macro-F1 and weighted-F1 mean and std."""
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    scoring = {
        "macro_f1": "f1_macro",
        "weighted_f1": "f1_weighted",
        "precision": "precision_macro",
        "recall": "recall_macro",
    }
    cv_results = cross_validate(pipeline, X, y, cv=skf, scoring=scoring)

    return {
        "cv_splits": cv_splits,
        "mean_macro_f1": round(float(np.mean(cv_results["test_macro_f1"])), 4),
        "std_macro_f1": round(float(np.std(cv_results["test_macro_f1"])), 4),
        "mean_weighted_f1": round(float(np.mean(cv_results["test_weighted_f1"])), 4),
        "std_weighted_f1": round(float(np.std(cv_results["test_weighted_f1"])), 4),
        "mean_precision": round(float(np.mean(cv_results["test_precision"])), 4),
        "mean_recall": round(float(np.mean(cv_results["test_recall"])), 4),
    }


def save_evaluation_report(report_data: Dict[str, Any], output_path: str = "reports/evaluation_report.json") -> str:
    """Save evaluation metrics to reports JSON."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    return output_path
