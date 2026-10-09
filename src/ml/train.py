"""Training workflow for ML failure classification pipelines."""

import argparse
import logging
import os
import sys
from typing import Any, Dict, Optional, Tuple
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.ml.dataset import (
    CANONICAL_CATEGORIES,
    DatasetLoader,
    SyntheticDatasetGenerator,
)
from src.ml.evaluate import evaluate_classifier, run_cross_validation, save_evaluation_report
from src.ml.feature_engineering import build_ml_pipeline
from src.ml.model_registry import ModelRegistry

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def prepare_training_data(
    real_data_path: Optional[str] = None,
    synthetic_samples_per_class: int = 50,
    seed: int = 42,
    output_dataset_path: str = "data/training/failure_dataset.json",
) -> pd.DataFrame:
    """Prepares and validates a balanced dataset by loading real records and augmenting with labeled synthetic examples."""
    loader = DatasetLoader(min_samples_per_class=10)
    real_df = pd.DataFrame()

    if real_data_path and os.path.exists(real_data_path):
        logger.info(f"Loading real failure records from {real_data_path}")
        if real_data_path.endswith(".json"):
            real_df = loader.load_from_json(real_data_path)
        else:
            real_df = loader.load_from_csv(real_data_path)

    # Generate synthetic diverse records
    syn_gen = SyntheticDatasetGenerator(seed=seed)
    synthetic_df = syn_gen.generate_dataset(samples_per_category=synthetic_samples_per_class)

    # Combine real and synthetic records
    combined_df = pd.concat([real_df, synthetic_df], ignore_index=True)
    # Deduplicate
    combined_df = combined_df.drop_duplicates(subset=["test_name", "error_message", "stack_trace"]).reset_index(drop=True)

    dist_info = loader.check_class_distribution(combined_df)
    logger.info(f"Dataset prepared: total={dist_info['total']} records across {len(dist_info['counts'])} classes.")
    for cat, count in dist_info["counts"].items():
        logger.info(f"  - {cat}: {count} records")

    # Persist dataset for reproducibility and audit
    os.makedirs(os.path.dirname(os.path.abspath(output_dataset_path)), exist_ok=True)
    records = combined_df.to_dict(orient="records")
    with open(output_dataset_path, "w", encoding="utf-8") as f:
        import json
        json.dump(records, f, indent=2, ensure_ascii=False)

    return combined_df


def train_and_evaluate(
    df: pd.DataFrame,
    test_size: float = 0.25,
    seed: int = 42,
    model_version: str = "v1.0.0",
    models_dir: str = "models",
    reports_dir: str = "reports",
) -> Dict[str, Any]:
    """Train ML models, prevent data leakage, evaluate vs baseline, and persist best model."""
    y = df["canonical_category"]
    X = df[["test_name", "error_type", "error_message", "stack_trace", "log_excerpt", "duration_seconds"]]

    # Stratified train/test split preserving class distribution
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(sss.split(X, y))

    X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
    y_train, y_test = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()

    logger.info(f"Split completed: Train={len(X_train)} samples, Test={len(X_test)} samples (seed={seed})")

    # Candidate 1: Logistic Regression (Baseline)
    lr_pipeline = build_ml_pipeline(model_type="logistic_regression", random_state=seed)
    lr_pipeline.fit(X_train, y_train)
    lr_eval = evaluate_classifier(lr_pipeline, X_test, y_test)
    logger.info(f"Logistic Regression Macro-F1: {lr_eval['ml_macro_f1']:.4f}")

    # Candidate 2: Linear Support Vector Classifier (Calibrated)
    svc_pipeline = build_ml_pipeline(model_type="linear_svc", random_state=seed)
    svc_pipeline.fit(X_train, y_train)
    svc_eval = evaluate_classifier(svc_pipeline, X_test, y_test)
    logger.info(f"Linear SVC Macro-F1: {svc_eval['ml_macro_f1']:.4f}")

    # Compare and choose best model
    if svc_eval["ml_macro_f1"] > lr_eval["ml_macro_f1"]:
        best_name = "linear_svc"
        best_pipeline = svc_pipeline
        best_eval = svc_eval
    else:
        best_name = "logistic_regression"
        best_pipeline = lr_pipeline
        best_eval = lr_eval

    logger.info(f"Selected best model: {best_name} (Macro-F1={best_eval['ml_macro_f1']:.4f})")

    # Run 5-fold cross-validation on the entire dataset for generalization assessment
    cv_metrics = run_cross_validation(best_pipeline, X, y, cv_splits=5)
    logger.info(f"5-Fold CV Mean Macro-F1: {cv_metrics['mean_macro_f1']:.4f} (+/- {cv_metrics['std_macro_f1']:.4f})")

    # Build model metadata
    metadata = {
        "model_name": best_name,
        "model_version": model_version,
        "random_seed": seed,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "classes": list(best_pipeline.classes_),
        "macro_f1": best_eval["ml_macro_f1"],
        "weighted_f1": best_eval["ml_weighted_f1"],
        "baseline_macro_f1": best_eval["baseline_macro_f1"],
        "baseline_weighted_f1": best_eval["baseline_weighted_f1"],
        "cv_metrics": cv_metrics,
        "leakage_prevention": "Strict train/test split before fitting vectorizer and model.",
    }

    # Save best model to registry
    registry = ModelRegistry(models_dir=models_dir)
    save_result = registry.save_model(best_pipeline, metadata, model_version=model_version, is_latest=True)

    # Save full evaluation report
    report_data = {
        "metadata": metadata,
        "evaluation_summary": best_eval,
        "candidate_comparison": {
            "logistic_regression_macro_f1": lr_eval["ml_macro_f1"],
            "linear_svc_macro_f1": svc_eval["ml_macro_f1"],
            "baseline_rule_macro_f1": best_eval["baseline_macro_f1"],
        },
        "cv_metrics": cv_metrics,
    }
    report_file = os.path.join(reports_dir, "evaluation_report.json")
    save_evaluation_report(report_data, report_file)
    logger.info(f"Evaluation report saved to {report_file}")

    return {
        "best_model_name": best_name,
        "model_paths": save_result,
        "report_path": report_file,
        "evaluation": best_eval,
        "metadata": metadata,
    }


def main():
    parser = argparse.ArgumentParser(description="Train ML failure triage models.")
    parser.add_argument("--data", type=str, default="data/normalized/normalized_results.json", help="Path to normalized Phase 2 records.")
    parser.add_argument("--samples-per-class", type=int, default=50, help="Synthetic samples per class.")
    parser.add_argument("--seed", type=int, default=42, help="Fixed random seed.")
    parser.add_argument("--version", type=str, default="v1.0.0", help="Model version tag.")
    parser.add_argument("--models-dir", type=str, default="models", help="Directory for model artifacts.")
    parser.add_argument("--reports-dir", type=str, default="reports", help="Directory for evaluation reports.")

    args = parser.parse_args()

    df = prepare_training_data(
        real_data_path=args.data if os.path.exists(args.data) else None,
        synthetic_samples_per_class=args.samples_per_class,
        seed=args.seed,
    )
    result = train_and_evaluate(
        df=df,
        seed=args.seed,
        model_version=args.version,
        models_dir=args.models_dir,
        reports_dir=args.reports_dir,
    )
    print("\nTraining and Evaluation Complete:")
    print(f"  Best Model: {result['best_model_name']}")
    print(f"  ML Macro-F1: {result['metadata']['macro_f1']}")
    print(f"  Rule-Based Baseline Macro-F1: {result['metadata']['baseline_macro_f1']}")
    print(f"  5-Fold CV Mean Macro-F1: {result['metadata']['cv_metrics']['mean_macro_f1']}")
    print(f"  Model Artifact: {result['model_paths']['model_path']}")
    print(f"  Evaluation Report: {result['report_path']}")


if __name__ == "__main__":
    main()
