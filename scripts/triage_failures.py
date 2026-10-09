"""Triage CLI integrating ML failure predictions with Phase 2 normalized test records."""

import argparse
import json
import os
import sys
from typing import Any, Dict, List
import pandas as pd

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ml.predict import FailureClassifierPredictor


def triage_file(
    input_file: str,
    output_dir: str = "data/triaged",
    base_name: str = "triaged_results",
    model_version: str = "latest",
    confidence_threshold: float = 0.60,
    models_dir: str = "models",
) -> Dict[str, str]:
    """Ingest normalized Phase 2 records, apply ML triage, and save enriched records."""
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    if input_file.endswith(".json"):
        with open(input_file, "r", encoding="utf-8") as f:
            records = json.load(f)
    else:
        df = pd.read_csv(input_file)
        records = df.to_dict(orient="records")

    predictor = FailureClassifierPredictor(
        model_version=model_version,
        confidence_threshold=confidence_threshold,
        models_dir=models_dir,
    )

    triaged_records: List[Dict[str, Any]] = []
    print(f"\nProcessing {len(records)} test records with ML Model ({predictor.model_version})...\n")

    for rec in records:
        status = str(rec.get("status", "")).lower().strip()
        record_copy = dict(rec)

        if status in ["failed", "error"]:
            prediction = predictor.predict(rec)
            record_copy["ml_triage"] = prediction
        else:
            record_copy["ml_triage"] = None

        triaged_records.append(record_copy)

    os.makedirs(output_dir, exist_ok=True)
    out_json = os.path.join(output_dir, f"{base_name}.json")
    out_csv = os.path.join(output_dir, f"{base_name}.csv")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(triaged_records, f, indent=2, ensure_ascii=False)

    # Flatten for CSV export
    flat_rows = []
    for r in triaged_records:
        row = dict(r)
        triage = row.pop("ml_triage", None)
        if triage:
            row["ml_predicted_category"] = triage.get("predicted_category")
            row["ml_confidence"] = triage.get("confidence")
            row["ml_needs_review"] = triage.get("needs_review")
            row["ml_model_version"] = triage.get("model_version")
            row["ml_rule_based_category"] = triage.get("rule_based_category")
            row["ml_is_agreement"] = triage.get("is_agreement")
        else:
            row["ml_predicted_category"] = None
            row["ml_confidence"] = None
            row["ml_needs_review"] = None
            row["ml_model_version"] = None
            row["ml_rule_based_category"] = None
            row["ml_is_agreement"] = None
        flat_rows.append(row)

    pd.DataFrame(flat_rows).to_csv(out_csv, index=False)

    print("Triage Summary Table:")
    print(f"{'Test ID':<42} {'Status':<8} {'Rule Category':<20} {'ML Category':<22} {'Confidence':<10} {'Review?'}")
    print("-" * 115)
    for r in triaged_records:
        triage = r.get("ml_triage")
        if triage:
            print(
                f"{r.get('test_id', ''):<42} {r.get('status', ''):<8} "
                f"{triage.get('rule_based_category', 'none'):<20} "
                f"{triage.get('predicted_category', 'unknown'):<22} "
                f"{triage.get('confidence', 0.0):<10.2f} "
                f"{str(triage.get('needs_review', False))}"
            )

    print(f"\nOutputs written:")
    print(f"  - JSON: {out_json}")
    print(f"  - CSV:  {out_csv}")

    return {"json": out_json, "csv": out_csv}


def main():
    parser = argparse.ArgumentParser(description="Triage normalized test failures with ML pipeline.")
    parser.add_argument("--input", "-i", type=str, default="data/normalized/normalized_results.json", help="Path to normalized Phase 2 output.")
    parser.add_argument("--output-dir", "-o", type=str, default="data/triaged", help="Output directory.")
    parser.add_argument("--threshold", "-t", type=float, default=0.60, help="Confidence threshold for review.")
    parser.add_argument("--models-dir", type=str, default="models", help="Directory where models are stored.")
    args = parser.parse_args()

    triage_file(
        input_file=args.input,
        output_dir=args.output_dir,
        confidence_threshold=args.threshold,
        models_dir=args.models_dir,
    )


if __name__ == "__main__":
    main()
