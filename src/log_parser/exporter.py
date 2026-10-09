"""Data exporter for normalized test failure records supporting JSON, CSV, and historical accumulation."""

import csv
import json
import os
from typing import Dict, List, Optional
import pandas as pd

from src.log_parser.normalizer import NormalizedTestRecord


class RecordExporter:
    """Exports normalized test failure records to JSON, CSV, and Pandas DataFrames preserving run history."""

    COLUMNS = [
        "test_id",
        "test_name",
        "test_suite",
        "timestamp",
        "status",
        "duration_seconds",
        "error_type",
        "error_message",
        "stack_trace",
        "log_excerpt",
        "failure_category",
        "source_file",
    ]

    def to_dataframe(self, records: List[NormalizedTestRecord]) -> pd.DataFrame:
        """Convert records into a clean Pandas DataFrame for downstream analytics and ML pipelines."""
        data = [r.model_dump() for r in records]
        df = pd.DataFrame(data)
        # Ensure all standard columns are present
        for col in self.COLUMNS:
            if col not in df.columns:
                df[col] = None
        return df[self.COLUMNS]

    def export_to_json(
        self,
        records: List[NormalizedTestRecord],
        output_path: str,
        append: bool = False,
    ) -> str:
        """Export records to JSON. If append is True, merges with existing records to preserve execution history."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        new_data = [r.model_dump() for r in records]

        if append and os.path.exists(output_path):
            try:
                with open(output_path, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
                if isinstance(existing_data, list):
                    # Combine existing and new data
                    final_data = existing_data + new_data
                else:
                    final_data = new_data
            except Exception:
                final_data = new_data
        else:
            final_data = new_data

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_data, f, indent=2, ensure_ascii=False)

        return output_path

    def export_to_csv(
        self,
        records: List[NormalizedTestRecord],
        output_path: str,
        append: bool = False,
    ) -> str:
        """Export records to CSV. If append is True, preserves existing rows and appends new records."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        file_exists = os.path.exists(output_path) and os.path.getsize(output_path) > 0

        mode = "a" if (append and file_exists) else "w"
        write_header = not (append and file_exists)

        with open(output_path, mode, newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.COLUMNS)
            if write_header:
                writer.writeheader()
            for r in records:
                row_dict = r.model_dump()
                # Ensure missing values write cleanly as empty strings for CSV compatibility
                formatted_row = {
                    k: ("" if row_dict.get(k) is None else row_dict.get(k))
                    for k in self.COLUMNS
                }
                writer.writerow(formatted_row)

        return output_path

    def export_all(
        self,
        records: List[NormalizedTestRecord],
        output_dir: str,
        base_name: str = "normalized_results",
        append: bool = True,
    ) -> Dict[str, str]:
        """Convenience method exporting both JSON and CSV formats."""
        os.makedirs(output_dir, exist_ok=True)
        json_path = os.path.join(output_dir, f"{base_name}.json")
        csv_path = os.path.join(output_dir, f"{base_name}.csv")

        self.export_to_json(records, json_path, append=append)
        self.export_to_csv(records, csv_path, append=append)

        return {"json": json_path, "csv": csv_path}
