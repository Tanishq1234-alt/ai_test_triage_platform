"""Model registry managing persistence, metadata tracking, and safe loading of trained pipelines."""

import datetime
import json
import logging
import os
from typing import Any, Dict, Optional, Tuple
import joblib
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Handles serialization, versioning, and loading of trained scikit-learn triage pipelines and metadata."""

    def __init__(self, models_dir: str = "models"):
        self.models_dir = os.path.abspath(models_dir)
        os.makedirs(self.models_dir, exist_ok=True)

    def save_model(
        self,
        pipeline: Pipeline,
        metadata: Dict[str, Any],
        model_version: str = "v1.0.0",
        is_latest: bool = True,
    ) -> Dict[str, str]:
        """Serialize trained pipeline and write metadata JSON."""
        model_filename = f"failure_classifier_{model_version}.joblib"
        metadata_filename = f"model_metadata_{model_version}.json"

        model_path = os.path.join(self.models_dir, model_filename)
        metadata_path = os.path.join(self.models_dir, metadata_filename)

        # Attach metadata properties
        full_metadata = dict(metadata)
        full_metadata["model_version"] = model_version
        full_metadata["saved_at"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        full_metadata["model_file"] = model_filename

        # Save model pipeline with joblib
        joblib.dump(pipeline, model_path, compress=3)

        # Save metadata JSON
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(full_metadata, f, indent=2, ensure_ascii=False)

        latest_paths = {}
        if is_latest:
            latest_model_path = os.path.join(self.models_dir, "latest.joblib")
            latest_meta_path = os.path.join(self.models_dir, "latest_metadata.json")
            joblib.dump(pipeline, latest_model_path, compress=3)
            with open(latest_meta_path, "w", encoding="utf-8") as f:
                json.dump(full_metadata, f, indent=2, ensure_ascii=False)
            latest_paths = {"latest_model": latest_model_path, "latest_meta": latest_meta_path}

        logger.info(f"Model {model_version} persisted to {model_path}")
        result = {"model_path": model_path, "metadata_path": metadata_path}
        result.update(latest_paths)
        return result

    def load_model(self, model_version: str = "latest") -> Tuple[Pipeline, Dict[str, Any]]:
        """Safely load pipeline artifact and metadata.

        Safe loading principle: Restricts loading to designated trusted local artifact directory.
        """
        if model_version == "latest":
            model_path = os.path.join(self.models_dir, "latest.joblib")
            meta_path = os.path.join(self.models_dir, "latest_metadata.json")
        else:
            model_path = os.path.join(self.models_dir, f"failure_classifier_{model_version}.joblib")
            meta_path = os.path.join(self.models_dir, f"model_metadata_{model_version}.json")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model artifact not found at {model_path}")

        try:
            pipeline = joblib.load(model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to safely load model artifact from '{model_path}': {e}")

        metadata: Dict[str, Any] = {}
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

        return pipeline, metadata

    def list_models(self) -> List[Dict[str, Any]]:
        """List all saved model artifacts and their metadata in the registry."""
        models = []
        for fname in os.listdir(self.models_dir):
            if fname.startswith("model_metadata_") and fname.endswith(".json") and fname != "latest_metadata.json":
                meta_path = os.path.join(self.models_dir, fname)
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                        models.append(meta)
                except Exception:
                    continue
        return sorted(models, key=lambda x: x.get("saved_at", ""), reverse=True)
