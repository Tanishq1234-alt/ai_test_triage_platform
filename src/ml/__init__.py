"""Machine learning failure classification package."""

from src.ml.dataset import DatasetLoader, SyntheticDatasetGenerator, normalize_failure_category
from src.ml.feature_engineering import FailureFeatureExtractor, build_ml_pipeline
from src.ml.model_registry import ModelRegistry
from src.ml.predict import FailureClassifierPredictor

__all__ = [
    "DatasetLoader",
    "SyntheticDatasetGenerator",
    "normalize_failure_category",
    "FailureFeatureExtractor",
    "build_ml_pipeline",
    "ModelRegistry",
    "FailureClassifierPredictor",
]
