"""Inference and triage prediction interface for normalized test failure records."""

import logging
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.log_parser.normalizer import NormalizedTestRecord, RuleBasedCategorizer
from src.ml.dataset import CANONICAL_CATEGORIES, normalize_failure_category
from src.ml.model_registry import ModelRegistry

logger = logging.getLogger(__name__)


class FailureClassifierPredictor:
    """Predicts root-cause failure categories for incoming test failure records."""

    def __init__(
        self,
        pipeline: Optional[Pipeline] = None,
        metadata: Optional[Dict[str, Any]] = None,
        confidence_threshold: float = 0.50,
        model_version: str = "latest",
        models_dir: str = "models",
    ):
        self.confidence_threshold = confidence_threshold
        self.rule_categorizer = RuleBasedCategorizer()

        if pipeline is not None:
            self.pipeline = pipeline
            self.metadata = metadata or {"model_version": model_version}
        else:
            registry = ModelRegistry(models_dir=models_dir)
            self.pipeline, self.metadata = registry.load_model(model_version=model_version)

        self.model_version = self.metadata.get("model_version", "unknown")
        # Extract classes known to model
        self.classes_ = list(getattr(self._get_classifier(), "classes_", CANONICAL_CATEGORIES))

    def _get_classifier(self):
        """Helper retrieving the base or final classifier in the pipeline."""
        if hasattr(self.pipeline, "named_steps"):
            return self.pipeline.named_steps.get("classifier", self.pipeline)
        return self.pipeline

    def predict(
        self,
        record: Union[NormalizedTestRecord, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Classify a test failure record, providing confidence, top features, and review status."""
        rec_dict = record.model_dump() if isinstance(record, NormalizedTestRecord) else dict(record)

        # Evaluate Phase 2 rule-based categorization for comparison
        raw_rule_cat = self.rule_categorizer.categorize(
            status=rec_dict.get("status", "failed"),
            error_type=rec_dict.get("error_type"),
            error_message=rec_dict.get("error_message"),
            stack_trace=rec_dict.get("stack_trace"),
            log_excerpt=rec_dict.get("log_excerpt"),
        )
        rule_based_canonical = normalize_failure_category(raw_rule_cat)

        # Prepare single-row DataFrame for pipeline
        df_input = pd.DataFrame([{
            "test_name": rec_dict.get("test_name", ""),
            "error_type": rec_dict.get("error_type", ""),
            "error_message": rec_dict.get("error_message", ""),
            "stack_trace": rec_dict.get("stack_trace", ""),
            "log_excerpt": rec_dict.get("log_excerpt", ""),
            "duration_seconds": rec_dict.get("duration_seconds", 0.0),
        }])

        # Check if record has completely empty textual content
        has_content = any(
            bool(str(df_input[col].iloc[0]).strip())
            for col in ["error_type", "error_message", "stack_trace", "log_excerpt"]
        )

        if not has_content:
            return {
                "test_id": rec_dict.get("test_id", "unknown"),
                "test_name": rec_dict.get("test_name", "unknown"),
                "predicted_category": "unknown",
                "confidence": 0.0,
                "probabilities": {cat: 0.0 for cat in self.classes_},
                "model_version": self.model_version,
                "needs_review": True,
                "review_reason": "Insufficient log content: error fields and stack trace are empty.",
                "top_features": [],
                "rule_based_category": rule_based_canonical,
                "is_agreement": rule_based_canonical == "unknown",
            }

        # Predict with model
        classifier = self._get_classifier()
        has_proba = hasattr(self.pipeline, "predict_proba")

        if has_proba:
            proba_matrix = self.pipeline.predict_proba(df_input)[0]
            pred_idx = int(np.argmax(proba_matrix))
            raw_predicted_class = self.classes_[pred_idx]
            confidence = round(float(proba_matrix[pred_idx]), 4)
            probabilities = {
                cls_name: round(float(p), 4)
                for cls_name, p in zip(self.classes_, proba_matrix)
            }
        else:
            raw_predicted_class = self.pipeline.predict(df_input)[0]
            confidence = 1.0
            probabilities = {raw_predicted_class: 1.0}

        # Apply confidence thresholding and human review triggers
        needs_review = False
        review_reason = None
        predicted_category = raw_predicted_class

        if confidence < self.confidence_threshold:
            needs_review = True
            predicted_category = "unknown"
            review_reason = f"Low prediction confidence ({confidence:.2f} < threshold {self.confidence_threshold:.2f})."
        elif raw_predicted_class == "unknown":
            needs_review = True
            review_reason = "Model classified failure as unknown/unrecognized pattern."

        # Extract top contributing features/tokens
        top_features = self._extract_top_features(df_input, raw_predicted_class)

        is_agreement = (predicted_category == rule_based_canonical) or (
            raw_predicted_class == rule_based_canonical
        )

        return {
            "test_id": rec_dict.get("test_id", "unknown"),
            "test_name": rec_dict.get("test_name", "unknown"),
            "predicted_category": predicted_category,
            "raw_predicted_category": raw_predicted_class,
            "confidence": confidence,
            "probabilities": probabilities,
            "model_version": self.model_version,
            "needs_review": needs_review,
            "review_reason": review_reason,
            "top_features": top_features,
            "rule_based_category": rule_based_canonical,
            "is_agreement": is_agreement,
        }

    def _extract_top_features(self, df_input: pd.DataFrame, predicted_class: str) -> List[str]:
        """Identify key tokens/n-grams from the input text driving the classification."""
        try:
            if not hasattr(self.pipeline, "named_steps"):
                return []
            extractor = self.pipeline.named_steps.get("extractor")
            preprocessor = self.pipeline.named_steps.get("preprocessor")
            classifier = self.pipeline.named_steps.get("classifier")

            if not extractor or not preprocessor or not classifier:
                return []

            # Extract text transformer
            text_trans = preprocessor.named_transformers_.get("text")
            if not text_trans or not hasattr(text_trans, "get_feature_names_out"):
                return []

            feature_names = text_trans.get_feature_names_out()
            extracted_df = extractor.transform(df_input)
            transformed_sparse = text_trans.transform(extracted_df["combined_text"])

            nonzero_indices = transformed_sparse.nonzero()[1]
            if len(nonzero_indices) == 0:
                return []

            # Score features by classifier coefficients if available, else by TF-IDF weight
            if hasattr(classifier, "coef_") and predicted_class in self.classes_:
                cls_idx = self.classes_.index(predicted_class)
                weights = classifier.coef_[cls_idx]
                scored = [(feature_names[i], float(weights[i] * transformed_sparse[0, i])) for i in nonzero_indices]
                scored.sort(key=lambda x: x[1], reverse=True)
                return [name for name, _ in scored[:5]]
            else:
                # Rank by TF-IDF weight
                scored = [(feature_names[i], float(transformed_sparse[0, i])) for i in nonzero_indices]
                scored.sort(key=lambda x: x[1], reverse=True)
                return [name for name, _ in scored[:5]]
        except Exception as e:
            logger.debug(f"Could not compute top features: {e}")
            return []
