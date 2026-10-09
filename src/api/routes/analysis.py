"""Route for real-time root-cause triage analysis using ML and rule-based models."""

import logging
from fastapi import APIRouter, Depends, HTTPException, status

from src.api.schemas import FailureAnalyzeRequest, FailureAnalyzeResponse
from src.services.analysis_service import AnalysisService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/analyze", tags=["Failure Triage & Analysis"])


def get_analysis_service() -> AnalysisService:
    return AnalysisService()


@router.post("", response_model=FailureAnalyzeResponse)
def analyze_failure(
    payload: FailureAnalyzeRequest,
    analysis: AnalysisService = Depends(get_analysis_service),
) -> FailureAnalyzeResponse:
    """Analyze a failed test record or raw log text, infer ML root cause, and return debugging actions."""
    try:
        req_data = payload.model_dump()
        result = analysis.analyze_failure(req_data, save_to_db=payload.save_to_db)
        return FailureAnalyzeResponse(
            test_id=result["test_id"],
            test_name=result["test_name"],
            normalized_record=result["normalized_record"],
            rule_based_category=result["rule_based_category"],
            ml_prediction=result["ml_prediction"],
            predicted_category=result["predicted_category"],
            confidence=result["confidence"],
            model_version=result["model_version"],
            needs_review=result["needs_review"],
            review_reason=result["review_reason"],
            top_features=result["top_features"],
            suggested_actions=result["suggested_actions"],
            evidence=result["evidence"],
        )
    except ValueError as ve:
        logger.warning(f"Validation error during failure analysis: {ve}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid failure payload: {ve}",
        )
    except Exception as e:
        logger.error(f"Internal error during failure analysis: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while triaging the failure record.",
        )
