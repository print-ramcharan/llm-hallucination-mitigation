"""Evaluation endpoints used by the interactive research evaluation workspace."""

from fastapi import APIRouter

from src.evaluation.service import EvaluationRunRequest, evaluate

router = APIRouter(prefix="/api/evaluation", tags=["Research Evaluation"])


@router.post(
    "/run",
    summary="Run paired retrieval and answer evaluation against the live application index",
)
def run_evaluation(payload: EvaluationRunRequest):
    """Evaluate annotated queries using the same indexes and pipeline as the application."""
    return evaluate(payload)
