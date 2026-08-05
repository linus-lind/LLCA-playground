"""Aggregate per-model evidence into a common-universe cross-model comparison."""

from llca.analytics.comparison.aggregation import (
    ComparisonEvaluation,
    ModelEvaluationResult,
    build_comparison,
)
from llca.analytics.comparison.inference import (
    ComparisonInference,
    ComparisonMatrix,
    ModelConfidenceSummary,
    build_comparison_matrices,
    build_model_confidence_summary,
    build_model_significance_frame,
    evaluate_comparison_inference,
)

__all__ = [
    "ComparisonEvaluation",
    "ComparisonInference",
    "ComparisonMatrix",
    "ModelConfidenceSummary",
    "ModelEvaluationResult",
    "build_comparison",
    "build_comparison_matrices",
    "build_model_confidence_summary",
    "build_model_significance_frame",
    "evaluate_comparison_inference",
]
