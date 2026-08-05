"""Turn one aligned portfolio prediction into objective, portfolio, and signal evidence."""

from llca.analytics.evaluation.predictions import (
    evaluate_predictions,
    require_supported_prediction_kind,
)

__all__ = [
    "evaluate_predictions",
    "require_supported_prediction_kind",
]
