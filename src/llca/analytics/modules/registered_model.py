from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from omegaconf import DictConfig

from llca.analytics.modules.analytics_config import RegisteredModelConfig


@dataclass(frozen=True, slots=True)
class RegisteredModelMetadata:
    """Registry identity and immutable test-window metadata without a loaded estimator."""

    config: RegisteredModelConfig
    run_id: str
    model_uri: str
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    pipeline_config: DictConfig
    data_manifest: dict[str, Any]
