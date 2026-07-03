from pathlib import Path

import pandas as pd
from omegaconf import DictConfig

from llca.core.paths import DATA_DIR
from llca.data.index_spec import IndexSpec, index_spec
from llca.data.ingestion import load_dataset
from llca.data.modules.panels import Panels


def build_dataset(spec: DictConfig, index: IndexSpec) -> pd.DataFrame:
    """Load one dataset using the shared canonical index contract."""
    return load_dataset(spec, index)


def build_datasets(data_cfg: DictConfig) -> Panels:
    """Build every named dataset with identical time/entity index semantics."""
    index = index_spec(data_cfg)
    return {name: build_dataset(spec, index) for name, spec in data_cfg.datasets.items()}


def data_source_path(spec: DictConfig) -> Path:
    return DATA_DIR / str(spec.path)
