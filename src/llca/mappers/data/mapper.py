from pathlib import Path

import pandas as pd
from omegaconf import DictConfig

from llca.core.paths import DATA_DIR
from llca.data.index_spec import IndexSpec, index_spec
from llca.data.ingestion import load_dataset, load_datasets
from llca.data.modules.panels import Panels
from llca.pipeline.contracts import DataPlan, DatasetQuery


def build_dataset(spec: DictConfig, index: IndexSpec) -> pd.DataFrame:
    """Load one dataset using the shared canonical index contract."""
    return load_dataset(spec, index)


def build_datasets(data_cfg: DictConfig, plan: DataPlan | None = None) -> Panels:
    """Load only planned logical datasets, sharing scans of identical physical sources."""
    index = index_spec(data_cfg)
    names = list(plan.datasets) if plan is not None else [str(name) for name in data_cfg.datasets]
    specs = {name: data_cfg.datasets[name] for name in names}
    queries = plan.datasets if plan is not None else {name: DatasetQuery() for name in names}
    return load_datasets(
        specs,
        index,
        queries,
        csv_chunk_size=plan.csv_chunk_size if plan is not None else 250_000,
    )


def data_source_path(spec: DictConfig) -> Path:
    return DATA_DIR / str(spec.path)
