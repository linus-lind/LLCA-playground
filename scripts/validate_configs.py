"""Compose and validate every supported Hydra application root."""

from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig

from llca.mappers import validate_config
from llca.mappers.analytics.config_validator import validate_analytics_config

CONFIG_ROOT = Path(__file__).resolve().parents[1] / "hydra" / "configs"
TRAINING_ROOT = CONFIG_ROOT / "training"
ANALYTICS_ROOT = CONFIG_ROOT / "analytics"
TRAINING_CONFIGS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("fmg-ctct-2", ("experiment=fmg-ctct-2",)),
    (
        "fmg-ctct-2-mse",
        ("experiment=fmg-ctct-2", "loss=mse"),
    ),
    (
        "fmg-ctct-1",
        ("experiment=fmg-ctct-1", "model.target.entity_id=1"),
    ),
    ("fmg-ctt", ("experiment=fmg-ctt", "model.target.entity_id=1")),
    ("fmg-clstm", ("experiment=fmg-clstm", "model.target.entity_id=1")),
    ("rf", ("experiment=rf",)),
    ("elastic-net", ("experiment=elastic-net",)),
    ("equal-weight", ("experiment=equal-weight",)),
    ("inverse-volatility", ("experiment=inverse-volatility",)),
)


def _compose(root: Path, config_name: str, overrides: tuple[str, ...] = ()) -> DictConfig:
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(root.resolve())):
        return compose(config_name=config_name, overrides=list(overrides))


def main() -> None:
    for label, overrides in TRAINING_CONFIGS:
        config = _compose(TRAINING_ROOT, "train", overrides)
        validate_config(config)
        print(f"validated train experiment={label}")
    analytics = _compose(ANALYTICS_ROOT, "analytics")
    validate_analytics_config(analytics)
    print("validated analytics")


if __name__ == "__main__":
    main()
