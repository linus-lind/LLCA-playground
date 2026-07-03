"""Configuration mapping for temporal validation split strategies."""

from llca.mappers.split import config_validator
from llca.mappers.split.mapper import build_split

__all__ = ["build_split", "config_validator"]
