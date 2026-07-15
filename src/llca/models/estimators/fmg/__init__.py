"""Concrete estimators for the four supported FMG research architectures."""

from llca.models.estimators.fmg.fmg_clstm import FmgClstmEstimator
from llca.models.estimators.fmg.fmg_ctct_1 import FmgCtct1Estimator
from llca.models.estimators.fmg.fmg_ctct_2 import FmgCtct2Estimator
from llca.models.estimators.fmg.fmg_ctt import FmgCttEstimator

__all__ = [
    "FmgClstmEstimator",
    "FmgCtct1Estimator",
    "FmgCtct2Estimator",
    "FmgCttEstimator",
]
