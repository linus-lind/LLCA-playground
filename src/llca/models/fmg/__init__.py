"""Canonical FMG model family with four concrete research architectures."""

from llca.models.fmg.fmg_clstm import FmgClstm
from llca.models.fmg.fmg_ctct_1 import FmgCtct1
from llca.models.fmg.fmg_ctct_2 import FmgCtct2
from llca.models.fmg.fmg_ctt import FmgCtt

__all__ = ["FmgClstm", "FmgCtct1", "FmgCtct2", "FmgCtt"]
