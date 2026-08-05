"""`python -m llca` entry point.

The training pipeline is the default command; it is implemented in
:mod:`llca.training.__main__` (analytics lives in :mod:`llca.analytics.__main__`).
This shim runs the training module as ``__main__`` so Hydra resolves its config
search path from the training package, keeping ``python -m llca`` equivalent to
``python -m llca.training``.
"""

from __future__ import annotations

import runpy

if __name__ == "__main__":
    runpy.run_module("llca.training", run_name="__main__", alter_sys=True)
