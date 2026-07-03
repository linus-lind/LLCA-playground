from __future__ import annotations

import os
import random
from collections.abc import Mapping
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor

_CUBLAS_WORKSPACE_CONFIG = ":4096:8"


def configure_determinism(enabled: bool) -> None:
    """Configure PyTorch and cuDNN execution for deterministic or benchmarked kernels."""
    if enabled:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", _CUBLAS_WORKSPACE_CONFIG)
    torch.use_deterministic_algorithms(enabled)
    torch.backends.cudnn.benchmark = not enabled
    torch.backends.cudnn.deterministic = enabled


def seed_everything(seed: int) -> None:
    """Reset Python, NumPy, CPU PyTorch, and all available CUDA RNG streams."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> dict[str, Any]:
    """Capture every RNG stream required to resume stochastic training exactly."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """Restore a previously captured cross-library RNG snapshot."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(cast(Tensor, state["torch"]).cpu())
    if torch.cuda.is_available() and "cuda" in state:
        cuda_states = [item.cpu() for item in cast(list[Tensor], state["cuda"])]
        torch.cuda.set_rng_state_all(cuda_states)
