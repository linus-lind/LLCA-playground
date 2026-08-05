"""Software, platform, and accelerator evidence for a training run."""

from __future__ import annotations

import platform
import sys
from collections.abc import Callable
from importlib.metadata import distributions
from typing import Any, cast

import torch

ENVIRONMENT_MANIFEST_SCHEMA_VERSION = 1


def build_environment_manifest() -> dict[str, Any]:
    """Capture the runtime needed to interpret and reproduce a model."""
    packages = sorted(
        (
            {
                "name": str(distribution.metadata.get("Name") or "<unknown>"),
                "version": str(distribution.version),
            }
            for distribution in distributions()
        ),
        key=lambda package: package["name"].casefold(),
    )
    cuda_available = torch.cuda.is_available()
    cudnn_version = cast(Callable[[], int | None], torch.backends.cudnn.version)()
    accelerator: dict[str, Any] = {
        "cuda_available": cuda_available,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": cudnn_version,
        "devices": [],
    }
    if cuda_available:
        accelerator["devices"] = [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "compute_capability": list(torch.cuda.get_device_capability(index)),
            }
            for index in range(torch.cuda.device_count())
        ]
    return {
        "schema_version": ENVIRONMENT_MANIFEST_SCHEMA_VERSION,
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "compiler": platform.python_compiler(),
            "byteorder": sys.byteorder,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "torch": {"version": torch.__version__, "accelerator": accelerator},
        "packages": packages,
    }
