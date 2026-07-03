from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor


class Standardizer:
    """Standardize the final tensor axis with NaN-tolerant fitted statistics.

    All leading dimensions are treated as observations, so both ``[R, F]`` and
    ``[N, T, F]`` follow the same contract. Missing or degenerate results are mapped to
    zero after scaling, representing no standardized information. Statistics retain only
    shape ``[F]`` and can be restored independently of model weights.
    """

    def __init__(self, mean: Tensor, std: Tensor) -> None:
        self._mean = mean
        self._std = std

    @classmethod
    def fit(cls, values: Tensor) -> Standardizer:
        """Estimate per-feature mean and population standard deviation over leading axes."""
        flat = values.reshape(-1, values.shape[-1])
        mean = torch.nanmean(flat, dim=0)
        variance = torch.nanmean((flat - mean) ** 2, dim=0)
        std = variance.sqrt().clamp_min(1e-8)
        return cls(torch.nan_to_num(mean), std)

    def transform(self, values: Tensor) -> Tensor:
        """Apply fitted statistics by broadcasting over every leading dimension."""
        return torch.nan_to_num((values - self._mean) / self._std, nan=0.0)

    def state_dict(self) -> dict[str, Tensor]:
        """Return serializable fitted statistics without model-specific metadata."""
        return {"mean": self._mean, "std": self._std}

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, Tensor],
        *,
        device: torch.device | str | None = None,
    ) -> Standardizer:
        """Restore statistics on an explicit execution device when one is required.

        `torch.load(map_location=...)` also relocates nested scaler tensors. Estimators
        standardizing compact CPU buffers before lazy GPU window transfer therefore pass
        `device="cpu"`, independently of the device holding the neural-network weights.
        """
        mean = state["mean"]
        std = state["std"]
        if device is not None:
            mean = mean.to(device)
            std = std.to(device)
        return cls(mean, std)
