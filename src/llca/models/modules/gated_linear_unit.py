import torch
from torch import Tensor, nn


class GatedLinearUnit(nn.Module):
    """Gate a learned value projection with an independent sigmoid projection.

    The operation maps ``[..., input_size]`` to ``[..., output_size]`` and preserves all
    leading dimensions. It lets a surrounding residual block suppress individual output
    channels without removing the skip path.
    """

    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self.gate = nn.Linear(input_size, output_size)
        self.value = nn.Linear(input_size, output_size)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the element-wise learned gate to the projected values."""
        out: Tensor = torch.sigmoid(self.gate(x)) * self.value(x)
        return out
