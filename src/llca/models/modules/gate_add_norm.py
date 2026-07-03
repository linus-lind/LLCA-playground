from torch import Tensor, nn

from llca.models.modules.gated_linear_unit import GatedLinearUnit


class GateAddNorm(nn.Module):
    """Merge a transformed branch with an unchanged residual using gating and normalization.

    Dropout and a gated linear unit are applied only to ``x``. The result must have the
    same trailing width as ``residual``; both tensors may contain arbitrary, broadcast-
    compatible leading dimensions.
    """

    def __init__(self, input_size: int, output_size: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout: nn.Module = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.glu = GatedLinearUnit(input_size, output_size)
        self.norm = nn.LayerNorm(output_size)

    def forward(self, x: Tensor, residual: Tensor) -> Tensor:
        """Return ``LayerNorm(residual + GLU(Dropout(x)))``."""
        out: Tensor = self.norm(residual + self.glu(self.dropout(x)))
        return out
