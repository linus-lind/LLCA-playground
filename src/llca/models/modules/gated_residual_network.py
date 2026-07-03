from torch import Tensor, nn

from llca.models.modules.gate_add_norm import GateAddNorm


class GatedResidualNetwork(nn.Module):
    """Apply a context-conditioned nonlinear transform with a gated residual path.

    The module acts on the final axis and therefore supports tensors such as ``[N, D]``
    and ``[N, T, D]`` without assigning meaning to their leading dimensions. An optional
    context is projected additively into the hidden representation and may broadcast over
    leading axes. The skip path is projected whenever input and output widths differ.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int | None = None,
        context_size: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        output_size = output_size if output_size is not None else input_size
        self.skip: nn.Module = (
            nn.Identity() if input_size == output_size else nn.Linear(input_size, output_size)
        )
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.context_projection: nn.Module | None = (
            nn.Linear(context_size, hidden_size, bias=False) if context_size is not None else None
        )
        self.elu = nn.ELU()
        self.hidden_projection = nn.Linear(hidden_size, hidden_size)
        self.gate_add_norm = GateAddNorm(hidden_size, output_size, dropout)

    def forward(self, a: Tensor, context: Tensor | None = None) -> Tensor:
        """Transform ``a`` while preserving its leading dimensions."""
        hidden = self.input_projection(a)
        if self.context_projection is not None and context is not None:
            hidden = hidden + self.context_projection(context)
        eta1 = self.hidden_projection(self.elu(hidden))
        out: Tensor = self.gate_add_norm(eta1, self.skip(a))
        return out
