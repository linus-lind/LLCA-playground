from typing import cast

from torch import Tensor, nn

from llca.models.modules.gate_add_norm import GateAddNorm
from llca.models.modules.gated_residual_network import GatedResidualNetwork


class TemporalLSTM(nn.Module):
    """Reduce a sequence through LSTM state, terminal dropout, and gated refinement.

    Input size, hidden size, and returned output size are all ``model_dim``. Batch-first
    layout is an internal tensor contract rather than a tunable modeling choice. The
    unidirectional constraint preserves that dimensional contract without an additional
    projection and keeps the recurrent state causal within the available history. The
    final pre-LSTM state is the residual for the same Gate/AddNorm and GRN pattern used by
    the attention blocks; ``output_dropout`` is applied to the LSTM branch inside that
    terminal gate.
    """

    def __init__(
        self,
        model_dim: int,
        num_layers: int,
        recurrent_dropout: float,
        output_dropout: float,
        bias: bool,
        bidirectional: bool,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("LSTM num_layers must be positive")
        if not 0.0 <= recurrent_dropout < 1.0:
            raise ValueError("LSTM recurrent_dropout must be in [0, 1)")
        if not 0.0 <= output_dropout < 1.0:
            raise ValueError("LSTM output_dropout must be in [0, 1)")
        if num_layers == 1 and recurrent_dropout != 0.0:
            raise ValueError("LSTM recurrent_dropout must be zero when num_layers is one")
        if bidirectional:
            raise ValueError(
                "bidirectional LSTM would violate the model_dim output contract"
            )

        self.lstm = nn.LSTM(
            input_size=model_dim,
            hidden_size=model_dim,
            num_layers=num_layers,
            bias=bias,
            batch_first=True,
            dropout=recurrent_dropout,
            bidirectional=bidirectional,
        )
        self.gate_add_norm = GateAddNorm(model_dim, model_dim, output_dropout)
        self.grn = GatedResidualNetwork(
            model_dim, model_dim, model_dim, dropout=output_dropout
        )

    def forward(self, sequence: Tensor) -> Tensor:
        """Return the gated top-layer final state as ``[N, D]``."""
        _, state = self.lstm(sequence)
        hidden, _ = cast(tuple[Tensor, Tensor], state)
        gated = self.gate_add_norm(hidden[-1], sequence[:, -1, :])
        return cast(Tensor, self.grn(gated))
