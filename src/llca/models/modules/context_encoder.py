from collections.abc import Sequence

from torch import Tensor, nn

from llca.models.modules.continuous_variable_encoder import ContinuousVariableEncoder
from llca.models.modules.gated_residual_network import GatedResidualNetwork
from llca.models.modules.variable_selection_network import VariableSelectionNetwork


class ContextEncoder(nn.Module):
    """Encode point-in-time variables into named, independently trainable contexts.

    ``context`` and ``age`` have shape ``[..., C]`` with no required time axis. Variable
    selection reduces them to one ``D``-dimensional representation, from which a separate
    residual projection is learned for every named downstream consumer. The method
    returns ``({name: [..., D]}, weights[..., C])``.
    """

    def __init__(
        self,
        num_context_vars: int,
        model_dim: int,
        grn_names: Sequence[str],
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not grn_names:
            raise ValueError("ContextEncoder requires at least one GRN name")
        if len(set(grn_names)) != len(grn_names):
            raise ValueError(f"ContextEncoder GRN names must be unique, got {list(grn_names)}")
        self.model_dim = model_dim
        self.encoder = ContinuousVariableEncoder(num_context_vars, model_dim)
        self.variable_selection = VariableSelectionNetwork(
            num_context_vars, model_dim, dropout=dropout
        )
        self.context_grns = nn.ModuleDict(
            {
                name: GatedResidualNetwork(model_dim, model_dim, model_dim, dropout=dropout)
                for name in grn_names
            }
        )

    def forward(self, context: Tensor, age: Tensor) -> tuple[dict[str, Tensor], Tensor]:
        """Return named context embeddings and their variable-selection weights."""
        embedded = self.encoder(context, age)
        selected, weights = self.variable_selection(embedded, age)
        contexts: dict[str, Tensor] = {
            name: grn(selected) for name, grn in self.context_grns.items()
        }
        return contexts, weights
