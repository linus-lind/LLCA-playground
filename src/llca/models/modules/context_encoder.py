from __future__ import annotations

from collections.abc import Sequence

from torch import Tensor, nn

from llca.models.modules.continuous_variable_encoder import ContinuousVariableEncoder
from llca.models.modules.gated_residual_network import GatedResidualNetwork
from llca.models.modules.variable_selection_network import VariableSelectionNetwork


class ContextEncoder(nn.Module):
    """Encode point-in-time variables into named, independently trainable contexts.

    ``context`` and ``age`` have shape ``[..., C]`` with no required time axis. Each of the
    ``C`` variables is embedded independently into ``embedding_dim`` (``E``) -- the same
    cheap per-variable width used by the stock-feature encoder -- so the intermediate
    representation is ``[..., C, E]`` and never the wider ``[..., C, D]``. Variable
    selection reduces that to one ``E``-dimensional representation, from which a separate
    residual refinement is learned for every named downstream consumer. Context is only
    ever consumed as conditioning by downstream GRNs (which project it into their own
    hidden width), so it is intentionally never expanded to ``model_dim`` here. The method
    returns ``({name: [..., E]}, weights[..., C])``.
    """

    def __init__(
        self,
        num_context_vars: int,
        embedding_dim: int,
        grn_names: Sequence[str],
        dropout: float,
    ) -> None:
        super().__init__()
        if not grn_names:
            raise ValueError("ContextEncoder requires at least one GRN name")
        if len(set(grn_names)) != len(grn_names):
            raise ValueError(f"ContextEncoder GRN names must be unique, got {list(grn_names)}")
        self.embedding_dim = embedding_dim
        self.encoder = ContinuousVariableEncoder(num_context_vars, embedding_dim)
        self.variable_selection = VariableSelectionNetwork(num_context_vars, embedding_dim, dropout)
        self.context_grns = nn.ModuleDict(
            {
                name: GatedResidualNetwork(embedding_dim, embedding_dim, dropout, embedding_dim)
                for name in grn_names
            }
        )

    def forward(self, context: Tensor, age: Tensor) -> tuple[dict[str, Tensor], Tensor]:
        """Return named ``[..., E]`` context embeddings and their selection weights."""
        embedded = self.encoder(context, age)
        selected, weights = self.variable_selection(embedded, age)
        contexts: dict[str, Tensor] = {
            name: grn(selected) for name, grn in self.context_grns.items()
        }
        return contexts, weights
