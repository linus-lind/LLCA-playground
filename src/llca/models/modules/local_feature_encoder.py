from __future__ import annotations

from torch import Tensor, nn

from llca.models.modules.continuous_variable_encoder import ContinuousVariableEncoder
from llca.models.modules.conv_layer import ConvLayer
from llca.models.modules.gate_add_norm import GateAddNorm
from llca.models.modules.gated_residual_network import GatedResidualNetwork
from llca.models.modules.temporal_cnn import TemporalCNN
from llca.models.modules.variable_selection_network import VariableSelectionNetwork


class LocalFeatureEncoder(nn.Module):
    """Encode independent local feature histories into fixed-width temporal embeddings.

    Inputs ``features`` and ``age`` have shape ``[N, T + B, F]``, where ``B`` is the
    convolutional buffer. Variables are first embedded as ``[N, T + B, F, E]`` and
    selected before projection to model width ``D``. The causal convolution consumes the
    buffer, so the returned encoding has shape ``[N, T, D]`` and selection weights have
    shape ``[N, T + B, F]``. No information is mixed between rows in ``N``.

    Both the selection and the pre-CNN-gate GRN condition on external context vectors.
    Those two vectors are distinct signals but share one width, so a single ``context_dim``
    configures both conditioning projections; leaving it ``None`` disables conditioning.
    """

    def __init__(
        self,
        num_features: int,
        model_dim: int,
        feature_embedding_dim: int,
        cnn_layers: list[ConvLayer],
        dropout: float,
        context_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.encoder = ContinuousVariableEncoder(num_features, feature_embedding_dim)
        self.variable_selection = VariableSelectionNetwork(
            num_vars=num_features,
            embedding_dim=feature_embedding_dim,
            dropout=dropout,
            context_size=context_dim,
        )
        self.feature_projection: nn.Module = (
            nn.Identity()
            if feature_embedding_dim == model_dim
            else nn.Linear(feature_embedding_dim, model_dim)
        )
        self.cnn = TemporalCNN(model_dim, cnn_layers, dropout=dropout)
        self.gate = GateAddNorm(model_dim, model_dim, dropout=dropout)
        self.grn = GatedResidualNetwork(
            model_dim, model_dim, dropout, model_dim, context_size=context_dim
        )

    @property
    def buffer_size(self) -> int:
        return self.cnn.buffer_size

    def forward(
        self,
        features: Tensor,
        age: Tensor,
        selection_context: Tensor | None = None,
        grn_context: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Encode one collection of local histories and return variable weights."""
        embedded = self.encoder(features, age)
        selection_context = _broadcast_over_time(selection_context)
        selected, weights = self.variable_selection(embedded, age, selection_context)
        projected = self.feature_projection(selected)
        convolved = self.cnn(projected)
        residual = projected[:, self.buffer_size :, :]
        gated = self.gate(convolved, residual)
        encoded: Tensor = self.grn(gated, _broadcast_over_time(grn_context))
        return encoded, weights


def _broadcast_over_time(context: Tensor | None) -> Tensor | None:
    """Insert a singleton time axis into a point-in-time context ``[N, context_dim]``."""
    return context.unsqueeze(1) if context is not None else None
