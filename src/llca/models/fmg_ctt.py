from torch import Tensor

from llca.models.fmg_ctcst import FmgTemporalModel
from llca.models.modules.conv_layer import ConvLayer


class FmgCtt(FmgTemporalModel):
    """Allocate one asset exclusively from its own temporal inputs and context.

    Unlike the cross-sectional FMG variants, the model accepts exactly one instrument and
    sends its locally encoded temporal series directly into temporal aggregation. No
    representation, parameter, or computation associated with another asset is present.
    """

    def __init__(
        self,
        num_features: int,
        num_context_vars: int,
        model_dim: int,
        feature_embedding_dim: int,
        sequence_length: int,
        cnn_layers: list[ConvLayer],
        n_heads: int,
        dropout: float,
        score_activation: str,
    ) -> None:
        super().__init__(
            num_features=num_features,
            num_context_vars=num_context_vars,
            model_dim=model_dim,
            feature_embedding_dim=feature_embedding_dim,
            sequence_length=sequence_length,
            cnn_layers=cnn_layers,
            n_heads=n_heads,
            dropout=dropout,
            score_activation=score_activation,
        )

    def forward(
        self,
        features: Tensor,
        feature_age: Tensor,
        context: Tensor,
        context_age: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Return one direct allocation and variable-selection diagnostics."""
        if features.size(0) != 1:
            raise ValueError(
                f"fmg-ctt expects exactly one asset per forward call, got {features.size(0)}"
            )
        local, selection_weights = self._encode_local(
            features, feature_age, context, context_age
        )
        target_embedding = self.aggregation(local)
        allocation = self.head(target_embedding)
        return allocation, selection_weights
