import unittest

import torch
from torch import nn

from llca.models.modules.conv_layer import ConvLayer
from llca.models.modules.local_feature_encoder import LocalFeatureEncoder

_NUM_FEATURES = 3
_FEATURE_EMBEDDING_DIM = 4
_MODEL_DIM = 16
# Context is wired at the feature embedding width in the FMG models; keep it != model_dim.
_CONTEXT_DIM = _FEATURE_EMBEDDING_DIM


def _encoder(context_dim: int | None) -> LocalFeatureEncoder:
    return LocalFeatureEncoder(
        num_features=_NUM_FEATURES,
        model_dim=_MODEL_DIM,
        feature_embedding_dim=_FEATURE_EMBEDDING_DIM,
        cnn_layers=[ConvLayer(2, 2, 1, 0, 0)],
        dropout=0.0,
        context_dim=context_dim,
    )


class LocalFeatureEncoderTest(unittest.TestCase):
    def test_single_context_dim_configures_both_conditioning_projections(self) -> None:
        self.assertNotEqual(_CONTEXT_DIM, _MODEL_DIM)
        encoder = _encoder(_CONTEXT_DIM)
        selection_projection = encoder.variable_selection.weight_grn.context_projection
        grn_projection = encoder.grn.context_projection
        assert isinstance(selection_projection, nn.Linear)
        assert isinstance(grn_projection, nn.Linear)

        # One context_dim drives both conditioning inputs...
        self.assertEqual(selection_projection.in_features, _CONTEXT_DIM)
        self.assertEqual(grn_projection.in_features, _CONTEXT_DIM)
        # ...while each projects into its own hidden width (E for selection, D for the GRN).
        self.assertEqual(selection_projection.out_features, _FEATURE_EMBEDDING_DIM)
        self.assertEqual(grn_projection.out_features, _MODEL_DIM)

    def test_forward_consumes_two_context_vectors_of_context_dim_width(self) -> None:
        encoder = _encoder(_CONTEXT_DIM)
        rows, timesteps = 2, 3
        window = timesteps + encoder.buffer_size
        features = torch.randn(rows, window, _NUM_FEATURES, requires_grad=True)
        age = torch.zeros(rows, window, _NUM_FEATURES)
        selection_context = torch.randn(rows, _CONTEXT_DIM)
        grn_context = torch.randn(rows, _CONTEXT_DIM)

        encoded, weights = encoder(features, age, selection_context, grn_context)
        encoded.pow(2).sum().backward()

        self.assertEqual(encoded.shape, (rows, timesteps, _MODEL_DIM))
        self.assertEqual(weights.shape, (rows, window, _NUM_FEATURES))
        assert features.grad is not None
        self.assertGreater(float(features.grad.abs().sum()), 0.0)

    def test_context_dim_none_disables_both_conditioning_projections(self) -> None:
        encoder = _encoder(None)
        self.assertIsNone(encoder.variable_selection.weight_grn.context_projection)
        self.assertIsNone(encoder.grn.context_projection)


if __name__ == "__main__":
    unittest.main()
