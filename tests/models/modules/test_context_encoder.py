import unittest

import torch

from llca.models.modules.context_encoder import ContextEncoder

# embedding_dim and model_dim are kept deliberately unequal so shape assertions cannot
# pass by coincidence: a context path that leaked model_dim would produce width 16, not 4.
_EMBEDDING_DIM = 4
_MODEL_DIM = 16
_NUM_CONTEXT_VARS = 3
_GRN_NAMES = ("feature_selection", "pre_transformer")


def _encoder() -> ContextEncoder:
    return ContextEncoder(_NUM_CONTEXT_VARS, _EMBEDDING_DIM, _GRN_NAMES, dropout=0.0)


class ContextEncoderTest(unittest.TestCase):
    def test_named_contexts_and_weights_use_embedding_dim_not_model_dim(self) -> None:
        self.assertNotEqual(_EMBEDDING_DIM, _MODEL_DIM)
        encoder = _encoder()
        rows = 5
        context = torch.randn(rows, _NUM_CONTEXT_VARS)
        age = torch.zeros(rows, _NUM_CONTEXT_VARS)

        contexts, weights = encoder(context, age)

        self.assertEqual(set(contexts), set(_GRN_NAMES))
        for name in _GRN_NAMES:
            self.assertEqual(contexts[name].shape, (rows, _EMBEDDING_DIM))
        self.assertEqual(weights.shape, (rows, _NUM_CONTEXT_VARS))
        self.assertEqual(encoder.embedding_dim, _EMBEDDING_DIM)
        # The encoder no longer knows about model_dim at all.
        self.assertFalse(hasattr(encoder, "model_dim"))

    def test_intermediate_representation_is_C_by_embedding_dim(self) -> None:
        encoder = _encoder()
        rows = 2
        context = torch.randn(rows, _NUM_CONTEXT_VARS)
        age = torch.zeros(rows, _NUM_CONTEXT_VARS)

        # Per-variable embedding is [N, C, E] and never the wider [N, C, model_dim].
        embedded = encoder.encoder(context, age)
        self.assertEqual(embedded.shape, (rows, _NUM_CONTEXT_VARS, _EMBEDDING_DIM))

        # Variable selection collapses the variable axis to a single [N, E] vector.
        selected, weights = encoder.variable_selection(embedded, age)
        self.assertEqual(selected.shape, (rows, _EMBEDDING_DIM))
        self.assertEqual(weights.shape, (rows, _NUM_CONTEXT_VARS))

    def test_context_variables_remain_independently_represented(self) -> None:
        """Changing one context variable must not move another variable's embedding."""
        encoder = _encoder()
        context = torch.randn(1, _NUM_CONTEXT_VARS)
        age = torch.zeros(1, _NUM_CONTEXT_VARS)
        baseline = encoder.encoder(context, age)

        perturbed_input = context.clone()
        perturbed_input[0, 0] += 10.0
        perturbed = encoder.encoder(perturbed_input, age)

        self.assertFalse(torch.equal(baseline[0, 0], perturbed[0, 0]))
        for index in range(1, _NUM_CONTEXT_VARS):
            self.assertTrue(torch.equal(baseline[0, index], perturbed[0, index]))

    def test_gradients_reach_embedding_and_named_projection_parameters(self) -> None:
        encoder = _encoder()
        context = torch.randn(4, _NUM_CONTEXT_VARS)
        age = torch.zeros(4, _NUM_CONTEXT_VARS)

        contexts, _ = encoder(context, age)
        loss = torch.stack([value.pow(2).sum() for value in contexts.values()]).sum()
        loss.backward()  # type: ignore[no-untyped-call]

        first_projection = encoder.encoder.projections[0]
        assert isinstance(first_projection, torch.nn.Linear)
        assert first_projection.weight.grad is not None
        self.assertGreater(float(first_projection.weight.grad.abs().sum()), 0.0)
        for name in _GRN_NAMES:
            grn = encoder.context_grns[name]
            self.assertTrue(
                any(
                    parameter.grad is not None and float(parameter.grad.abs().sum()) > 0.0
                    for parameter in grn.parameters()
                )
            )


if __name__ == "__main__":
    unittest.main()
