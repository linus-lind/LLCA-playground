import unittest

import torch

from llca.models.modules.continuous_variable_encoder import ContinuousVariableEncoder

_NUM_VARS = 3
_EMBEDDING_DIM = 4


def _encoder() -> ContinuousVariableEncoder:
    return ContinuousVariableEncoder(_NUM_VARS, _EMBEDDING_DIM)


class ContinuousVariableEncoderTest(unittest.TestCase):
    def test_missing_positions_select_the_learned_missing_embedding(self) -> None:
        encoder = _encoder()
        # Variable 1 is missing: age -1 with a NaN value, exactly as MaskedPanel stores it.
        x = torch.tensor([[1.0, float("nan"), 0.5]])
        age = torch.tensor([[0.0, -1.0, 2.0]])

        out = encoder(x, age)

        self.assertEqual(out.shape, (1, _NUM_VARS, _EMBEDDING_DIM))
        self.assertTrue(torch.isfinite(out).all())
        # The missing variable's embedding is the learned parameter, independent of its value.
        self.assertTrue(torch.equal(out[0, 1], encoder.missing[1]))

    def test_missing_value_magnitude_does_not_change_the_output(self) -> None:
        """A missing slot is defined by age < 0 alone; its stored value must not leak."""
        encoder = _encoder()
        age = torch.tensor([[0.0, -1.0, 2.0]])
        baseline = encoder(torch.tensor([[1.0, float("nan"), 0.5]]), age)
        perturbed = encoder(torch.tensor([[1.0, 1e9, 0.5]]), age)

        self.assertTrue(torch.equal(baseline, perturbed))

    def test_nan_missing_values_do_not_produce_non_finite_gradients(self) -> None:
        """Regression: NaN inputs at age -1 kept a finite forward but poisoned the backward
        pass through torch.where's discarded branch (0 * NaN), making the gradient norm
        non-finite and crashing gradient clipping during training."""
        encoder = _encoder()
        x = torch.tensor([[1.0, float("nan"), 0.5]])
        age = torch.tensor([[0.0, -1.0, 2.0]])

        loss = encoder(x, age).sum()
        self.assertTrue(torch.isfinite(loss).all())
        loss.backward()  # type: ignore[no-untyped-call]

        for parameter in encoder.parameters():
            self.assertIsNotNone(parameter.grad)
            assert parameter.grad is not None
            self.assertTrue(
                torch.isfinite(parameter.grad).all(),
                msg="missing-value handling produced a non-finite gradient",
            )

    def test_gradients_reach_observed_variable_projections(self) -> None:
        encoder = _encoder()
        x = torch.tensor([[1.0, float("nan"), 0.5]])
        age = torch.tensor([[0.0, -1.0, 2.0]])

        encoder(x, age).sum().backward()  # type: ignore[no-untyped-call]

        observed = encoder.projections[0]
        assert isinstance(observed, torch.nn.Linear)
        assert observed.weight.grad is not None
        self.assertGreater(float(observed.weight.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
