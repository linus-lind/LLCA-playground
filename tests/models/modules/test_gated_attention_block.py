import unittest

import torch

from llca.models.modules.gated_attention_block import GatedAttentionBlock
from llca.models.modules.sinusoidal_positional_encoding import SinusoidalPositionalEncoding

_D_MODEL = 8
_N_HEADS = 2


class GatedAttentionBlockTest(unittest.TestCase):
    def test_block_has_no_context_conditioning_pathway(self) -> None:
        block = GatedAttentionBlock(_D_MODEL, _N_HEADS, dropout=0.0)
        self.assertIsNone(block.grn.context_projection)

    def test_forward_rejects_a_context_argument(self) -> None:
        block = GatedAttentionBlock(_D_MODEL, _N_HEADS, dropout=0.0)
        x = torch.randn(2, 3, _D_MODEL)
        # nn.Module.__call__ is Any-typed, so this is a runtime (not static) rejection.
        with self.assertRaises(TypeError):
            block(x, context=torch.randn(2, _D_MODEL))

    def test_forward_preserves_shape_and_propagates_gradient(self) -> None:
        block = GatedAttentionBlock(
            _D_MODEL,
            _N_HEADS,
            dropout=0.0,
            positional_encoding=SinusoidalPositionalEncoding(_D_MODEL, max_len=3),
        )
        x = torch.randn(2, 3, _D_MODEL, requires_grad=True)
        out = block(x)
        out.pow(2).sum().backward()

        self.assertEqual(out.shape, (2, 3, _D_MODEL))
        assert x.grad is not None
        self.assertGreater(float(x.grad.abs().sum()), 0.0)

    def test_key_padding_mask_is_now_the_second_positional_argument(self) -> None:
        block = GatedAttentionBlock(_D_MODEL, _N_HEADS, dropout=0.0)
        x = torch.randn(2, 3, _D_MODEL)
        mask = torch.zeros(2, 3, dtype=torch.bool)
        out = block(x, mask)
        self.assertEqual(out.shape, (2, 3, _D_MODEL))


if __name__ == "__main__":
    unittest.main()
