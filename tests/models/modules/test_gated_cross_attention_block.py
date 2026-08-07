import unittest

import torch

from llca.models.modules.gated_cross_attention_block import GatedCrossAttentionBlock

_D_MODEL = 8
_N_HEADS = 2


class GatedCrossAttentionBlockTest(unittest.TestCase):
    def test_block_has_no_context_conditioning_pathway(self) -> None:
        block = GatedCrossAttentionBlock(_D_MODEL, _N_HEADS, dropout=0.0)
        self.assertIsNone(block.grn.context_projection)

    def test_forward_rejects_a_context_argument(self) -> None:
        block = GatedCrossAttentionBlock(_D_MODEL, _N_HEADS, dropout=0.0)
        query = torch.randn(2, 1, _D_MODEL)
        key_value = torch.randn(2, 4, _D_MODEL)
        # nn.Module.__call__ is Any-typed, so this is a runtime (not static) rejection.
        with self.assertRaises(TypeError):
            block(query, key_value, context=torch.randn(2, _D_MODEL))

    def test_only_query_rows_returned_with_gradient_flowing_to_all_keys(self) -> None:
        block = GatedCrossAttentionBlock(_D_MODEL, _N_HEADS, dropout=0.0)
        query = torch.randn(2, 1, _D_MODEL, requires_grad=True)
        key_value = torch.randn(2, 4, _D_MODEL, requires_grad=True)

        out = block(query, key_value)
        out.pow(2).sum().backward()

        self.assertEqual(out.shape, (2, 1, _D_MODEL))
        assert key_value.grad is not None
        self.assertGreater(float(key_value.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
