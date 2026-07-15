import unittest

from omegaconf import OmegaConf

from llca.core.resolvers import register_resolvers


class ResolverTest(unittest.TestCase):
    def test_neg_resolver_is_idempotent_and_preserves_numeric_type(self) -> None:
        register_resolvers()
        register_resolvers()
        config = OmegaConf.create({"integer": "${neg:2}", "floating_point": "${neg:1.5}"})

        self.assertEqual(config.integer, -2)
        self.assertEqual(config.floating_point, -1.5)


if __name__ == "__main__":
    unittest.main()
