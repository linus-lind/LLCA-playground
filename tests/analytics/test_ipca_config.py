from __future__ import annotations

import unittest
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, open_dict

from llca.mappers.analytics.config_validator import validate_analytics_config
from llca.mappers.analytics.mapper import analytics_data_requirements
from llca.mappers.modules.config_validation_error import ConfigValidationError

_CONFIG_DIR = str(
    (Path(__file__).resolve().parents[2] / "hydra" / "configs" / "analytics").resolve()
)


def _analytics_config(*overrides: str) -> DictConfig:
    with initialize_config_dir(config_dir=_CONFIG_DIR, version_base=None):
        return compose(config_name="analytics", overrides=list(overrides))


class IpcaHydraConfigurationTest(unittest.TestCase):
    def test_portfolio_only_config_has_no_classification_settings(self) -> None:
        cfg = _analytics_config()

        validate_analytics_config(cfg)

        self.assertNotIn("probability_bins", cfg.analytics)
        self.assertNotIn("classification_threshold", cfg.analytics)

    def test_rejects_unknown_or_retired_top_level_analytics_fields(self) -> None:
        for field in ("probability_bins", "classification_threshold", "unknown_setting"):
            with self.subTest(field=field):
                cfg = _analytics_config()
                with open_dict(cfg.analytics):
                    cfg.analytics[field] = 10

                with self.assertRaisesRegex(
                    ConfigValidationError,
                    rf"analytics has unsupported field\(s\).*{field}",
                ):
                    validate_analytics_config(cfg)

    def test_rejects_unknown_nested_analytics_fields(self) -> None:
        cfg = _analytics_config()
        with open_dict(cfg.analytics.models[0]):
            cfg.analytics.models[0].unsupported = True
        with open_dict(cfg.analytics.factor_analysis):
            cfg.analytics.factor_analysis.unsupported = True

        with self.assertRaises(ConfigValidationError) as raised:
            validate_analytics_config(cfg)

        message = str(raised.exception)
        self.assertIn("analytics.models[0] has unsupported field", message)
        self.assertIn("analytics.factor_analysis has unsupported field", message)

    def test_default_analytics_config_owns_a_valid_ipca_pipeline(self) -> None:
        cfg = _analytics_config()

        validate_analytics_config(cfg)

        self.assertEqual(cfg.analytics.factor_analysis.aligning_dataset, "asset_returns")
        ipca = cfg.analytics.factor_analysis.ipca
        self.assertTrue(ipca.enabled)
        self.assertNotIn("primary_dataset", ipca.inputs)
        self.assertEqual(ipca.inputs.returns.dataset, "asset_returns")
        self.assertEqual(ipca.inputs.returns.return_type, "simple")
        self.assertEqual(ipca.inputs.returns.realization_lag, 2)
        self.assertTrue(ipca.inputs.returns.excess)
        self.assertEqual(ipca.inputs.characteristics.dataset, "firm_characteristics")
        self.assertNotIn("columns", ipca.inputs.characteristics)
        self.assertNotIn("specification", ipca)
        self.assertNotIn("missing", ipca)
        self.assertEqual(ipca.min_characteristic_coverage, 0.8)
        self.assertEqual(ipca.max_age.default, 126)
        self.assertEqual(ipca.max_age.columns.interest_coverage, 378)
        self.assertIn("asset_returns", cfg.data.datasets)
        self.assertIn("asset_returns", cfg.preprocessing)
        self.assertEqual(cfg.features.asset_returns[0].name, "simple_change")
        self.assertEqual(cfg.features.asset_returns[0].shift, -2)
        self.assertGreater(len(cfg.features.firm_characteristics), ipca.n_factors)

    def test_ipca_factor_count_cannot_exceed_characteristics_plus_constant(self) -> None:
        cfg = _analytics_config()
        outputs = len(cfg.features.firm_characteristics)
        cfg.analytics.factor_analysis.ipca.n_factors = outputs + 2

        with self.assertRaisesRegex(ConfigValidationError, "n_factors must not exceed"):
            validate_analytics_config(cfg)

    def test_rejects_non_positive_factor_count(self) -> None:
        cfg = _analytics_config()
        cfg.analytics.factor_analysis.ipca.n_factors = 0

        with self.assertRaisesRegex(ConfigValidationError, "n_factors"):
            validate_analytics_config(cfg)

    def test_rejects_out_of_range_min_characteristic_coverage(self) -> None:
        cfg = _analytics_config()
        cfg.analytics.factor_analysis.ipca.min_characteristic_coverage = 1.1

        with self.assertRaisesRegex(
            ConfigValidationError, "min_characteristic_coverage must be in"
        ):
            validate_analytics_config(cfg)

    def test_rolling_beta_window_must_leave_residual_degrees_of_freedom(self) -> None:
        cfg = _analytics_config("analytics.factor_analysis.rolling_beta_window=7")

        with self.assertRaisesRegex(
            ConfigValidationError,
            "rolling_beta_window must exceed.*intercept plus 6 factors",
        ):
            validate_analytics_config(cfg)

    def test_disabled_ipca_skips_source_validation_and_preparation_requirements(self) -> None:
        cfg = _analytics_config("analytics.factor_analysis.ipca.enabled=false")
        cfg.analytics.factor_analysis.aligning_dataset = "unavailable_primary"
        cfg.analytics.factor_analysis.ipca.inputs.returns.dataset = "unavailable_returns"
        cfg.analytics.factor_analysis.ipca.inputs.characteristics.dataset = (
            "unavailable_characteristics"
        )
        for group in (cfg.data.datasets, cfg.preprocessing, cfg.features):
            with open_dict(group):
                del group["asset_returns"]
                del group["firm_characteristics"]

        validate_analytics_config(cfg)
        requirements, data_view = analytics_data_requirements(cfg.analytics)

        required_datasets = {requirement.name for requirement in requirements.datasets}
        self.assertEqual(data_view, "independent")
        self.assertNotIn("unavailable_primary", required_datasets)
        self.assertNotIn("unavailable_returns", required_datasets)
        self.assertNotIn("unavailable_characteristics", required_datasets)

    def test_all_factor_sources_use_root_pipeline_feature_outputs(self) -> None:
        cfg = _analytics_config()

        validate_analytics_config(cfg)

        self.assertNotIn("datasets", cfg.analytics)
        self.assertNotIn("index", cfg.analytics)
        self.assertEqual(cfg.analytics.return_realization_lag, 2)
        self.assertEqual(cfg.analytics.risk_free.dataset, "fama_french")
        self.assertEqual(cfg.analytics.risk_free.column, "rf")
        factors = cfg.analytics.factor_analysis.factors
        self.assertEqual(factors.dataset, "fama_french")
        self.assertEqual(list(factors.ff6), ["mktrf", "smb", "hml", "rmw", "cma", "umd"])
        self.assertEqual(factors.market, "mktrf")
        timing = cfg.analytics.factor_analysis.timing.instruments
        self.assertEqual(timing.dataset, "macro")
        self.assertEqual(
            list(timing.columns), ["nfci", "baa10y_spread", "tips_10y_yield", "move", "vix"]
        )
        self.assertEqual(cfg.analytics.factor_analysis.timing.instrument_lag, 1)
        self.assertIn("fama_french", cfg.data.datasets)
        self.assertIn("fama_french", cfg.preprocessing)
        self.assertIn("fama_french", cfg.features)
        factor_features = {str(spec.get("as")): spec for spec in cfg.features.fama_french}
        for column in [*factors.ff6, cfg.analytics.risk_free.column]:
            self.assertEqual(factor_features[str(column)].shift, -2)
        timing_features = {str(spec.get("as")): spec for spec in cfg.features.macro}
        for column in timing.columns:
            self.assertIsNone(timing_features[str(column)].get("shift"))

    def test_return_like_factor_sources_must_declare_daily_frequency(self) -> None:
        cfg = _analytics_config()
        cfg.data.datasets.fama_french.frequency = "weekly"
        cfg.data.datasets.asset_returns.frequency = "weekly"

        with self.assertRaises(ConfigValidationError) as raised:
            validate_analytics_config(cfg)

        message = str(raised.exception)
        self.assertIn("risk_free dataset 'fama_french' must declare frequency 'daily'", message)
        self.assertIn(
            "ipca.inputs.returns dataset 'asset_returns' must declare frequency 'daily'", message
        )

    def test_global_zero_realization_lag_resolves_without_manual_feature_overrides(self) -> None:
        cfg = _analytics_config("analytics.return_realization_lag=0")
        # An omitted shift and an explicit zero have identical same-period semantics.
        with open_dict(cfg.features.fama_french[0]):
            del cfg.features.fama_french[0]["shift"]

        validate_analytics_config(cfg)

        self.assertEqual(cfg.analytics.factor_analysis.ipca.inputs.returns.realization_lag, 0)
        self.assertEqual(cfg.features.asset_returns[0].shift, 0)
        self.assertTrue(all(spec.get("shift") in (None, 0) for spec in cfg.features.fama_french))

    def test_timing_reference_accepts_transformed_alias(self) -> None:
        cfg = _analytics_config()
        cfg.features.macro = [
            {
                "name": "simple_change",
                "column": "vix",
                "horizon": 1,
                "shift": 0,
                "as": "vix_change",
            }
        ]
        cfg.analytics.factor_analysis.timing.instruments.columns = ["vix_change"]

        validate_analytics_config(cfg)

    def test_timing_reference_rejects_leading_feature(self) -> None:
        cfg = _analytics_config()
        with open_dict(cfg.features.macro[0]):
            cfg.features.macro[0].shift = -1

        with self.assertRaisesRegex(ConfigValidationError, "must not set a non-zero shift"):
            validate_analytics_config(cfg)

    def test_timing_reference_rejects_feature_lag_outside_instrument_lag(self) -> None:
        cfg = _analytics_config()
        with open_dict(cfg.features.macro[0]):
            cfg.features.macro[0].shift = 1

        with self.assertRaisesRegex(ConfigValidationError, "instrument_lag is the single"):
            validate_analytics_config(cfg)

    def test_rejects_negative_timing_instrument_lag(self) -> None:
        cfg = _analytics_config()
        cfg.analytics.factor_analysis.timing.instrument_lag = -1

        with self.assertRaisesRegex(ConfigValidationError, "instrument_lag must be >= 0"):
            validate_analytics_config(cfg)

    def test_rejects_factor_output_not_aligned_to_global_realization_lag(self) -> None:
        cfg = _analytics_config()
        cfg.features.fama_french[0].shift = -1

        with self.assertRaisesRegex(
            ConfigValidationError, "match analytics.return_realization_lag"
        ):
            validate_analytics_config(cfg)

    def test_rejects_factor_column_not_created_by_selected_features(self) -> None:
        cfg = _analytics_config()
        cfg.analytics.factor_analysis.factors.ff6[0] = "not_a_factor_output"
        cfg.analytics.factor_analysis.factors.market = "smb"

        with self.assertRaisesRegex(ConfigValidationError, "not produced by features.fama_french"):
            validate_analytics_config(cfg)

    def test_hydra_data_pipeline_groups_compose_from_analytics_defaults(self) -> None:
        # The Analytics app composes its own data/preprocessing/features/masking groups. Post
        # refactor each group ships a single ``default`` option; the named training variants
        # (e.g. ``sp500-crsp-compustat``, ``analytics-features``) deliberately live under
        # configs/training and are intentionally not exposed as analytics alternatives, so
        # composition selects the analytics defaults and still resolves the sp500 data identity.
        cfg = _analytics_config(
            "data=default",
            "preprocessing=default",
            "features=default",
            "masking=sp500-membership",
        )

        validate_analytics_config(cfg)
        self.assertEqual(cfg.data.name, "sp500-crsp-compustat")
        for group in ("data", "preprocessing", "features", "masking"):
            self.assertIn(group, cfg)

    def test_rejects_return_type_or_lag_inconsistent_with_feature(self) -> None:
        cfg = _analytics_config()
        cfg.analytics.factor_analysis.ipca.inputs.returns.return_type = "log"
        cfg.analytics.factor_analysis.ipca.inputs.returns.realization_lag = 1

        with self.assertRaises(ConfigValidationError) as raised:
            validate_analytics_config(cfg)

        message = str(raised.exception)
        self.assertIn("must use 'log_change'", message)
        self.assertIn("must set shift to -1", message)

    def test_rejects_staleness_override_for_unknown_characteristic(self) -> None:
        cfg = _analytics_config()
        with open_dict(cfg.analytics.factor_analysis.ipca.max_age.columns):
            cfg.analytics.factor_analysis.ipca.max_age.columns.not_a_characteristic = 100

        with self.assertRaisesRegex(ConfigValidationError, "unselected or unknown characteristics"):
            validate_analytics_config(cfg)


if __name__ == "__main__":
    unittest.main()
