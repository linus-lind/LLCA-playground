import unittest
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from llca.analytics.factors.factor_models import FactorAlpha, TimingModel
from llca.analytics.reporting.factor_tables import (
    FactorAnalysis,
    _ModelFactors,
    build_additional_statistics_table,
    build_factor_alpha_tables,
    build_factor_figures,
)
from llca.analytics.reporting.statistics_tables import _merge_additional_statistics
from llca.analytics.reporting.table_rendering import _formatted_frame
from llca.analytics.reporting.table_types import PublicationTable


def _alpha(columns: tuple[str, ...]) -> FactorAlpha:
    return FactorAlpha(
        alpha=0.0005,
        alpha_std_error=0.0001,
        alpha_t_statistic=5.0,
        alpha_p_value=0.001,
        annualized_alpha=0.126,
        betas={column: float(index + 1) / 10.0 for index, column in enumerate(columns)},
        beta_p_values={column: 0.04 if index == 0 else 0.2 for index, column in enumerate(columns)},
        r_squared=0.65,
        observations=300,
    )


def _timing() -> TimingModel:
    coefficients = {
        "alpha_vix": 0.001,
        "mktrf": 0.8,
        "mktrf_x_vix": 0.2,
        "smb": -0.1,
        "mktrf_squared": 0.3,
    }
    return TimingModel(
        alpha=0.0004,
        alpha_p_value=0.009,
        annualized_alpha=0.1008,
        market_beta=0.8,
        market_beta_p_value=0.001,
        timing_gamma=0.3,
        timing_p_value=0.03,
        r_squared=0.7,
        observations=299,
        coefficients=coefficients,
        coefficient_p_values={column: 0.04 for column in coefficients},
    )


class FactorReportingTest(unittest.TestCase):
    def setUp(self) -> None:
        dates = pd.bdate_range("2020-01-01", periods=20, name="date")
        ff6_columns = ("mktrf", "smb")
        ipca_columns = ("ipca_1", "ipca_2")
        entries = []
        for index, label in enumerate(("A", "B", "C")):
            rolling = pd.DataFrame(
                {
                    "mktrf": np.linspace(0.5, 0.8, len(dates)),
                    "smb": np.linspace(-0.1, 0.1, len(dates)),
                },
                index=dates,
            )
            entries.append(
                _ModelFactors(
                    label=label,
                    excess=pd.Series(0.0, index=dates),
                    ff6=_alpha(ff6_columns),
                    ipca=_alpha(ipca_columns),
                    timing=_timing(),
                    spanning_statistic=3.1 + index,
                    spanning_p_value=0.02 + index * 0.1,
                    rolling_betas=rolling,
                    cumulative_alpha=pd.Series(np.linspace(0.0, 0.1, len(dates)), index=dates),
                )
            )
        self.analysis = FactorAnalysis(
            models=tuple(entries),
            ff6_columns=ff6_columns,
            ipca_columns=ipca_columns,
            market_column="mktrf",
            rolling_beta_window=126,
            alpha_difference=pd.DataFrame(np.nan, index=["A", "B", "C"], columns=["A", "B", "C"]),
            joint_alpha_statistic=2.5,
            joint_alpha_p_value=0.04,
            correction_label="HOLM-adjusted",
        )

    def test_factor_models_use_separate_tables_and_inline_stars(self) -> None:
        tables = build_factor_alpha_tables(self.analysis)
        self.assertEqual(
            [table.title for table in tables],
            ["Fama-French + Momentum", "IPCA", "Conditional Timing"],
        )
        for table in tables:
            self.assertFalse(any("p-value" in str(label) for label in table.frame.index))
        self.assertEqual(list(tables[0].frame.index[1:3]), ["Market", "Size"])
        ff6_display = _formatted_frame(tables[0])
        self.assertIn("(***)", str(ff6_display.iloc[0, 0]))
        self.assertIn("(**)", str(ff6_display.iloc[1, 0]))

    def test_additional_statistics_merge_without_pvalue_rows(self) -> None:
        additional = build_additional_statistics_table(self.analysis)
        base = PublicationTable(
            name="statistical_significance",
            title="Statistical Significance",
            caption="Predictive tests.",
            frame=pd.DataFrame(
                {"A": [1.2], "B": [0.8], "C": [0.4]},
                index=pd.Index(["Predictive statistic"], name="Statistic"),
            ),
            row_formats=("decimal",),
            cell_p_values=pd.DataFrame(
                {"A": [0.01], "B": [0.2], "C": [0.3]},
                index=pd.Index(["Predictive statistic"], name="Statistic"),
            ),
        )
        merged = _merge_additional_statistics(base, additional)
        self.assertEqual(
            list(merged.frame.index),
            [
                "Predictive statistic",
                "Mean-variance spanning statistic (HAC)",
                "Joint zero-alpha J-statistic (HAC)",
            ],
        )
        self.assertEqual(list(merged.frame.columns), ["A", "B", "C", "Joint model set"])
        self.assertFalse(any("p-value" in str(label) for label in merged.frame.index))
        display = _formatted_frame(merged)
        self.assertIn("(**)", str(display.loc["Mean-variance spanning statistic (HAC)", "A"]))
        self.assertIn(
            "(**)",
            str(display.loc["Joint zero-alpha J-statistic (HAC)", "Joint model set"]),
        )

    def test_single_model_additional_statistics_omit_redundant_joint_alpha(self) -> None:
        single = replace(self.analysis, models=self.analysis.models[:1])

        additional = build_additional_statistics_table(single)

        self.assertEqual(
            list(additional.frame.index),
            ["Mean-variance spanning statistic (HAC)"],
        )
        self.assertEqual(list(additional.frame.columns), ["A"])
        self.assertEqual(additional.row_formats, ("decimal",))
        self.assertNotIn("Joint model set", additional.frame.columns)
        self.assertNotIn("jointly zero", additional.caption)
        merged = _merge_additional_statistics(None, additional)
        self.assertNotIn("Joint zero-alpha J-statistic (HAC)", merged.frame.index)

    def test_rolling_beta_figure_uses_two_columns_and_names_window(self) -> None:
        figures = dict(build_factor_figures(self.analysis))
        rolling = figures["rolling_factor_betas"]
        self.assertEqual(len(rolling.axes), 4)
        self.assertFalse(rolling.axes[-1].get_visible())
        self.assertIn("126-Observation Window", rolling._suptitle.get_text())
        legend = rolling.axes[0].get_legend()
        assert legend is not None
        self.assertEqual([text.get_text() for text in legend.get_texts()], ["Market", "Size"])
        for figure in figures.values():
            plt.close(figure)


if __name__ == "__main__":
    unittest.main()
