"""Single-asset random-forest classifier scored as a directional portfolio position.

Trains a random-forest classifier on the target asset's point-in-time features to predict its
forward-return direction; the positive-class probability becomes a directional portfolio score
through the shared single-asset base. Unlike the logistic baseline the tree family is
scale-invariant, so the design matrix is only mean-imputed, never standardized.

Every meaningful :class:`~sklearn.ensemble.RandomForestClassifier` constructor argument is
configurable through Hydra. They fall into three groups: statistical hyperparameters that
control the fitted model's bias/variance and are candidates for the search grid (tree depth and
leaf/split sizes, feature and sample subsampling, class weighting, pruning); the ensemble size
``n_estimators``, kept fixed at a stable value rather than searched; and structural/runtime
controls (``bootstrap``, ``oob_score``, ``monotonic_cst``, ``verbose``, ``warm_start``, and the
seed / ``n_jobs`` supplied by the training policy) that never enter the grid.
"""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestClassifier  # type: ignore[import-untyped]

from llca.models.estimators.single_asset_tabular import SingleAssetClassifierEstimator
from llca.training.modules.sklearn_config import SklearnTrainingConfig


class RandomForestClassifierEstimator(SingleAssetClassifierEstimator):
    """Fit a random-forest classifier to one asset's return direction."""

    _MODEL_NAME = "rf"
    _BUNDLE_ARTIFACT = "rf_bundle"
    _BUNDLE_FILENAME = "rf.pkl"
    _STANDARDIZE = False

    def _construct(self, training: SklearnTrainingConfig) -> Any:
        config = self._config
        return RandomForestClassifier(
            # Ensemble size: fixed high for variance reduction, not a search dimension.
            n_estimators=int(config.n_estimators),
            # Statistical hyperparameters (candidates for the search grid). Absent optional
            # entries fall back to scikit-learn's own defaults; null is meaningful (unlimited)
            # for max_depth / max_leaf_nodes / max_samples / class_weight.
            criterion=self._tuned("criterion", "gini"),
            max_depth=self._optional_int("max_depth"),
            min_samples_split=self._tuned_number("min_samples_split", 2),
            min_samples_leaf=self._tuned_number("min_samples_leaf", 1),
            min_weight_fraction_leaf=self._tuned_float("min_weight_fraction_leaf", 0.0),
            max_features=self._tuned("max_features", "sqrt"),
            max_leaf_nodes=self._optional_int("max_leaf_nodes"),
            min_impurity_decrease=self._tuned_float("min_impurity_decrease", 0.0),
            class_weight=self._tuned("class_weight"),
            ccp_alpha=self._tuned_float("ccp_alpha", 0.0),
            max_samples=self._optional_number("max_samples"),
            # Structural controls (fixed per experiment, never searched).
            bootstrap=bool(config.get("bootstrap", True)),
            oob_score=bool(config.get("oob_score", False)),
            monotonic_cst=config.get("monotonic_cst"),
            # Runtime controls: reproducibility seed and parallelism from the training policy.
            random_state=int(training.seed),
            n_jobs=int(training.n_jobs),
            verbose=int(config.get("verbose", 0)),
            warm_start=bool(config.get("warm_start", False)),
        )
