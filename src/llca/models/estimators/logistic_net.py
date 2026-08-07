"""Single-asset elastic-net logistic classifier scored as a directional portfolio position.

A classical (non-neural, non-tree) statistical baseline: an elastic-net penalized logistic
regression on the target asset's point-in-time features predicts its forward-return direction,
and the positive-class probability becomes a signed portfolio score through the shared base.

The elastic-net mix is driven by ``l1_ratio`` (0.0 = ridge, 1.0 = lasso) with the ``saga``
solver, which is the only solver that supports a mixed L1/L2 penalty. Modern scikit-learn honors
``l1_ratio`` directly, so no separate (now-deprecated) ``penalty`` argument is passed. Standardized,
mean-imputed features feed the classifier because the logistic family is scale-sensitive.
"""

from __future__ import annotations

from typing import Any

from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]

from llca.models.estimators.single_asset_tabular import SingleAssetClassifierEstimator
from llca.training.modules.sklearn_config import SklearnTrainingConfig


class LogisticNetEstimator(SingleAssetClassifierEstimator):
    """Fit an elastic-net logistic regression to one asset's return direction.

    ``l1_ratio`` spans ridge (0.0) to lasso (1.0) and ``C`` is the inverse regularization
    strength; both are search-grid candidates. ``max_iter`` and ``tol`` are convergence
    controls (set robustly, never tuned), and ``fit_intercept``/``class_weight`` are exposed
    for completeness.
    """

    _MODEL_NAME = "elastic-net"
    _BUNDLE_ARTIFACT = "elastic-net_bundle"
    _BUNDLE_FILENAME = "elastic-net.pkl"
    _STANDARDIZE = True

    def _construct(self, training: SklearnTrainingConfig) -> Any:
        config = self._config
        return LogisticRegression(
            # saga is the only solver supporting a mixed L1/L2 penalty; l1_ratio then controls
            # the elastic-net mix directly (no separate, now-deprecated penalty argument).
            solver="saga",
            # Statistical hyperparameters (search-grid candidates).
            l1_ratio=self._tuned_float("l1_ratio"),
            C=self._tuned_float("C"),
            class_weight=self._tuned("class_weight"),
            fit_intercept=bool(config.get("fit_intercept", True)),
            # Convergence controls: set for robust convergence, never searched.
            max_iter=int(config.get("max_iter", 5000)),
            tol=float(config.get("tol", 1e-4)),
            random_state=int(training.seed),
        )
