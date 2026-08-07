"""Derive the supervision label's forward horizon from validated Hydra configuration.

Both the outer split and the inner-CV geometry must purge at least the number of observations
the prediction target looks ahead, so the last training label cannot realize inside the next
scored window. That horizon is a property of how the supervision column is built -- the negative
``shift`` of its feature spec -- so it is read from configuration here once and shared by every
purge validator rather than being restated per call site.
"""

from __future__ import annotations

from omegaconf import DictConfig, ListConfig


def supervision_forward_horizon(cfg: DictConfig) -> int | None:
    """Return the forward horizon (in observations) of the model's supervision label.

    The horizon is ``max(-shift, 0)`` of the feature spec that produces the supervision column:
    a one-step forward close-to-close return (``shift: -1``) looks one observation ahead. Returns
    ``None`` when the model, features, or supervision binding is absent or malformed, leaving the
    caller's other validators to report the underlying structural error.
    """
    model = cfg.get("model")
    features = cfg.get("features")
    if not isinstance(model, DictConfig) or not isinstance(features, DictConfig):
        return None
    supervision = model.get("supervision")
    if not isinstance(supervision, DictConfig):
        return None
    column = supervision.get("column")
    specs = features.get(supervision.get("dataset"))
    if not isinstance(specs, ListConfig) or not isinstance(column, str):
        return None
    for spec in specs:
        if isinstance(spec, DictConfig) and spec.get("as") == column:
            shift = spec.get("shift", 0)
            if isinstance(shift, int) and not isinstance(shift, bool):
                return max(-shift, 0)
    return None
