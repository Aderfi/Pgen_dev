"""Model assembly helpers: turn fitted encoders + config into head specs.

``infer_axis_specs`` bridges the gap between what :class:`TargetEncoder`
learned from the training data (per-axis label encoders) and the declarative
:class:`~src.model.architectures.config.AxisSpec` each prediction head needs.
Later tasks add ``create_gnn_model`` to this module, wiring the inferred
specs into :class:`~src.model.architectures.config.PharmagenConfig`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

from src.model.architectures.config import AxisSpec

if TYPE_CHECKING:
    from torch import Tensor

    from src.config.axes import AxesConfig


def _binary_pos_weight(labels: Tensor) -> float | None:
    """Return ``n_negative / n_positive`` for a 0/1-encoded axis.

    Returns ``None`` when there are no positive examples in ``labels`` (the
    ratio would be undefined / infinite), leaving the caller's default in
    place.
    """
    n_pos = int((labels == 1).sum().item())
    n_neg = int((labels == 0).sum().item())
    if n_pos == 0:
        return None
    return n_neg / n_pos


def _apply_override(spec: AxisSpec, overrides: AxesConfig) -> AxisSpec:
    """Return ``spec`` with any non-``None`` TOML override fields applied."""
    override = overrides.overrides.get(spec.name)
    if override is None:
        return spec
    update = override.model_dump(exclude_none=True)
    if not update:
        return spec
    return spec.model_copy(update=update)


def infer_axis_specs(
    encoders: dict[str, LabelEncoder | MultiLabelBinarizer],
    train_targets: dict[str, Tensor],
    multilabel_cols: set[str],
    overrides: AxesConfig,
    default_focal_gamma: float = 0.0,
) -> dict[str, AxisSpec]:
    """Infer one :class:`AxisSpec` per fitted encoder.

    Args:
        encoders: Fitted per-axis encoders from ``TargetEncoder.encoders``.
        train_targets: Per-axis training label tensors, used to derive
            ``pos_weight`` for binary single-label axes.
        multilabel_cols: Names of axes encoded with a
            :class:`MultiLabelBinarizer` (multi-binary heads).
        overrides: Per-axis TOML overrides; any non-``None`` field wins over
            the inferred value.
        default_focal_gamma: ``focal_gamma`` assigned to every axis unless a
            TOML override sets a different value.

    Returns:
        Mapping of axis name to its inferred (and override-applied)
        :class:`AxisSpec`.
    """
    specs: dict[str, AxisSpec] = {}
    for name, encoder in encoders.items():
        pos_weight: float | None = None
        if name in multilabel_cols or isinstance(encoder, MultiLabelBinarizer):
            kind = "binary"
            dim = len(encoder.classes_)
        else:
            n_classes = len(encoder.classes_)
            if n_classes <= 2:
                kind = "binary"
                dim = 1
                pos_weight = _binary_pos_weight(train_targets[name])
            else:
                kind = "multiclass"
                dim = n_classes

        spec = AxisSpec(
            name=name,
            dim=dim,
            kind=kind,
            pos_weight=pos_weight,
            focal_gamma=default_focal_gamma,
        )
        specs[name] = _apply_override(spec, overrides)
    return specs


__all__ = ["infer_axis_specs"]
