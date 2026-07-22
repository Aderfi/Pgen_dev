"""Model assembly helpers: turn fitted encoders + config into a live model.

``infer_axis_specs`` bridges the gap between what :class:`TargetEncoder`
learned from the training data (per-axis label encoders) and the declarative
:class:`~src.model.architectures.config.AxisSpec` each prediction head needs.
``create_gnn_model`` wires inferred dims + axis specs into a
:class:`~src.model.architectures.config.PharmagenConfig` and instantiates the
resulting :class:`~src.model.architectures.model.PharmagenTwoTower`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

from src.model.architectures.config import AxisSpec, PharmagenConfig
from src.model.architectures.model import PharmagenTwoTower

if TYPE_CHECKING:
    from torch import Tensor

    from src.config.axes import AxesConfig


def _binary_pos_weight(labels: Tensor) -> float | None:
    """Return ``n_negative / n_positive`` for a 0/1-encoded axis.

    Returns ``None`` when the axis is degenerate — no positive examples (the
    ratio would be infinite) or no negative examples (a ``0.0`` weight would
    zero out every positive sample's loss) — leaving the caller's default in
    place rather than injecting a meaningless weight.
    """
    n_pos = int((labels == 1).sum().item())
    n_neg = int((labels == 0).sum().item())
    if n_pos == 0 or n_neg == 0:
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
    # Re-validate the merged spec: model_copy(update=...) bypasses Pydantic
    # validation, so a typo'd override (e.g. an invalid ``kind`` in TOML) must
    # fail here rather than surface later wherever code branches on the field.
    return AxisSpec.model_validate(spec.model_dump() | update)


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
            ``pos_weight`` for binary single-label axes. May be empty (e.g.
            at inference time, when there are no train labels to inspect) —
            axes missing from this mapping simply get ``pos_weight=None``.
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
                if name in train_targets:
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


def create_gnn_model(
    *,
    dims: dict[str, dict[str, int]],
    drug_dim: int,
    geno_dim: int,
    axes: dict[str, AxisSpec],
    params: dict[str, Any],
    switches: dict[str, bool] | None = None,
) -> PharmagenTwoTower:
    """Assemble a :class:`PharmagenTwoTower` from dims, axes, and hyperparameters.

    Args:
        dims: Nested per-tower dim spec as produced by
            ``src.model.engine.base.extract_tower_dims`` (``{"drugs": {...},
            "geno": {...}}``). ``edges`` feeds the edge_dim, ``global``/
            ``admet`` feed the drug tower's auxiliary MLPs, and ``function``
            feeds the genotype tower's auxiliary MLP.
        drug_dim: Drug-tower node feature width, inferred from a real graph.
        geno_dim: Genotype-tower node feature width, inferred from a real
            graph.
        axes: One :class:`AxisSpec` per prediction head (see
            ``infer_axis_specs``).
        params: Model hyperparameters. Requires ``embedding_dim``,
            ``hidden_dim``, ``dropout_rate``, ``n_layers``, ``heads``.
        switches: Structural ablation flags (``use_polypharmacy``,
            ``use_cross_attention``). Both default to ``False`` when omitted
            or when a key is absent.

    Returns:
        An instantiated (not yet ``.to(device)``-moved) ``PharmagenTwoTower``.

    Raises:
        KeyError: If ``params`` is missing a required hyperparameter.
    """
    required = ("embedding_dim", "hidden_dim", "dropout_rate", "n_layers", "heads")
    missing = [key for key in required if key not in params]
    if missing:
        raise KeyError(f"Missing model parameters: {missing}")

    switches = switches or {}
    drug_dims = dims.get("drugs", {})
    geno_dims = dims.get("geno", {})

    cfg = PharmagenConfig(
        drug_in_features=drug_dim,
        drug_edge_dim=drug_dims.get("edges") or None,
        drug_hidden_dim=params["hidden_dim"],
        drug_global_dim=drug_dims.get("global", 0),
        drug_admet_dim=drug_dims.get("admet", 0),
        geno_in_features=geno_dim,
        geno_edge_dim=geno_dims.get("edges") or None,
        geno_hidden_dim=params["hidden_dim"],
        # The geno tower's aux "global" branch is fed the graph-level PGx
        # function vector (``geno_function``), tracked under the "function" key.
        geno_global_dim=geno_dims.get("function", 0),
        embedding_dim=params["embedding_dim"],
        num_layers=params["n_layers"],
        heads=params["heads"],
        dropout=params["dropout_rate"],
        use_polypharmacy=switches.get("use_polypharmacy", False),
        use_cross_attention=switches.get("use_cross_attention", False),
        axes=axes,
    )
    return PharmagenTwoTower(cfg)


__all__ = ["create_gnn_model", "infer_axis_specs"]
