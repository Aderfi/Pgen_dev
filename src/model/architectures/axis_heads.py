"""Per-axis prediction heads for the factorized-axes model output.

Each :class:`~src.model.architectures.config.AxisSpec` describes one
prediction head. Two families of axes exist:

* **Single-label axes** — ``kind`` is ``"multiclass"`` or ``"ordinal"``, or
  ``kind == "binary"`` with ``dim == 1``. These represent a single
  categorical/ordinal choice and get a companion class-embedding table so
  downstream composition steps can look up an embedding for the predicted
  (or ground-truth) class.
* **Multi-binary axes** — ``kind == "binary"`` with ``dim > 1`` (typically
  produced by a ``MultiLabelBinarizer``). These are independent boolean
  heads and are intentionally excluded from the embedding table: there is
  no single class to embed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import Tensor, nn

if TYPE_CHECKING:
    from src.model.architectures.config import AxisSpec

__all__ = ["AxisHeads", "is_single_label"]


def is_single_label(spec: AxisSpec) -> bool:
    """Return True if `spec` is a *composable* multi-way categorical axis.

    This predicate is the single source of truth for which axes get a class
    embedding and participate in the composed label vector. Both
    :class:`AxisHeads` and ``ComposeHead`` must agree on it, so it lives here
    and is imported rather than duplicated.

    Only ``multiclass`` and ``ordinal`` axes qualify: their target is a class
    index in ``[0, dim)`` that indexes an ``nn.Embedding(dim, ...)`` cleanly.
    A ``binary`` axis (a single sigmoid logit, ``dim == 1``, target in
    ``{0, 1}``) is intentionally excluded — its two states do not map onto a
    one-row embedding, and softmax over a single logit is degenerate. Binary
    axes still get their own prediction head and axis loss; they simply do not
    contribute to the composed label embedding.
    """
    return spec.kind in ("multiclass", "ordinal")


class AxisHeads(nn.Module):
    """A `nn.Linear` prediction head per axis, plus class embeddings.

    Args:
        in_dim: Dimensionality of the shared representation fed to every head.
        axes: Mapping of axis name to its `AxisSpec`.
    """

    def __init__(self, in_dim: int, axes: dict[str, AxisSpec]) -> None:
        super().__init__()
        self.axes = axes
        self._single_label_axes: list[str] = [
            name for name, spec in axes.items() if is_single_label(spec)
        ]

        self.heads = nn.ModuleDict(
            {name: nn.Linear(in_dim, spec.dim) for name, spec in axes.items()}
        )
        self.axis_embeddings = nn.ModuleDict(
            {
                name: nn.Embedding(spec.dim, spec.embedding_dim)
                for name, spec in axes.items()
                if name in self._single_label_axes
            }
        )

    def single_label_axes(self) -> list[str]:
        """Names of axes that carry a single categorical/ordinal label.

        These are the only axes with an entry in `axis_embeddings`, and thus
        the only ones eligible for downstream embedding composition.
        """
        return list(self._single_label_axes)

    def forward(self, z: Tensor) -> dict[str, Tensor]:
        """Compute per-axis logits from the shared representation `z`."""
        return {name: head(z) for name, head in self.heads.items()}
