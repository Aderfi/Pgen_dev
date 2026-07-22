"""Compose per-axis class embeddings into a single label vector.

:class:`ComposeHead` turns the factorized per-axis predictions produced by
:class:`~src.model.architectures.heads.axis_heads.AxisHeads` into one dense
label-space vector ``z: [B, out_dim]``. Two paths feed the same
``compose_mlp`` so their outputs live in the same space:

* **Soft path** (:meth:`ComposeHead.forward`) — used at training/inference
  time. Each single-label axis' logits are turned into a probability
  distribution and used as convex weights over that axis' embedding table,
  i.e. ``e_k = softmax(logits_k) @ axis_embeddings[k].weight``. This is
  differentiable end-to-end.
* **Hard path** (:meth:`ComposeHead.embed_tuples`) — used to build the label
  table from ground-truth (or enumerated) class-index tuples, looking up
  ``axis_embeddings[k].weight[index]`` directly instead of averaging.

Only single-label axes (``kind in {"multiclass", "ordinal"}`` or
``kind == "binary" and dim == 1``) participate: multi-binary axes have no
single class to embed and are excluded, mirroring
:class:`~src.model.architectures.heads.axis_heads.AxisHeads`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from src.model.architectures.heads.axis_heads import is_single_label

if TYPE_CHECKING:
    from src.model.architectures.config import AxisSpec

__all__ = ["ComposeHead"]


class ComposeHead(nn.Module):
    """Composes single-label axis embeddings into a fixed-size label vector.

    Args:
        axes: Mapping of axis name to its `AxisSpec`.
        out_dim: Dimensionality of the composed label vector `z`.
    """

    def __init__(self, axes: dict[str, AxisSpec], out_dim: int) -> None:
        super().__init__()
        self.axes = axes
        self.out_dim = out_dim
        # Fixed order: both the soft (forward) and hard (embed_tuples) paths
        # concatenate axis embeddings in this order before compose_mlp, so a
        # confident soft prediction lines up with the matching hard tuple.
        self._single_label_axes: list[str] = [
            name for name, spec in axes.items() if is_single_label(spec)
        ]

        in_dim = sum(axes[name].embedding_dim for name in self._single_label_axes)
        self.compose_mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ELU(),
            nn.Linear(out_dim, out_dim),
        )

    def single_label_axes(self) -> list[str]:
        """Names of axes participating in composition, in concatenation order."""
        return list(self._single_label_axes)

    def forward(
        self, logits: dict[str, Tensor], axis_embeddings: nn.ModuleDict
    ) -> Tensor:
        """Soft-compose class-probability-weighted embeddings into `z`.

        Args:
            logits: Per-axis raw logits, e.g. from `AxisHeads.forward`.
            axis_embeddings: Per-axis class-embedding tables (`AxisHeads.axis_embeddings`).

        Returns:
            `z`, shape `[B, out_dim]`, differentiable w.r.t. `logits`.
        """
        parts = []
        for name in self._single_label_axes:
            probs = torch.softmax(logits[name], dim=-1)
            parts.append(probs @ axis_embeddings[name].weight)
        cat = torch.cat(parts, dim=-1)
        return self.compose_mlp(cat)

    def embed_tuples(self, tuples: Tensor, axis_embeddings: nn.ModuleDict) -> Tensor:
        """Hard-compose ground-truth class-index tuples into a label table.

        Args:
            tuples: `[T, n_single_label_axes]` integer class indices, columns
                ordered as `single_label_axes()`.
            axis_embeddings: Per-axis class-embedding tables (`AxisHeads.axis_embeddings`).

        Returns:
            `[T, out_dim]`, produced by the same `compose_mlp` as `forward`.
        """
        parts = []
        for i, name in enumerate(self._single_label_axes):
            parts.append(axis_embeddings[name].weight[tuples[:, i]])
        cat = torch.cat(parts, dim=-1)
        return self.compose_mlp(cat)
