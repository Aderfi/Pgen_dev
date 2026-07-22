"""Compositional label table: enumerate label tuples, embed, and decode.

:class:`CompositionalLabelTable` holds a fixed catalogue of human-readable
labels, each keyed by a class-index tuple over the single-label axes that
:class:`~src.model.architectures.heads.compose.ComposeHead` composes (column
order matches :meth:`ComposeHead.single_label_axes`). :meth:`build` embeds
every tuple through the same hard path used for ground truth
(:meth:`ComposeHead.embed_tuples`), caching a `[T, out_dim]` table. Given a
composed vector `z` (e.g. from :meth:`ComposeHead.forward`), :meth:`decode`
finds the nearest table rows by cosine similarity, and :meth:`agreement`
checks whether the nearest row's tuple matches an externally computed
per-axis argmax.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torch.nn import functional as F

if TYPE_CHECKING:
    from torch import nn

    from src.model.architectures.heads.compose import ComposeHead

__all__ = ["CompositionalLabelTable"]


class CompositionalLabelTable:
    """Fixed catalogue of label tuples, embedded and searchable by similarity.

    Args:
        tuples: Per-row class-index tuples, one per single-label axis, in the
            column order produced by `ComposeHead.single_label_axes()`.
        labels: Human-readable label per tuple; `labels[i]` decodes `tuples[i]`.
    """

    def __init__(self, tuples: list[tuple[int, ...]], labels: list[str]) -> None:
        if len(tuples) != len(labels):
            raise ValueError(
                f"tuples and labels must have the same length, "
                f"got {len(tuples)} tuples and {len(labels)} labels"
            )
        self.tuples = tuples
        self.labels = labels
        self._table: Tensor | None = None
        self._tuple_index: Tensor | None = None

    def build(self, compose: ComposeHead, axis_embeddings: nn.ModuleDict) -> Tensor:
        """Embed every tuple through `compose`'s hard path and cache the table.

        Args:
            compose: The `ComposeHead` whose `compose_mlp` produces the shared
                label space.
            axis_embeddings: Per-axis class-embedding tables
                (`AxisHeads.axis_embeddings`).

        Returns:
            `[T, out_dim]` embedded label table (also cached on `self`).
        """
        device = next(compose.compose_mlp.parameters()).device
        tuple_index = torch.as_tensor(self.tuples, dtype=torch.long, device=device)
        table = compose.embed_tuples(tuple_index, axis_embeddings)
        self._tuple_index = tuple_index
        self._table = table
        return table

    def _require_table(self) -> Tensor:
        if self._table is None:
            raise RuntimeError(
                "CompositionalLabelTable.build() must be called before "
                "decode()/agreement()"
            )
        return self._table

    def decode(self, z: Tensor, top_k: int = 3) -> list[list[tuple[str, float]]]:
        """Decode composed vectors `z` to their nearest labels by cosine similarity.

        Args:
            z: `[B, out_dim]` composed label vectors.
            top_k: Number of nearest labels to return per row, most similar first.

        Returns:
            One list per row of `z`, each a list of `(label, score)` pairs
            sorted by descending cosine similarity.
        """
        table = self._require_table()
        z_norm = F.normalize(z, dim=-1)
        table_norm = F.normalize(table, dim=-1)
        sims = z_norm @ table_norm.t()  # [B, T]

        k = min(top_k, sims.shape[-1])
        top_scores, top_idx = torch.topk(sims, k=k, dim=-1)

        results: list[list[tuple[str, float]]] = []
        for row_scores, row_idx in zip(top_scores.tolist(), top_idx.tolist()):
            results.append(
                [(self.labels[idx], score) for idx, score in zip(row_idx, row_scores)]
            )
        return results

    def agreement(self, z: Tensor, argmax_tuple: Tensor) -> Tensor:
        """Check whether each row's nearest table tuple matches its argmax tuple.

        Args:
            z: `[B, out_dim]` composed label vectors.
            argmax_tuple: `[B, n_axes]` per-axis argmax class indices.

        Returns:
            `[B]` bool tensor, True where the nearest table row's tuple
            equals `argmax_tuple` for that row.
        """
        table = self._require_table()
        if self._tuple_index is None:
            raise RuntimeError(
                "CompositionalLabelTable.build() must be called before agreement()"
            )

        z_norm = F.normalize(z, dim=-1)
        table_norm = F.normalize(table, dim=-1)
        sims = z_norm @ table_norm.t()  # [B, T]
        nearest_idx = sims.argmax(dim=-1)  # [B]

        nearest_tuples = self._tuple_index[nearest_idx]  # [B, n_axes]
        argmax_tuple = argmax_tuple.to(
            device=nearest_tuples.device, dtype=nearest_tuples.dtype
        )
        return (nearest_tuples == argmax_tuple).all(dim=-1)
