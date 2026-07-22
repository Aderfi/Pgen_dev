"""Focal-anchored pseudo-patient assembly for the polypharmacy tower.

A pseudo-patient is a star graph built around one "focal" drug: the focal
drug plus up to ``max_neighbors`` of its known drug-drug-interaction (DDI)
partners, each connected to the focal drug by an undirected DDI edge.
``PseudoPatientBuilder`` bridges the static :class:`DDIGraph` adjacency
(keyed by PubChem CID) with the on-disk molecular graphs served by
:class:`~src.data.cache.GraphCache`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from src.data.library.ingest.graph_artifact import DDI_EDGE_DIM

if TYPE_CHECKING:
    from torch_geometric.data.data import Data

    from src.data.cache import GraphCache
    from src.data.library.ingest.graph_artifact import DDIGraph

logger = logging.getLogger(__name__)


class PseudoPatientBuilder:
    """Assemble a focal-anchored pseudo-patient sample from a focal drug cid.

    The focal drug is always local node index 0; kept neighbours occupy
    local indices ``1..n`` in the order the DDI graph returns them. A
    neighbour reported by the DDI graph but lacking an on-disk molecular
    graph in ``cache`` (``str(cid) not in cache.drug_index``) is dropped —
    its DDI edge and edge-attr row are dropped together via a single
    filter pass, so the kept-neighbour cid list and its edge-attr rows
    never drift out of alignment with each other.
    """

    def __init__(
        self, ddi: DDIGraph, cache: GraphCache, max_neighbors: int = 8
    ) -> None:
        self.ddi = ddi
        self.cache = cache
        self.max_neighbors = max_neighbors

    def build(self, focal_cid: str) -> dict:
        """Build the pseudo-patient dict for ``focal_cid``.

        Returns a dict with:
            - ``molecules``: ``list[Data]``, focal graph at index 0.
            - ``mol_to_patient``: ``Tensor[long]``, all zeros (one patient).
            - ``ddi_edge_index``: ``Tensor[long, 2, 2n]``, both directions
              per kept neighbour.
            - ``ddi_edge_attr``: ``Tensor[float, 2n, DDI_EDGE_DIM]``, aligned
              row-for-row with ``ddi_edge_index`` columns.
            - ``is_focal``: ``Tensor[long]``, 1 at index 0, 0 elsewhere.
        """
        molecules: list[Data] = [self.cache.get_drug(focal_cid)]
        is_focal: list[int] = [1]

        neigh_cids, neigh_attr = self.ddi.neighbors(
            int(focal_cid), k=self.max_neighbors
        )

        kept_cids: list[int] = []
        kept_rows: list[torch.Tensor] = []
        for cid, row in zip(neigh_cids, neigh_attr, strict=True):
            if str(cid) in self.cache.drug_index:
                kept_cids.append(cid)
                kept_rows.append(row)
            else:
                logger.debug(
                    "Dropping DDI neighbour %s of focal %s: no molecular "
                    "graph in cache",
                    cid,
                    focal_cid,
                )

        for cid in kept_cids:
            molecules.append(self.cache.get_drug(str(cid)))
            is_focal.append(0)

        src: list[int] = []
        dst: list[int] = []
        attr_rows: list[torch.Tensor] = []
        for j, row in enumerate(kept_rows, start=1):
            src.extend((0, j))
            dst.extend((j, 0))
            attr_rows.extend((row, row))

        if attr_rows:
            ddi_edge_index = torch.tensor([src, dst], dtype=torch.long)
            ddi_edge_attr = torch.stack(attr_rows, dim=0)
        else:
            ddi_edge_index = torch.empty((2, 0), dtype=torch.long)
            ddi_edge_attr = torch.empty((0, DDI_EDGE_DIM))

        mol_to_patient = torch.zeros(len(molecules), dtype=torch.long)
        # ``long`` chosen (not float) — is_focal is a boolean/categorical
        # flag consumed as an index/mask, not a continuous feature.
        is_focal_t = torch.tensor(is_focal, dtype=torch.long)

        return {
            "molecules": molecules,
            "mol_to_patient": mol_to_patient,
            "ddi_edge_index": ddi_edge_index,
            "ddi_edge_attr": ddi_edge_attr,
            "is_focal": is_focal_t,
        }


__all__ = ["PseudoPatientBuilder"]
