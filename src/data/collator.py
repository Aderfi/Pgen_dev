"""Custom collator for batching drug-genotype graph pairs.

Handles metadata cleaning and efficient batching for PyG Data objects.
"""

import logging
from typing import Any

import torch
from torch_geometric.data.batch import Batch
from torch_geometric.data.data import Data

logger = logging.getLogger(__name__)


class PolyData(Data):
    """``Data`` subclass for molecule-level polypharmacy drug graphs.

    A polypharmacy sample packs every molecule of one patient (focal drug +
    kept DDI neighbours) into a single graph: ``x`` is molecule-level
    (one row per molecule), ``mol_to_patient`` maps each molecule to its
    patient, and ``ddi_edge_index`` holds molecule-local DDI edges.

    Vanilla PyG's default ``__inc__`` already offsets any key containing
    ``"index"`` (so plain ``ddi_edge_index`` happens to get offset by
    ``num_nodes`` for free) but does nothing for ``mol_to_patient`` — it
    would just be concatenated verbatim, leaving every patient at index 0.
    This subclass makes both offsets explicit and intentional rather than
    relying on substring-matching in PyG's default implementation:

    - ``ddi_edge_index`` is offset by the running molecule count (batch's
      cumulative ``num_nodes``) and concatenated along dim 1.
    - ``mol_to_patient`` is offset by the running patient count so that
      batching two 1-patient samples yields ``[0, 0, 1, 1]`` instead of
      ``[0, 0, 0, 0]``.
    - ``is_focal`` needs no offset (plain per-molecule flag) and is left to
      PyG's default (dim-0 concatenation, zero increment).
    """

    def __inc__(self, key: str, value: Any, *args: Any, **kwargs: Any) -> Any:
        if key == "ddi_edge_index":
            return self.num_nodes
        if key == "mol_to_patient":
            return int(value.max()) + 1 if value.numel() else 1
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key: str, value: Any, *args: Any, **kwargs: Any) -> Any:
        if key == "ddi_edge_index":
            return 1
        return super().__cat_dim__(key, value, *args, **kwargs)


def _as_poly_data(data: Data) -> None:
    """Re-stamp ``data`` in-place to :class:`PolyData` when it carries
    polypharmacy attrs (``mol_to_patient``).

    ``Data.__setattr__`` redirects plain attribute assignment (including
    ``__class__``) into its internal storage dict, so a normal
    ``data.__class__ = PolyData`` is silently swallowed and never changes
    the real Python class. ``object.__setattr__`` bypasses that override
    and performs the actual class reassignment, which is what lets
    ``Batch.from_data_list`` dispatch to ``PolyData.__inc__``/
    ``__cat_dim__`` for samples built as plain ``Data`` (e.g. by callers
    that don't construct ``PolyData`` directly).
    """
    if hasattr(data, "mol_to_patient") and not isinstance(data, PolyData):
        object.__setattr__(data, "__class__", PolyData)


class DoubleTowerCollater:
    """Collator for two-tower graph datasets.

    Features:
    - Efficient metadata removal for training
    - Optional metadata preservation for inference
    - Type-safe batching

    Example:
        >>> collater = DoubleTowerCollater(inference_mode=False)
        >>> batch = collater(list_of_samples)
    """

    def __init__(self, inference_mode: bool = False):
        """Initialize DoubleTowerCollater.

        Args:
            inference_mode: If True, preserves graph IDs for inference.
        """
        self.inference_mode = inference_mode

        # Priority order for ID extraction (``gene`` identifies genotype graphs)
        self.id_priority_keys = ["cid", "gene", "variant_name", "graph_id", "name"]

        # Keys to remove (string / non-tensor attrs incompatible with batching)
        self.keys_to_sanitize = [
            "cid",
            "gene",
            "labels",
            "variant_name",
            "name",
            "smiles",
            "gene_context",
            "graph_id",
        ]

    def _sanitize_graphs(self, graph_list: list[Data]) -> None:
        """Remove string attributes from graphs (in-place).
        Modifies graphs to contain only tensor data compatible with PyG batching.

        Args:
            graph_list: List of PyG Data objects.
        """
        for data in graph_list:
            for key in self.keys_to_sanitize:
                if hasattr(data, key):
                    delattr(data, key)

    def _extract_ids(self, graph_list: list[Data]) -> list[str]:
        """Extract IDs from graphs before sanitization.

        Args:
            graph_list: List of PyG Data objects.

        Returns:
            List of extracted IDs.
        """
        extracted_ids = []

        for data in graph_list:
            found_id = "Unknown"

            # Try keys in priority order
            for key in self.id_priority_keys:
                val = getattr(data, key, None)
                if val is not None:
                    found_id = str(val)
                    break

            extracted_ids.append(found_id)

        return extracted_ids

    def __call__(self, batch_list: list[dict[str, Any]]) -> dict[str, Any]:
        """Collate samples into batch.

        Args:
            batch_list: List of dictionaries from Dataset.__getitem__().

        Returns:
            Dictionary with batched graphs and stacked targets.

        Raises:
            ValueError:  If batch_list is empty or malformed.
        """
        if not batch_list:
            raise ValueError("Empty batch_list provided to collator")

        # 1. Separate components
        drug_graphs = [sample["drug_data"] for sample in batch_list]
        geno_graphs = [sample["geno_data"] for sample in batch_list]

        # 2. Extract IDs if needed
        drug_ids = None
        geno_ids = None

        if self.inference_mode:
            drug_ids = self._extract_ids(drug_graphs)
            geno_ids = self._extract_ids(geno_graphs)

        # 3. Clean metadata
        self._sanitize_graphs(drug_graphs)
        self._sanitize_graphs(geno_graphs)

        # 3b. Polypharmacy samples (carrying mol_to_patient / ddi_edge_index)
        # need PolyData's __inc__/__cat_dim__ overrides for correct batch-wide
        # offsetting; re-stamp in-place so Batch.from_data_list dispatches to
        # them even when the sample was built as a plain Data object.
        for drug in drug_graphs:
            _as_poly_data(drug)

        # 4. Batch graphs
        try:
            batch_drug = Batch.from_data_list(drug_graphs)
            batch_geno = Batch.from_data_list(geno_graphs)
        except Exception as e:
            logger.error(f"Failed to batch graphs: {e}")
            raise ValueError(f"Graph batching failed: {e}") from e

        # 5. Re-attach metadata if in inference mode
        if self.inference_mode:
            batch_drug.meta_ids = drug_ids  # type: ignore[attr-defined]
            batch_geno.meta_ids = geno_ids  # type: ignore[attr-defined]

        # 6. Stack targets
        first_target_keys = batch_list[0]["targets"].keys()
        batched_targets = {
            key: torch.stack([s["targets"][key] for s in batch_list])
            for key in first_target_keys
        }

        return {
            "drug_batch": batch_drug,
            "geno_batch": batch_geno,
            "targets": batched_targets,
        }
