"""Custom collator for batching drug-haplotype graph pairs.

Handles metadata cleaning and efficient batching for PyG Data objects.
"""

import logging
from typing import Any

import torch
from torch_geometric.data.batch import Batch
from torch_geometric.data.data import Data

logger = logging.getLogger(__name__)

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

        # Priority order for ID extraction
        self.id_priority_keys = ["cid", "variant_name", "graph_id", "name"]

        # Keys to remove (string attributes incompatible with batching)
        self.keys_to_sanitize = [
            "cid",
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
        haplo_graphs = [sample["haplo_data"] for sample in batch_list]

        # 2. Extract IDs if needed
        drug_ids = None
        haplo_ids = None

        if self.inference_mode:
            drug_ids = self._extract_ids(drug_graphs)
            haplo_ids = self._extract_ids(haplo_graphs)

        # 3. Clean metadata
        self._sanitize_graphs(drug_graphs)
        self._sanitize_graphs(haplo_graphs)

        # 4. Batch graphs
        try:
            batch_drug = Batch.from_data_list(drug_graphs)
            batch_haplo = Batch.from_data_list(haplo_graphs)
        except Exception as e:
            logger.error(f"Failed to batch graphs: {e}")
            raise ValueError(f"Graph batching failed: {e}") from e

        # 5. Re-attach metadata if in inference mode
        if self.inference_mode:
            batch_drug.meta_ids = drug_ids # type: ignore[attr-defined]
            batch_haplo.meta_ids = haplo_ids # type: ignore[attr-defined]

        # 6. Stack targets
        first_target_keys = batch_list[0]["targets"].keys()
        batched_targets = {
            key: torch.stack([s["targets"][key] for s in batch_list])
            for key in first_target_keys
        }

        return {
            "drug_batch": batch_drug,
            "haplo_batch": batch_haplo,
            "targets": batched_targets,
        }
