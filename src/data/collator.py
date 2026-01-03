"""Custom collator for batching drug-haplotype graph pairs."""

from typing import Any

import torch
from torch_geometric.data import Batch, Data


class DoubleTowerCollater:
    """
    Collates drug and haplotype graph data into batches.

    This collator handles the batching of paired drug-haplotype graph data,
    extracting IDs, sanitizing string attributes, and creating PyTorch
    Geometric batches.

    """
    def __init__(self) -> None:
        """Initialize the collator with ID extraction and sanitization rules."""
        # 1. Definimos la estrategia de prioridad para encontrar el ID
        # Buscará en orden: primero 'cid' (drogas), luego 'variant_name' (haplos), etc.
        self.id_priority_keys = ["cid", "variant_name", "graph_id", "name"]

        # 2. Definimos qué atributos textuales deben ser PURGADOS antes
        # de crear el Batch para evitar el TypeError: new(): invalid
        # data type 'str'
        self.keys_to_sanitize = [
            "cid",
            "variant_name",
            "name",
            "smiles",
            "gene_context",
            "graph_id",
        ]

    def _extract_and_sanitize(self, graph_list: list[Data]) -> list[str]:
        """
        Extrae IDs y elimina atributos conflictivos (strings).

        Modifica los objetos 'in-place'.
        """
        extracted_ids = []

        for data in graph_list:
            # A. Extracción Polimórfica del ID
            found_id = "Unknown"
            for key in self.id_priority_keys:
                if hasattr(data, key):
                    val = getattr(data, key)
                    if val is not None:
                        found_id = str(val)
                        break
            extracted_ids.append(found_id)

            # B. Sanitización (Borrado de strings)
            # Es crítico borrar CUALQUIER atributo string antes de
            # llamar a Batch.from_data_list
            for key in self.keys_to_sanitize:
                if hasattr(data, key):
                    delattr(data, key)

        return extracted_ids

    def __call__(self, batch_list: list[dict[str, Any]]) -> dict[str, Any]:
        """Collate a batch of samples into batched graphs and targets.

        Input: List of dicts from Dataset.__getitem__
        Output: Dict with Batched graphs and Stacked targets
        """
        # 1. Separar componentes
        drug_graphs = [sample["drug_data"] for sample in batch_list]
        haplo_graphs = [sample["haplo_data"] for sample in batch_list]

        # 2. Marshalling: Extraer IDs y limpiar strings
        # Esto soluciona tanto el KeyError (busca varias claves)
        # como el TypeError (elimina los strings antes de batching)
        drug_ids = self._extract_and_sanitize(drug_graphs)
        haplo_ids = self._extract_and_sanitize(haplo_graphs)

        # 3. Batching Seguro (Ahora los grafos solo tienen tensores numéricos)
        batch_drug = Batch.from_data_list(drug_graphs)
        batch_haplo = Batch.from_data_list(haplo_graphs)

        # 4. Re-inyección de Metadatos (Opcional, útil para
        # debug/logging). Los pegamos como listas de Python simples,
        # fuera de la estructura tensorial de PyG
        batch_drug["meta_ids"] = drug_ids  # type: ignore[index]
        batch_haplo["meta_ids"] = haplo_ids  # type: ignore[index]

        # 5. Stack Targets
        target_keys = batch_list[0]["targets"].keys()
        batched_targets = {}

        for key in target_keys:
            batched_targets[key] = torch.stack(
                [sample["targets"][key] for sample in batch_list],
            )

        return {
            "drug_batch": batch_drug,
            "haplo_batch": batch_haplo,
            "targets": batched_targets,
        }
