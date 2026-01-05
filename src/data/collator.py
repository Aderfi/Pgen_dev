"""Custom collator for batching drug-haplotype graph pairs."""

from collections.abc import MutableSequence
from typing import Any

import torch
from torch_geometric.data import Batch, Data


class DoubleTowerCollater:
    def __init__(self, inference_mode=False):
        # Si es True, guardamos IDs (lento). Si es False, velocidad máxima.
        self.inference_mode = inference_mode

        # Keys que sabemos que SIEMPRE son tensores y la GPU necesita
        # Esto es más rápido que borrar lo que NO queremos.
        self.allowed_keys = {'x', 'edge_index', 'edge_attr', 'batch', 'ptr'}


        self.id_priority_keys = ["cid", "variant_name", "graph_id", "name"]
        self.keys_to_sanitize = ["cid", "variant_name", "name", "smiles", "gene_context", "graph_id"]

    def _sanitize_fast(self, graph_list: list[Data]) -> list[str]:
        """
        Extrae IDs y elimina atributos conflictivos (strings) de los objetos Data.
        Modifica los objetos 'in-place'.
        """
        for data in graph_list:
            for key in self.keys_to_sanitize:
                if hasattr(data, key):
                    delattr(data, key)

        return graph_list

    def __call__(self, batch_list):
        """
        Input: List of dicts from Dataset.__getitem__
        Output: Dict with Batched graphs and Stacked targets
        """
        # 1. Separar componentes
        drug_graphs = [sample["drug_data"] for sample in batch_list]
        haplo_graphs = [sample["haplo_data"] for sample in batch_list]

        # 2. Marshalling: Extraer IDs y limpiar strings
        # Esto soluciona tanto el KeyError (busca varias claves)
        # como el TypeError (elimina los strings antes de batching)
        self._sanitize_fast(drug_graphs)
        self._sanitize_fast(haplo_graphs)

        # 3. Batching Seguro (Ahora los grafos solo tienen tensores numéricos)
        batch_drug = Batch.from_data_list(drug_graphs)
        batch_haplo = Batch.from_data_list(haplo_graphs)

        # 4. Re-inyección de Metadatos (Opcional, pero útil para debug/logging)
        # Los pegamos como listas de Python simples, fuera de la estructura tensorial de PyG
        if self.inference_mode:
            batch_drug.meta_ids = self._extract_ids(drug_graphs)
            batch_haplo.meta_ids = self._extract_ids(haplo_graphs)


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

    def _extract_ids(self, graph_list):
        # Tu lógica original, movida aquí para usarla solo bajo demanda
        extracted_ids = []
        for data in graph_list:
            found_id = "Unknown"
            for key in self.id_priority_keys:
                val = getattr(data, key, None)
                if val is not None:
                    found_id = str(val)
                    break
            extracted_ids.append(found_id)
        return extracted_ids
