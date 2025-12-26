from typing import Dict, List

from torch import stack as torch_stack
from torch import Tensor
from torch_geometric.data import Batch

class DoubleTowerCollater:
    def __call__(self, batch_list):
        """
        Input: List of dicts from Dataset.__getitem__
               [{'drug_data': Data, 'haplo_data': Data, 'targets': {...}}, ...]
        Output: Dict with Batched graphs and Stacked targets
        """
        # 1. Separate the components
        drug_graphs = [sample['drug_data'] for sample in batch_list]
        haplo_graphs = [sample['haplo_data'] for sample in batch_list]
        
        # 2. Batch the Graphs using PyG's Batch.from_data_list
        # This creates a super-graph with disconnected components, preserving edge_indices
        batch_drug = Batch.from_data_list(drug_graphs)
        batch_haplo = Batch.from_data_list(haplo_graphs)

        # 3. Stack Targets
        # We assume targets are already Tensors from the Dataset
        target_keys = batch_list[0]['targets'].keys()
        batched_targets = {}
        
        for key in target_keys:
            # Stack creates (Batch_Size, ...)
            batched_targets[key] = torch_stack([sample['targets'][key] for sample in batch_list])

        return {
            "drug_batch": batch_drug,
            "haplo_batch": batch_haplo,
            "targets": batched_targets
        }