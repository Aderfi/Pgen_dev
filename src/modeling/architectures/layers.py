# Pharmagen - Modeling
# Architecture: Two-Tower Graph Neural Network (GNN)
# Tower A: Drug (Molecular Graph)
# Tower B: Haplotype/Genome (Interaction Graph using GATv2)

from typing import Dict, Any, Union

import torch.nn as nn

from .gnn import PharmagenTwoTower

MODEL_ARCHITECTURES = {
    'DeepFM': None,  # Placeholder for other models
    'TwoTowerGAT': PharmagenTwoTower,
}

def create_model(
    model_name: str, # 'TwoTowerGAT'
    drug_config: Dict[str, int], # {num_features, edge_dim}
    haplo_config: Dict[str, int], # {num_features, edge_dim}
    target_dims: Dict[str, int],
    params: Dict[str, Any]
) -> nn.Module:
    """
    Factory simplificada para instanciar el modelo Two-Tower.
    """
    
    # Extracción segura de parámetros con valores por defecto
    embedding_dim = params.get("embedding_dim", 128)
    hidden_dim = params.get("hidden_dim", 64)
    dropout = params.get("dropout_rate", 0.1)
    layers = params.get("n_layers", 3)
    heads = params.get("heads", 4)

    return PharmagenTwoTower(
        drug_in_features=drug_config['num_features'],
        drug_edge_dim=drug_config.get('edge_dim', 0),
        drug_hidden_dim=hidden_dim,
        
        haplo_in_features=haplo_config['num_features'],
        haplo_edge_dim=haplo_config.get('edge_dim', 0),
        haplo_hidden_dim=hidden_dim,
        
        embedding_dim=embedding_dim,
        target_dims=target_dims,
        num_layers=layers,
        heads=heads,
        dropout=dropout
    )

def create_gnn_model(
    model_name: str, # 'TwoTowerGAT'
    drug_config: Dict[str, int], # {num_features, edge_dim}
    haplo_config: Dict[str, int], # {num_features, edge_dim}
    target_dims: Dict[str, int],
    params: Dict[str, Any]
) -> nn.Module:
    """
    Factory simplificada para instanciar el modelo Two-Tower.
    """
    
    # Extracción segura de parámetros con valores por defecto
    embedding_dim = params.get("embedding_dim", 128)
    hidden_dim = params.get("hidden_dim", 64)
    dropout = params.get("dropout_rate", 0.1)
    layers = params.get("n_layers", 3)
    heads = params.get("heads", 4)

    return PharmagenTwoTower(
        drug_in_features=drug_config['num_features'],
        drug_edge_dim=drug_config.get('edge_dim', 0),
        drug_hidden_dim=hidden_dim,
        
        haplo_in_features=haplo_config['num_features'],
        haplo_edge_dim=haplo_config.get('edge_dim', 0),
        haplo_hidden_dim=hidden_dim,
        
        embedding_dim=embedding_dim,
        target_dims=target_dims,
        num_layers=layers,
        heads=heads,
        dropout=dropout
    )