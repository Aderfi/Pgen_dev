from typing import Optional, Union, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.data import Data, HeteroData
from torch_geometric.data.batch import Batch
from torch import Tensor

class GATv2Tower(nn.Module):
    """
    Torre de codificación basada en GATv2 (Graph Attention Network v2).
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.1,
        edge_dim: Optional[int] = None,
        pooling: str = 'mean'
    ):
        super().__init__()
        self.num_layers:int = num_layers 
        self.pooling: str = pooling
        self.dropout: float = dropout

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.skips = nn.ModuleList()

        # Input Layer
        self.convs.append(GATv2Conv(in_channels, hidden_channels, heads=heads, edge_dim=edge_dim, concat=True))
        self.bns.append(nn.BatchNorm1d(hidden_channels * heads))
        self.skips.append(nn.Linear(in_channels, hidden_channels * heads))

        # Hidden Layers
        # Calculamos la dimensión de entrada de las capas ocultas basándonos en la concatenación de cabezas
        hidden_input = hidden_channels * heads
        
        for _ in range(num_layers - 1):
            self.convs.append(GATv2Conv(hidden_input, hidden_channels, heads=heads, edge_dim=edge_dim, concat=True))
            self.bns.append(nn.BatchNorm1d(hidden_input))
            self.skips.append(nn.Linear(hidden_input, hidden_input))

        # Proyección final post-pooling
        self.post_pool_mlp = nn.Sequential(
            nn.Linear(hidden_input, hidden_input),
            nn.ReLU(),
            nn.Linear(hidden_input, out_channels)
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor], batch: torch.Tensor) -> torch.Tensor:
        """
        x: Características de los nodos [Num_Nodes, Num_Features]
        edge_index: Conectividad del grafo [2, Num_Edges]
        edge_attr: Características de los enlaces [Num_Edges, Edge_Dim] (opcional)
        batch: Vector de asignación de nodos a grafos en el batch [Num_Nodes]
        """
        
        for i in range(self.num_layers):
            x_in = x
            
            # 1. Message Passing (Attention)
            x = self.convs[i](x, edge_index, edge_attr=edge_attr)
            
            # 2. Skip Connection (Residual) + BatchNorm + Activation
            # Proyectamos la entrada original si las dimensiones no coinciden para la suma residual
            if self.skips[i] is not None:
                x_in = self.skips[i](x_in)
            
            x = x + x_in
            x = self.bns[i](x)
            x = F.elu(x) # ELU suele funcionar mejor con GATs que ReLU
            x = F.dropout(x, p=self.dropout, training=self.training)

        # 3. Global Pooling (Readout) - Convierte nodo-level a grafo-level
        if self.pooling == 'mean':
            x_graph = global_mean_pool(x, batch)
        elif self.pooling == 'add':
            x_graph = global_add_pool(x, batch)
        elif self.pooling == 'max':
            x_graph = global_max_pool(x, batch)
        else:
            x_graph = global_mean_pool(x, batch)

        # 4. Final Projection
        return self.post_pool_mlp(x_graph)

class PharmagenTwoTower(nn.Module):
    def __init__(
        self,
        # Configuración Torre Fármaco
        drug_in_features: int,
        drug_edge_dim: int,
        drug_hidden_dim: int,
        
        # Configuración Torre Haplotipo (GATv2)
        haplo_in_features: int,
        haplo_edge_dim: int, # Ejemplo: peso de LD o distancia genómica
        haplo_hidden_dim: int,
        
        # Configuración Global
        embedding_dim: int, # Dimensión del espacio latente compartido
        target_dims: dict[str, int], # { 'IC50': 1, 'SideEffect': 1, 'Class': 3 }
        
        # Hiperparámetros
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.target_dims: dict[str, int] = target_dims

        # --- Torre 1: Fármaco ---
        # Se puede usar GATv2 también aquí, o cambiar a GINEConv si se prefiere,
        # pero GATv2 es excelente para capturar farmacóforos complejos.
        self.drug_tower = GATv2Tower(
            in_channels=drug_in_features,
            hidden_channels=drug_hidden_dim,
            out_channels=embedding_dim,
            num_layers=num_layers,
            heads=heads,
            edge_dim=drug_edge_dim,
            dropout=dropout,
            pooling='mean' # Promedio para representar la molécula entera
        )

        # --- Torre 2: Haplotipo / Genoma ---
        # REQUERIMIENTO: Utilizar GATv2 para los grafos del genoma.
        self.haplo_tower: GATv2Tower = GATv2Tower(
            in_channels=haplo_in_features,
            hidden_channels=haplo_hidden_dim,
            out_channels=embedding_dim,
            num_layers=num_layers,
            heads=heads,
            edge_dim=haplo_edge_dim,
            dropout=dropout,
            pooling='add' # 'add' suele ser mejor para sumar efectos de variantes genéticas
        )

        # --- Interaction & Prediction Heads ---
        # Combinamos las dos torres. La dimensión será embedding_dim * 2 (concatenación)
        combined_dim = embedding_dim * 2
        
        # Creamos una red densa (MLP) para procesar la interacción antes de los cabezales finales
        self.interaction_mlp = nn.Sequential(
            nn.Linear(combined_dim, combined_dim),
            nn.LayerNorm(combined_dim),
            nn.ELU(),
            nn.Dropout(dropout)
        )

        # Cabezales dinámicos (Multi-task)
        self.heads = nn.ModuleDict()
        for target_name, out_dim in target_dims.items():
            self.heads[target_name] = nn.Sequential(
                nn.Linear(combined_dim, combined_dim // 2),
                nn.ELU(),
                nn.Linear(combined_dim // 2, out_dim)
            )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(
        self, 
        drug_data: Data, 
        haplo_data: Data
    ) -> Dict[str, Tensor]:
        """
        Forward pass dinámico.
        
        Args:
            drug_data: Batch de grafos moleculares (PyG Batch).
            haplo_data: Batch de grafos de haplotipos (PyG Batch).
        """

        # Validaciones básicas
        if not (hasattr(drug_data, 'x') and hasattr(drug_data, 'edge_index')):
            raise ValueError("El objeto drug_data no tiene la estructura de grafo necesaria (x, edge_index)")
        
        
        # 1. Forward Torre Fármaco
        # Extraemos atributos de aristas si existen, si no None
        drug_edge_attr = getattr(drug_data, 'edge_attr', None)
        drug_emb = self.drug_tower(
            x=drug_data.x, 
            edge_index=drug_data.edge_index, 
            edge_attr=drug_edge_attr, 
            batch=getattr(drug_data, 'batch', None)
        )

        if not (hasattr(haplo_data, 'x') and hasattr(haplo_data, 'edge_index')):
            raise ValueError("El objeto haplo_data no tiene la estructura de grafo necesaria (x, edge_index)")
            
        # 2. Forward Torre Haplotipo (GATv2)
        haplo_edge_attr = getattr(haplo_data, 'edge_attr', None)
        haplo_emb = self.haplo_tower(
            x=haplo_data.x, 
            edge_index=haplo_data.edge_index, 
            edge_attr=haplo_edge_attr, 
            batch=getattr(haplo_data, 'batch', None)
        )

        # 3. Interacción (Concatenación)
        # Aquí se unen el espacio químico y el espacio biológico
        combined = torch.cat([drug_emb, haplo_emb], dim=1)
        
        # 4. Procesamiento conjunto
        interacted = self.interaction_mlp(combined)

        # 5. Predicciones Dinámicas
        outputs = {}
        for target_name, head_layer in self.heads.items():
            outputs[target_name] = head_layer(interacted)
            
        return outputs