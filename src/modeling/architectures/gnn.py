from torch import Tensor, cat, nn
from torch.nn import functional as F
from torch_geometric.data import Data
from torch_geometric.nn import (
    GATv2Conv,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)
from torch_geometric.nn.norm import GraphNorm


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
        edge_dim: int | None = None,
        pooling: str = "mean",
    ):
        super().__init__()
        self.num_layers: int = num_layers
        self.pooling: str = pooling
        self.dropout: float = dropout

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList() # Normalizaciones de grafo
        self.skips = nn.ModuleList()

        # Input ayer
        curr_in = in_channels
        for i in range(num_layers):
            out_dim = hidden_channels * heads
            self.convs.append(
                GATv2Conv(curr_in, hidden_channels, heads=heads, edge_dim=edge_dim, concat=True)
            )
            self.norms.append(GraphNorm(out_dim))
            self.skips.append(nn.Linear(curr_in, out_dim))
            curr_in = out_dim

        # Proyección final post-pooling
        self.post_pool_mlp = nn.Sequential(
            nn.Linear(curr_in, curr_in),
            nn.ELU(),
            nn.Linear(curr_in, out_channels),
        )

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor | None,
        batch: Tensor,
    ) -> Tensor:
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

            # 2. Optimized: Skip Connection (Residual) + BatchNorm + Activation
            # Use inplace operations where safe for memory efficiency
            if self.skips[i] is not None:
                x_in = self.skips[i](x_in)

            x = x + x_in
            x = self.norms[i](x, batch)
            x = F.elu(x, inplace=True)  # Inplace operation for memory efficiency
            x = F.dropout(x, p=self.dropout, training=self.training, inplace=False)

        # 3. Optimized: Global Pooling with dictionary lookup
        pooling_ops = {
            "mean": global_mean_pool,
            "add": global_add_pool,
            "max": global_max_pool,
        }
        pool_fn = pooling_ops.get(self.pooling, global_mean_pool)
        x_graph = pool_fn(x, batch)

        # 4. Final Projection
        return self.post_pool_mlp(x_graph)


class PharmagenTwoTower(nn.Module):
    def __init__(
        self,
        # Configuración Torre Fármaco
        drug_in_features: int,
        drug_edge_dim: int,
        drug_hidden_dim: int,
        # Configuración Torre Haplotipo
        haplo_in_features: int,
        haplo_edge_dim: int,
        haplo_hidden_dim: int,
        # Configuración Global
        embedding_dim: int,
        target_dims: dict[str, int],
        # Hiperparámetros
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.1,
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
            pooling="mean",  # Promedio para representar la molécula entera
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
            pooling="add",  # 'add' suele ser mejor para sumar efectos de variantes genéticas
        )

        # --- Interaction & Prediction Heads ---
        # Combinamos las dos torres. La dimensión será embedding_dim * 2 (concatenación)
        combined_dim = embedding_dim * 2

        # Creamos una red densa (MLP) para procesar la interacción antes de los cabezales finales
        self.interaction_mlp = nn.Sequential(
            nn.Linear(combined_dim, combined_dim),
            nn.LayerNorm(combined_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(combined_dim, combined_dim),
            nn.ELU(),
        )

        # Cabezales dinámicos (Multi-task)
        self.heads = nn.ModuleDict()
        for target_name, out_dim in target_dims.items():
            self.heads[target_name] = nn.Sequential(
                nn.Linear(combined_dim, combined_dim // 2),
                nn.ELU(),
                nn.Linear(combined_dim // 2, out_dim),
            )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, drug_data: Data, haplo_data: Data) -> dict[str, Tensor]:
        """
        Forward pass dinámico.

        Args:
            drug_data: Batch de grafos moleculares (PyG Batch).
            haplo_data: Batch de grafos de haplotipos (PyG Batch).
        """

        # Validaciones básicas
        if not (hasattr(drug_data, "x") and hasattr(drug_data, "edge_index")):
            raise ValueError(
                "El objeto drug_data no tiene la estructura de grafo necesaria (x, edge_index)"
            )

        # 1. Forward Torre Fármaco
        # Extraemos atributos de aristas si existen, si no None
        drug_emb = self.drug_tower(
            x=drug_data.x,
            edge_index=drug_data.edge_index,
            edge_attr=getattr(drug_data, "edge_attr", None),
            batch=drug_data.batch,
        )

        if not (hasattr(haplo_data, "x") and hasattr(haplo_data, "edge_index")):
            raise ValueError(
                "El objeto haplo_data no tiene la estructura de grafo necesaria (x, edge_index)"
            )

        # 2. Forward Torre Haplotipo (GATv2)
        haplo_emb = self.haplo_tower(
            x=haplo_data.x,
            edge_index=haplo_data.edge_index,
            edge_attr=getattr(haplo_data, "edge_attr", None),
            batch=haplo_data.batch,
        )

        # 3. Interacción (Concatenación)
        # Aquí se unen el espacio químico y el espacio biológico
        combined = cat([drug_emb, haplo_emb], dim=1)

        # 4. Procesamiento conjunto
        interacted = self.interaction_mlp(combined)

        # 5. Predicciones Dinámicas
        outputs = {}
        for target_name, head_layer in self.heads.items():
            outputs[target_name] = head_layer(interacted)

        return outputs
