from pathlib import Path
from typing import Dict, Union

import pandas as pd
import torch
from torch_geometric.data import Data

from src.config.manager import PROJECT_ROOT

PROJECT_ROOT = Path(PROJECT_ROOT)


class DrugLoader:
    """
    Gestiona la carga de grafos de fármacos desde archivos almacenados en disco.
    Cada fármaco se representa como un objeto Data de PyTorch Geometric.
    """

    def __init__(
        self, drugs_dir: str | Path = PROJECT_ROOT / "library" / "drugs"
    ):
        self.drugs_dir = Path(drugs_dir)
        self.cache: dict[str, Data] = {}  # { 'drug_id': Data(...) }
        self._load_all_drugs()

    def _load_all_drugs(self):
        # Estructura de directorio: cada fármaco tiene un archivo .pt con su grafo
        # Nombre de los archivos: {compound_id}_{drug_name}.pt

        for drug_file in self.drugs_dir.glob("*.pt"):
            filename = drug_file.stem  # Sin extensión
            var_id = filename.split("_")[1]  # Asumimos que el ID es la primera parte

            # Cargar el objeto Data
            data_obj = torch.load(drug_file)
            self.cache[var_id] = data_obj  # Mapeo "NombreFármaco" -> Data

    def get_drug_data(self, drug_name: str) -> Data:
        """
        Recupera el objeto Data correspondiente a un ID de fármaco.
        Si no se encuentra, retorna un grafo vacío por defecto.
        """
        return self.cache.get(
            drug_name,
            Data(
                x=torch.zeros((1, 5)), edge_index=torch.empty((2, 0), dtype=torch.long)
            ),
        )


def map_variant_row_to_data(row: pd.Series, id_col: str = "variant_id") -> Data:
    """
    Mapea una fila de DataFrame a un objeto Data de PyTorch Geometric.

    Args:
        row: Fila del DataFrame que contiene la información del grafo.
        id_col: Nombre de la columna que contiene el ID del grafo.

    Returns:
        Objeto Data con atributos x, edge_index y edge_attr.
    """
    graph_dict = row.to_dict()
    graph_id = graph_dict.pop(id_col)

    # Asumimos que los datos del grafo están en formato serializado (ej. listas)
    x = torch.tensor(graph_dict.get("node_features", []), dtype=torch.float)
    edge_index = torch.tensor(graph_dict.get("edge_index", []), dtype=torch.long)
    edge_attr = torch.tensor(graph_dict.get("edge_attr", []), dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def map_drugs_to_data(
    df: pd.DataFrame, drug_col: str, drug_graphs_dir: str | Path
) -> dict[str, Data]:
    """
    Mapea IDs de fármacos a sus respectivos objetos Data cargados desde disco.

    Args:
        df: DataFrame que contiene los IDs de fármacos.
        drug_col: Nombre de la columna con los IDs de fármacos.
        drug_graphs_dir: Directorio donde se almacenan los archivos .pt de fármacos.

    Returns:
        Diccionario que mapea cada ID de fármaco a su objeto Data.
    """

    drug_data_map = {}
    drug_dir = Path(drug_graphs_dir)

    for drug_id in df[drug_col].unique():
        drug_path = drug_dir / f"{drug_id}.pt"
        try:
            drug_data = torch.load(drug_path)
            drug_data_map[drug_id] = drug_data
        except FileNotFoundError:
            # Manejo de error silencioso o ruidoso según prefieras
            drug_data_map[drug_id] = None  # O algún valor por defecto

    return drug_data_map
