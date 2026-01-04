import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data

from src.config.manager import MULTI_LABEL_COLS

# Logger
logger = logging.getLogger(__name__)


class DoubleTowerDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        drug_col: str,
        haplo_col: str,
        target_cols: list[str],
        multilabel_cols: list[str],
        encoders: None | (
            dict
        ) = None,  # Pass pre-fitted encoders here to ensure consistency across Train/Val/Test
        drug_lib: Path = Path("./src/library/drugs"),
        variant_lib: Path = Path("./src/library/gene_graphs"),
        preload_ram: bool = False,
        input_dimensions: dict[str, int] = {},
        type_data: str | None = None,
        inference_mode: bool = False,
    ):
        """
        Args:
            encoders: Dictionary of fitted LabelEncoders/MultiLabelBinarizers.
            preload_ram: If True, loads all referenced .pt files into RAM during init.
        """
        self.df = df.reset_index(drop=True)
        self.drug_col = "drugs_cid"  # drug_col
        self.haplo_col = "genotype"  # haplo_col
        self.target_cols = target_cols
        self.multilabel_cols = set(multilabel_cols) if multilabel_cols else set()
        self.input_dims = input_dimensions
        self.inference_mode = inference_mode

        # Paths
        self.drug_lib = drug_lib
        self.variant_lib = variant_lib

        # Indexing
        self.drug_id_to_path = self._build_drug_index()
        self.gene_variant_path = self._build_genes_index()

        self.encoders = encoders if encoders is not None else {}

        # Target Pre-processing (Assuming encoded in the input DF or processing here)
        # Note: Ideally, pass the DF already encoded or handle encoding consistently externally.
        self.targets = self._encode_targets(df)

        # Optimization: In-Memory Cache
        self.preload_ram = preload_ram
        self.drug_cache = {}
        self.haplo_cache = {}

        if self.preload_ram:
            self._preload_data()

    def _preload_data(self):
        logger.info("Preloading graphs into RAM...")
        # Preload Drugs
        unique_drugs = self.df[self.drug_col].unique().astype(str)
        for drug_id in unique_drugs:
            if drug_id in self.drug_id_to_path:
                self.drug_cache[drug_id] = torch.load(
                    self.drug_id_to_path[drug_id], weights_only=False
                )

        # Preload Variants
        unique_haplos = self.df["haplo_key"].unique().astype(str)
        for haplo_str in unique_haplos:
            gene, variant = haplo_str.split("_", 1)  # Split only on first underscore
            path = self.gene_variant_path.get(gene, {}).get(variant)
            if path:
                self.haplo_cache[haplo_str] = torch.load(path, weights_only=False)
        logger.info(
            f"Loaded {len(self.drug_cache)} drugs and {len(self.haplo_cache)} variants."
        )

    def _get_empty_graph(self, type_data: str, graph_id: str = "") -> Data:
        """
        Generates a dummy graph consistent with the library_creator.py dimensions.
        Creates 1 isolated node (no edges) with zero-tensors.
        """
        # 1. Configuration Pattern
        defaults = {
            "drug": {"x": 25, "edge": 7},
            "geno": {"x": 9, "edge": 3},
            "unknown": {"x": 10, "edge": 0},
        }

        # 2. Resolve Dimensions
        # (Lógica mantenida: Prioriza self.input_dims, fallback a defaults)
        if type_data == "drug":
            n_feats = self.input_dims.get("drug_feat", defaults["drug"]["x"])
            n_edge_feats = self.input_dims.get("drug_edge", defaults["drug"]["edge"])
        elif type_data == "geno":
            n_feats = self.input_dims.get("haplo_feat", defaults["geno"]["x"])
            n_edge_feats = self.input_dims.get("haplo_edge", defaults["geno"]["edge"])
        else:
            n_feats = defaults["unknown"]["x"]
            n_edge_feats = defaults["unknown"]["edge"]

        # 3. Construct Tensors
        x = torch.zeros((1, n_feats), dtype=torch.float)
        edge_index = torch.empty((2, 0), dtype=torch.long)  # 0 edges

        data = Data(x=x, edge_index=edge_index)

        # 4. Handle Edge Attributes (CRITICAL FIX applied correctly here)
        if n_edge_feats > 0:
            data.edge_attr = torch.empty((0, n_edge_feats), dtype=torch.float)

        # 5. Metadata Assignment (ROBUSTNESS FIX)
        # Asignamos 'cid' a AMBOS casos para garantizar que el Collater siempre encuentre un ID.
        data.cid = str(graph_id)
        data.smiles = ""

        if type_data == "drug":
            data.name = "dummy_drug"
        elif type_data == "geno":
            data.name = "dummy_variant"  # Agregado para simetría con drug
            data.variant_name = str(graph_id)  # Mantenemos tu campo específico

        # 6. Sanitize
        if hasattr(self, "_sanitize_data"):
            return self._sanitize_data(data)
        return data

    def _load_graph(
        self, cache: dict, key: str, path: Path | None, type_graph: str = ""
    ) -> Data:
        # 1. Check Cache
        if key in cache:
            data = cache[key]
            return data.clone() if self.inference_mode else data

        # 2. Check Disk
        if path and path.exists():
            try:
                data = torch.load(path, weights_only=False)
                
                if self.inference_mode:
                    if not hasattr(data, "cid"):
                        data.cid = str(key)
                else:
                    if hasattr(data, "cid"): del data.cid
                    if hasattr(data, "name"): del data.name
                    if hasattr(data, "smiles"): del data.smiles
                    if hasattr(data, "variant_name"): del data.variant_name

                return self._sanitize_data(data)

            except Exception as e:
                logger.warning(f"Corrupt file {path}: {e}")
                return self._get_empty_graph(type_data=type_graph, graph_id=key)

        # 3. Return Empty
        return self._get_empty_graph(type_data=type_graph, graph_id=key)

    def _sanitize_data(self, data: Data) -> Data:
        """
        Patrón: Memory Layout Enforcement.
        Asegura que los tensores sean contiguos y 'dueños' de su memoria.
        Esto previene el error 'storage not resizable' al usar DataLoaders con workers.
        """
        if hasattr(data, "x") and data.x is not None:
            # .contiguous() fuerza una copia en memoria si el tensor no es contiguo.
            # .clone() es una alternativa más agresiva si .contiguous() no basta.
            data.x = data.x.contiguous()

        if hasattr(data, "edge_index") and data.edge_index is not None:
            data.edge_index = data.edge_index.contiguous()

        # Si tienes atributos de borde adicionales (edge_attr), haz lo mismo:
        if hasattr(data, "edge_attr") and data.edge_attr is not None:
            data.edge_attr = data.edge_attr.contiguous()

        return data

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # --- Drug Loading ---
        drug_id = str(row[self.drug_col])
        drug_path = self.drug_id_to_path.get(drug_id)
        drug_data = self._load_graph(
            self.drug_cache, drug_id, drug_path, type_graph="drug"
        )

        # --- Variant Loading ---
        # haplo_str = str(row[self.haplo_col])
        haplo_str = str(row["haplo_key"])
        gene, variant = haplo_str.split("_", 1)
        haplo_path = self.gene_variant_path.get(gene, {}).get(variant)
        haplo_data = self._load_graph(
            self.haplo_cache, haplo_str, haplo_path, type_graph="geno"
        )

        # --- Targets ---
        # Fetch pre-processed targets for this index
        target_dict = {col: self.targets[col][idx] for col in self.target_cols}

        return {
            "drug_data": drug_data,
            "haplo_data": haplo_data,
            "targets": target_dict,
        }

    def _build_drug_index(self):
        """Mapea los compound_id con sus rutas reales en disco."""
        index_drugs = {}
        # Listamos todos los archivos .pt una sola vez
        for file_path in self.drug_lib.glob("*.pt"):
            # Extraemos el ID del nombre del archivo (ej: '10007' de '10007_chlorphentermine.pt')
            # El ID es todo lo que está antes del primer guion bajo
            match = re.match(r"^(\d+)_", file_path.name)
            if match:
                drug_id = match.group(1)
                index_drugs[drug_id] = file_path
        return index_drugs

    def _build_genes_index(self):
        """Mapea los gene_id con sus rutas reales en disco."""
        # Estructura del dict: { gene_id: str, variants: [{variant_name(star5 or rs...):Path}] }

        index_genes = {}
        # Listamos todos los archivos .pt una sola vez
        for dir in self.variant_lib.rglob("**/"):
            index_genes[dir.name] = {}

        for file_path in self.variant_lib.glob("**/*.pt"):
            # gene_id es todo lo que está antes del primer guion bajo
            filename = file_path.name  # Nombre sin extensión
            filename_clean = filename.replace(".pt", "")

            gene_id, variant = filename_clean.split("_", 1)
            if variant.startswith("star"):
                variant = variant.replace("star", "*")

            if gene_id not in index_genes:
                index_genes[gene_id] = {}
            index_genes[gene_id][variant] = file_path
        return index_genes

    def _encode_targets(self, df: pd.DataFrame) -> dict[str, torch.Tensor]:
        """
        Codifica los targets generando un diccionario de tensores optimizados por tipo.
        Args:
            df: DataFrame completo.
            target_cols: Lista de columnas a usar como targets.
            multilabel_cols: Lista de columnas que contienen múltiples valores (ej: efectos adversos).
        Returns:
            Dict[str, torch.Tensor]: Diccionario {nombre_columna: Tensor}.
        """
        encoded_targets = {}

        for col in self.target_cols:
            # 1. Prepare Data
            # Convert to string and handle NaNs to avoid encoder crashes
            raw_series = df[col].fillna("Unknown").astype(str)

            if col in self.multilabel_cols:
                # --- CASE: MULTI-LABEL (e.g., "Headache|Nausea") ---
                # Split string into list of labels. Adjust separator if needed (e.g., ';', ',')
                processed_data = raw_series.apply(
                    lambda x: x.split("|") if x != "Unknown" else []
                )

                # Check if encoder exists
                if col in self.encoders:
                    # TRANSFORM MODE
                    mlb = self.encoders[col]
                    # Note: MultiLabelBinarizer ignores unknown classes during transform automatically
                    matrix = mlb.transform(processed_data)
                else:
                    # FIT MODE
                    mlb = MultiLabelBinarizer()
                    matrix = mlb.fit_transform(processed_data)
                    self.encoders[col] = mlb

                # BCEWithLogitsLoss requires FloatTensor
                encoded_targets[col] = torch.tensor(matrix, dtype=torch.float32)

            else:
                # --- CASE: SINGLE-LABEL (e.g., "Metabolizer Type A") ---
                processed_data = raw_series.values

                if col in self.encoders:
                    # TRANSFORM MODE
                    le = self.encoders[col]

                    # Handle Unseen Labels gracefully (Optional but recommended)
                    # Maps unseen labels to a specific "Unknown" class if it exists, or errors out
                    # Here we use a safe approach: map unknown to -1 or a dummy index,
                    # but since PyTorch needs valid indices, we usually assume consistency.
                    # Simple approach:
                    known_classes = set(le.classes_)
                    processed_data = [
                        x if x in known_classes else "Unknown" for x in processed_data
                    ]

                    # If "Unknown" was not in training, this will crash.
                    # Ideally, ensure your training set covers classes or handle this strictly.
                    try:
                        indices = le.transform(processed_data)
                    except ValueError:
                        # Fallback: Force fit if strictly necessary or raise clear error
                        # For now, we assume valid validation data.
                        indices = le.transform(processed_data)
                else:
                    # FIT MODE
                    le = LabelEncoder()
                    indices = le.fit_transform(processed_data)
                    self.encoders[col] = le

                # CrossEntropyLoss requires LongTensor
                encoded_targets[col] = torch.tensor(indices, dtype=torch.long)

        return encoded_targets


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
