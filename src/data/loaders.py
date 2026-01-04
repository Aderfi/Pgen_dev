import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data

from src.config.manager import MULTI_LABEL_COLS

# Logger
logger = logging.getLogger(__name__)

# Compile regex patterns once for performance
_DRUG_ID_PATTERN = re.compile(r"^(\d+)_")


class DoubleTowerDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        drug_col: str,
        haplo_col: str,
        target_cols: List[str],
        multilabel_cols: List[str],
        encoders: Optional[
            dict
        ] = None,  # Pass pre-fitted encoders here to ensure consistency across Train/Val/Test
        drug_lib: Path = Path("./src/library/drugs"),
        variant_lib: Path = Path("./src/library/gene_graphs"),
        preload_ram: bool = False,
        input_dimensions: Dict[str, int] = {},
        type_data: str | None = None,
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
        """Optimized: Batch load graphs with error handling and progress tracking."""
        logger.info("Preloading graphs into RAM...")
        
        # Preload Drugs - vectorized operations
        unique_drugs = self.df[self.drug_col].unique()
        drug_count = 0
        for drug_id in unique_drugs:
            drug_id_str = str(drug_id)
            path = self.drug_id_to_path.get(drug_id_str)
            if path and path.exists():
                try:
                    self.drug_cache[drug_id_str] = torch.load(path, weights_only=False)
                    drug_count += 1
                except Exception as e:
                    logger.warning(f"Failed to load drug {drug_id_str}: {e}")

        # Preload Variants - optimized with batch processing
        unique_haplos = self.df["haplo_key"].unique()
        haplo_count = 0
        for haplo_str in unique_haplos:
            haplo_str = str(haplo_str)
            gene, variant = haplo_str.split("_", 1)
            path = self.gene_variant_path.get(gene, {}).get(variant)
            if path and path.exists():
                try:
                    self.haplo_cache[haplo_str] = torch.load(path, weights_only=False)
                    haplo_count += 1
                except Exception as e:
                    logger.warning(f"Failed to load variant {haplo_str}: {e}")
        
        logger.info(f"Loaded {drug_count} drugs and {haplo_count} variants into cache.")

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
            return self._sanitize_data(cache[key].clone())

        # 2. Check Disk
        if path and path.exists():
            try:
                data = torch.load(path, weights_only=False)

                if not hasattr(data, "cid"):
                    data.cid = str(key)

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
        """Optimized: Build drug index with compiled regex and single pass."""
        index_drugs = {}
        # Single pass through all .pt files
        for file_path in self.drug_lib.glob("*.pt"):
            # Use pre-compiled regex pattern for better performance
            match = _DRUG_ID_PATTERN.match(file_path.name)
            if match:
                drug_id = match.group(1)
                index_drugs[drug_id] = file_path
        logger.debug(f"Indexed {len(index_drugs)} drug graphs")
        return index_drugs

    def _build_genes_index(self):
        """Optimized: Build gene index with efficient directory scanning."""
        index_genes = {}
        
        # Pre-populate gene directories
        for gene_dir in self.variant_lib.iterdir():
            if gene_dir.is_dir():
                index_genes[gene_dir.name] = {}
        
        # Single pass through all .pt files with optimized parsing
        for file_path in self.variant_lib.glob("**/*.pt"):
            filename_clean = file_path.stem  # More efficient than replace
            
            # Split only once
            parts = filename_clean.split("_", 1)
            if len(parts) == 2:
                gene_id, variant = parts
                
                # Normalize star allele notation
                if variant.startswith("star"):
                    variant = variant.replace("star", "*")
                
                if gene_id not in index_genes:
                    index_genes[gene_id] = {}
                index_genes[gene_id][variant] = file_path
        
        logger.debug(f"Indexed {len(index_genes)} genes with variants")
        return index_genes

    def _encode_targets(self, df: pd.DataFrame) -> Dict[str, torch.Tensor]:
        """
        Optimized: Encode targets with vectorized operations and minimal allocations.
        Args:
            df: DataFrame completo.
        Returns:
            Dict[str, torch.Tensor]: Diccionario {nombre_columna: Tensor}.
        """
        encoded_targets = {}

        for col in self.target_cols:
            # Vectorized NaN handling
            raw_series = df[col].fillna("Unknown").astype(str)

            if col in self.multilabel_cols:
                # Optimized: Use list comprehension instead of apply
                processed_data = [
                    x.split("|") if x != "Unknown" else [] for x in raw_series
                ]

                if col in self.encoders:
                    # TRANSFORM MODE
                    mlb = self.encoders[col]
                    matrix = mlb.transform(processed_data)
                else:
                    # FIT MODE
                    mlb = MultiLabelBinarizer()
                    matrix = mlb.fit_transform(processed_data)
                    self.encoders[col] = mlb

                # Ensure we have a dense numpy array (handles both sparse and dense)
                matrix = np.asarray(matrix)
                encoded_targets[col] = torch.from_numpy(matrix).float()

            else:
                # SINGLE-LABEL: Optimized with numpy operations
                if col in self.encoders:
                    # TRANSFORM MODE
                    le = self.encoders[col]
                    known_classes_set = set(le.classes_)
                    
                    # Vectorized unknown class handling
                    processed_data = raw_series.where(
                        raw_series.isin(known_classes_set), "Unknown"
                    ).values

                    try:
                        indices = le.transform(processed_data)
                    except ValueError as e:
                        logger.warning(f"Label encoding error for {col}: {e}")
                        indices = le.transform(processed_data)
                else:
                    # FIT MODE
                    le = LabelEncoder()
                    indices = le.fit_transform(raw_series.values)
                    self.encoders[col] = le

                # Direct tensor creation (faster than torch.tensor)
                encoded_targets[col] = torch.from_numpy(indices).long()

        return encoded_targets


class DoubleTowerCollater:
    def __init__(self):
        # Tuple is more memory efficient than list for immutable data
        self.id_priority_keys = ("cid", "variant_name", "graph_id", "name")
        self.keys_to_sanitize = (
            "cid",
            "variant_name",
            "name",
            "smiles",
            "gene_context",
            "graph_id",
        )

    def _extract_and_sanitize(self, graph_list: List[Data]) -> List[str]:
        """
        Optimized: Extract IDs and remove string attributes in a single pass.
        Modifies objects in-place for memory efficiency.
        """
        extracted_ids = []

        for data in graph_list:
            # A. Polymorphic ID extraction - early exit optimization
            found_id = "Unknown"
            for key in self.id_priority_keys:
                val = getattr(data, key, None)
                if val is not None:
                    found_id = str(val)
                    break
            extracted_ids.append(found_id)

            # B. String sanitization - batch deletion
            for key in self.keys_to_sanitize:
                try:
                    delattr(data, key)
                except AttributeError:
                    pass  # Key doesn't exist, continue

        return extracted_ids

    def __call__(self, batch_list):
        """
        Optimized: Batch processing with minimal memory allocations.
        Input: List of dicts from Dataset.__getitem__
        Output: Dict with Batched graphs and Stacked targets
        """
        # 1. Extract components - single pass
        drug_graphs = [sample["drug_data"] for sample in batch_list]
        haplo_graphs = [sample["haplo_data"] for sample in batch_list]

        # 2. Extract IDs and sanitize strings
        drug_ids = self._extract_and_sanitize(drug_graphs)
        haplo_ids = self._extract_and_sanitize(haplo_graphs)

        # 3. Safe batching (graphs now contain only numeric tensors)
        batch_drug = Batch.from_data_list(drug_graphs)
        batch_haplo = Batch.from_data_list(haplo_graphs)

        # 4. Re-inject metadata (optional but useful for debugging)
        batch_drug.meta_ids = drug_ids
        batch_haplo.meta_ids = haplo_ids

        # 5. Stack targets - optimized with direct stacking
        target_keys = batch_list[0]["targets"].keys()
        batched_targets = {
            key: torch.stack([sample["targets"][key] for sample in batch_list])
            for key in target_keys
        }

        return {
            "drug_batch": batch_drug,
            "haplo_batch": batch_haplo,
            "targets": batched_targets,
        }
