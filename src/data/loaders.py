import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import re
from typing import Dict, List, Union, Optional
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

# Logger
logger = logging.getLogger(__name__)

class DoubleTowerDataset(Dataset):
    def __init__(
        self, 
        df: pd.DataFrame, 
        drug_col: str,
        haplo_col: str,
        target_cols: List[str],
        multilabel_cols: List[str],
        encoders: Optional[dict] = None, # Pass pre-fitted encoders here to ensure consistency across Train/Val/Test
        drug_lib: Path = Path("./src/library/drugs"),
        variant_lib: Path = Path("./src/library/gene_graphs"),
        preload_ram: bool = False
    ):
        """
        Args:
            encoders: Dictionary of fitted LabelEncoders/MultiLabelBinarizers.
            preload_ram: If True, loads all referenced .pt files into RAM during init.
        """
        self.df = df.reset_index(drop=True)
        self.drug_col = drug_col
        self.haplo_col = haplo_col
        self.target_cols = target_cols
        self.multilabel_cols = set(multilabel_cols) if multilabel_cols else set()        
        
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
                self.drug_cache[drug_id] = torch.load(self.drug_id_to_path[drug_id], weights_only=False)
        
        # Preload Variants
        unique_haplos = self.df[self.haplo_col].unique().astype(str)
        for haplo_str in unique_haplos:
            gene, variant = haplo_str.split("_", 1) # Split only on first underscore
            path = self.gene_variant_path.get(gene, {}).get(variant)
            if path:
                self.haplo_cache[haplo_str] = torch.load(path, weights_only=False)
        logger.info(f"Loaded {len(self.drug_cache)} drugs and {len(self.haplo_cache)} variants.")

    def _get_empty_graph(self):
        # Create a safe empty graph with necessary attributes for GATv2
        # GATv2 requires x (features) and edge_index
        return Data(x=torch.zeros((1, 1), dtype=torch.float), edge_index=torch.empty((2, 0), dtype=torch.long))

    def _load_graph(self, cache: dict, key: str, path: Path | None):
        # 1. Check Cache
        if key in cache:
            return cache[key]
        
        # 2. Check Disk
        if path and path.exists():
            try:
                data = torch.load(path, weights_only=False)
                # Optional: Add to cache if using dynamic caching (not implemented here for simplicity)
                return data
            except Exception as e:
                logger.warning(f"Corrupt file {path}: {e}")
                return self._get_empty_graph()
        
        # 3. Return Empty
        return self._get_empty_graph()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # --- Drug Loading ---
        drug_id = str(row[self.drug_col])
        drug_path = self.drug_id_to_path.get(drug_id)
        drug_data = self._load_graph(self.drug_cache, drug_id, drug_path)

        # --- Variant Loading ---
        haplo_str = str(row[self.haplo_col])
        gene, variant = haplo_str.split("_", 1)
        haplo_path = self.gene_variant_path.get(gene, {}).get(variant)
        haplo_data = self._load_graph(self.haplo_cache, haplo_str, haplo_path)

        # --- Targets ---
        # Fetch pre-processed targets for this index
        target_dict = {col: self.targets[col][idx] for col in self.target_cols}

        return {
            "drug_data": drug_data,
            "haplo_data": haplo_data,
            "targets": target_dict
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
            filename = file_path.name # Nombre sin extensión
            filename_clean = filename.replace(".pt", "")
            
            gene_id, variant = filename_clean.split("_", 1)
            if variant.startswith("star"):
                variant = variant.replace("star", "*")

            if gene_id not in index_genes:
                index_genes[gene_id] = {}
            index_genes[gene_id][variant] = file_path
        return index_genes

    def _encode_targets(self, df: pd.DataFrame) -> Dict[str, torch.Tensor]:
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
                processed_data = raw_series.apply(lambda x: x.split('|') if x != "Unknown" else [])
                
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
                    processed_data = [x if x in known_classes else "Unknown" for x in processed_data]
                    
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
    

class DataLoaderUtils:
    @staticmethod
    def load_dataframe(csv_path: Union[str, Path]) -> pd.DataFrame:
        """Carga el DataFrame desde CSV."""
        if str(csv_path).endswith('.csv'):
            df = pd.read_csv(csv_path)
        else:
            df = pd.read_csv(csv_path, sep='\t')   
        return df
    
    @staticmethod
    def _build_drug_index(drug_lib: Path) -> Dict[str, Path]:
        """Mapea los compound_id con sus rutas reales en disco."""
        index_drugs = {}
        # Listamos todos los archivos .pt una sola vez
        for file_path in drug_lib.glob("*.pt"):
            # Extraemos el ID del nombre del archivo (ej: '10007' de '10007_chlorphentermine.pt')
            # El ID es todo lo que está antes del primer guion bajo
            match = re.match(r"^(\d+)_", file_path.name)
            if match:
                drug_id = match.group(1)
                index_drugs[drug_id] = file_path
        return index_drugs
    
    @staticmethod
    def _build_genes_index(variant_lib: Path) -> Dict[str, Dict[str, Path]]:
        """Mapea los gene_id con sus rutas reales en disco."""
        # Estructura del dict: { gene_id: str, variants: [{variant_name(star5 or rs...):Path}] }

        index_genes = {}
        # Listamos todos los archivos .pt una sola vez
        for file_path in variant_lib.glob("*.pt"):
            # gene_id es todo lo que está antes del primer guion bajo
            filename_clean = file_path.stem  # Nombre sin extensión
            
            gene_id, variant = filename_clean.split("_", 1)

            if gene_id not in index_genes:
                index_genes[gene_id] = {}
            index_genes[gene_id][variant] = file_path
        return index_genes
    

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
            batched_targets[key] = torch.stack([sample['targets'][key] for sample in batch_list])

        return {
            "drug_batch": batch_drug,
            "haplo_batch": batch_haplo,
            "targets": batched_targets
        }