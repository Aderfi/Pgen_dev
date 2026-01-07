"""Pharmagen - Data Handler.

Unified Data Loading, Preprocessing, and Dataset definition.
"""

import gc
import logging
import re
from collections.abc import Mapping, MutableSequence, Set
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from torch.utils.data import Dataset
from torch_geometric.data.data import Data

from src.config.manager import LIBRARY, MULTI_LABEL_COLS
from src.data.graph_indexing import GraphIndexBuilder
from src.utils.exceptions import DataError, EncoderError
from src.utils.validation import GraphValidator

logger = logging.getLogger(__name__)

# Constants
UNKNOWN_CATEGORY_LABEL = "__UNKNOWN__"
EMPTY_GRAPH_NODE_DIM = 5
PRELOAD_THRESHOLD = 5000  # Max samples for RAM preloading
GC_INTERVAL = 1000  # Garbage collection interval

DEFAULT_DIMENSIONS = {
    "drugs": {"features": 25, "edges": 7, "attrs": 0},
    "geno": {"features": 9, "edges": 3, "attrs": 0},
}

class PGenProcessor(BaseEstimator, TransformerMixin):
    """Handles encoding of categorical and multi-label features.

    Wraps LabelEncoder and MultiLabelBinarizer for sklearn compatibility.

    Example:
        >>> processor = PGenProcessor(["drug"], ["outcome"], ["adverse_events"])
        >>> processor.fit(train_df)
        >>> transformed_df = processor.transform(val_df)
    """

    def __init__(
        self,
        feature_cols: MutableSequence[str],
        target_cols:  MutableSequence[str],
        multi_label_cols: MutableSequence[str],
    ) -> None:
        """Initialize PGenProcessor.

        Args:
            feature_cols: List of feature column names.
            target_cols: List of target column names.
            multi_label_cols: List of multi-label column names.
        """
        self.feature_cols = [c.lower() for c in feature_cols]
        self.target_cols = [c.lower() for c in target_cols]
        self.multi_label_cols = {c.lower() for c in multi_label_cols}
        self.encoders:  dict[str, Any] = {}
        self.cols_to_process = set(self.feature_cols + self.target_cols)

    def fit(
        self, df: pd.DataFrame, y: None = None  # noqa: ARG002
    ) -> "PGenProcessor":
        """Fit encoders to data.

        Args:
            df: Input DataFrame.
            y: Ignored, present for sklearn compatibility.

        Returns:
            Self for method chaining.
        """
        logger.info("Fitting encoders...")

        for col in self.cols_to_process:
            if col not in df.columns:
                logger.warning(f"Column '{col}' not found in DataFrame")
                continue

            series = df[col]

            if col in self.multi_label_cols:
                # Multi-label:  split by "|"
                parsed = series.apply(lambda x: x.split("|") if x else [])
                enc = MultiLabelBinarizer()
                enc.fit(parsed)
                self.encoders[col] = enc
                logger.debug(f"Fitted MultiLabelBinarizer for '{col}':  {len(enc.classes_)} classes")
            else:
                # Single-label: LabelEncoder
                uniques = sorted({*series.dropna().unique(), UNKNOWN_CATEGORY_LABEL})
                enc = LabelEncoder()
                enc.fit(uniques)
                self.encoders[col] = enc
                logger.debug(f"Fitted LabelEncoder for '{col}': {len(enc.classes_)} classes")

        logger.info(f"✅ Fitted {len(self.encoders)} encoders")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted encoders.

        Args:
            df: Input DataFrame.

        Returns:
            Transformed DataFrame.

        Raises:
            EncoderError: If processor is not fitted.
        """
        if not self.encoders:
            raise EncoderError("Processor not fitted. Call fit() first.")

        df_out = df.copy()

        for col, enc in self.encoders.items():
            if col not in df_out.columns:
                continue

            if isinstance(enc, MultiLabelBinarizer):
                # Multi-label transform
                parsed = df_out[col].apply(
                    lambda x: (x.split("|") if isinstance(x, str) and x else []),
                )
                encoded = list(enc.transform(parsed))
                df_out[col] = pd.Series(encoded, index=df_out.index)
            else:
                # Single-label transform with unknown handling
                vals = df_out[col].astype(str).to_numpy()
                mask_unknown = ~np.isin(vals, enc.classes_)

                if mask_unknown.any():
                    n_unknown = mask_unknown.sum()
                    logger.warning(f"Column '{col}':  {n_unknown} unknown values mapped to '{UNKNOWN_CATEGORY_LABEL}'")
                    vals[mask_unknown] = UNKNOWN_CATEGORY_LABEL

                df_out[col] = enc.transform(vals)

        return df_out

class PGenDataset(Dataset):
    """Optimized Dataset using contiguous memory arrays for speed.

    Separates scalar features (LongTensor) from dense/multi-hot features
    (FloatTensor) for efficient batching.

    Example:
        >>> dataset = PGenDataset(df, ["drug"], ["outcome"], {"adverse_events"})
        >>> sample = dataset[0]  # {'drug':  tensor, 'outcome': tensor, ...}
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: MutableSequence[str],
        target_cols: MutableSequence[str],
        multi_label_cols: Set[str],
    ) -> None:
        """Initialize PGenDataset.

        Args:
            df: Input DataFrame.
            feature_cols: List of feature column names.
            target_cols: List of target column names.
            multi_label_cols: Set of multi-label column names.
        """
        self.scalar_data = {}
        self.dense_data = {}
        self.length = len(df)

        cols = [
            c.lower()
            for c in (list(feature_cols) + list(target_cols))
            if c in df.columns
        ]
        multi_label_cols = {c.lower() for c in multi_label_cols}

        # Separate scalar vs dense data
        for col in cols:
            series = df[col]
            if col in multi_label_cols:
                # Multi-label: stack as float matrix
                matrix = np.stack(series.tolist()).astype(np.float32)
                self.dense_data[col] = np.ascontiguousarray(matrix)
                logger.debug(f"Stored dense data for '{col}': shape {matrix.shape}")
            else:
                # Scalar: int64 array
                self.scalar_data[col] = series.to_numpy(dtype=np.int64)
                logger.debug(f"Stored scalar data for '{col}':  {len(series)} samples")

    def __len__(self) -> int:
        """Return dataset length."""
        return self.length

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get item at index.

        Args:
            idx: Index of item to retrieve.

        Returns:
            Dictionary with column names as keys and tensors as values.
        """
        batch = {}

        for col, data in self.dense_data.items():
            batch[col] = torch.from_numpy(data[idx])

        for col, data in self.scalar_data.items():
            batch[col] = torch.tensor(data[idx], dtype=torch.long)

        return batch

class DoubleTowerDataset(Dataset):
    """Two-tower dataset for drug-genotype graph pairs.

    Handles:
    - Graph loading from disk or RAM cache
    - Target encoding (single-label and multi-label)
    - Empty graph generation for missing data
    - Memory-efficient operation

    Example:
        >>> dataset = DoubleTowerDataset(
        ...    df, "drug_id", "geno_key", ["outcome"], []
        ...)
        >>> sample = dataset[0]  # {'drug_data': Data, 'geno_data': Data, 'targets': {...}}
    """
    def __init__(
        self,
        df: pd.DataFrame,
        drug_col: str,
        geno_col: str,
        target_cols: list[str],
        multilabel_cols: list[str],
        encoders: dict | None = None,
        drug_lib:  Path = LIBRARY / "drugs",
        variant_lib: Path = LIBRARY / "gene_graphs",
        preload_ram: bool = False,
        input_dimensions: dict[str, Mapping[str, int]] | None = None,
        type_data: str | None = None,
        inference_mode: bool = False,
    ):
        """Initialize DoubleTowerDataset.

        Args:
            df: DataFrame with samples.
            drug_col: Column name for drug IDs.
            geno_col: Column name for genotype keys.
            target_cols: List of target column names.
            multilabel_cols: List of multi-label column names.
            encoders: Pre-fitted encoders (REQUIRED for val/test sets).
            drug_lib: Path to drug graph library.
            variant_lib: Path to variant graph library.
            preload_ram: If True, loads all graphs into RAM.
            input_dimensions: Dictionary with expected dimensions.
            type_data: Type identifier for data.
            inference_mode: If True, preserves metadata for inference.

        Warning:
            Setting preload_ram=True with >10k samples may cause OOM.
            Only use with <10k samples and >32GB RAM.
        """

        self.df = df.reset_index(drop=True)
        self.drug_col = drug_col
        self.geno_col = geno_col
        self.target_cols = target_cols
        self.multilabel_cols = set(multilabel_cols) if multilabel_cols else set()
        self.input_dims = input_dimensions or {}
        self.inference_mode = inference_mode

        # Validate dimensions
        if input_dimensions:
            self._validate_dimensions(input_dimensions)

        # Validate preload_ram setting
        if preload_ram and len(df) > PRELOAD_THRESHOLD:
            logger.warning(
                f"⚠️ preload_ram=True with {len(df)} samples may cause OOM."
                f"Threshold: {PRELOAD_THRESHOLD}.Consider setting preload_ram=False."
            )

        # Paths
        self.drug_lib = drug_lib
        self.variant_lib = variant_lib

        # Indexing - Using dedicated builder for SRP
        logger.debug("Building graph indices...")
        self.drug_id_to_path = GraphIndexBuilder.build_drug_index(drug_lib)
        self.gene_variant_path = GraphIndexBuilder.build_gene_variant_index(variant_lib)
        logger.info(f"📚 Indexed {len(self.drug_id_to_path)} drugs, {sum(len(v) for v in self.gene_variant_path.values())} variants")

       # Encoders
        self.encoders = encoders if encoders is not None else {}

        # Target Pre-processing
        self.targets = self._encode_targets(df)

        # Optimization: In-Memory Cache
        self.preload_ram = preload_ram
        self.drug_cache = {}
        self.geno_cache = {}
        self._cache_stats = {"drug_hits": 0, "drug_misses": 0, "geno_hits": 0, "geno_misses": 0}

        if self.preload_ram:
            self._preload_data()

    def _validate_dimensions(self, dims: Mapping[str, Mapping[str, int]]):
        """Validate input dimensions.

        Args:
            dims:  Nested dictionary with structure:
                {"drugs": {"features": int, "edges": int, "attrs": int},
                "geno": {"features": int, "edges": int, "attrs": int}}

        Raises:
            DataError: If dimensions are invalid.
        """
        required_keys = ["drugs", "geno"]
        required_subkeys = ["features", "edges", "attrs"]

        for graph_type in required_keys:
            if graph_type not in dims:
                logger.warning(f"Missing dimension key:  '{graph_type}'.\
                               Using defaults.")
                continue

            if not isinstance(dims[graph_type], dict):
                raise DataError(f"Dimension '{graph_type}' must be a dict,\
                                got {type(dims[graph_type])}")

            for subkey in required_subkeys:
                if subkey not in dims[graph_type]:
                    logger.warning(f"Missing dimension '{graph_type}.{subkey}'.\
                                   Using default.")
                    continue

                value = dims[graph_type][subkey]
                if not isinstance(value, int) or value < 0:
                    raise DataError(
                        f"Invalid dimension '{graph_type}.{subkey}':  {value} "
                        "(must be non-negative int)"
                    )

    def _preload_data(self):
        """Preload graphs into RAM for faster access.

        Warning:
            This can consume significant memory.Monitor system resources.
        """
        logger.info("📥 Preloading graphs into RAM...")

        # Preload Drugs
        unique_drugs = self.df[self.drug_col].unique().astype(str)
        logger.debug(f"Preloading {len(unique_drugs)} unique drugs...")

        for i, drug_id in enumerate(unique_drugs):
            if drug_id in self.drug_id_to_path:
                try:
                    self.drug_cache[drug_id] = torch.load(
                        self.drug_id_to_path[drug_id], weights_only=False
                    )
                except Exception as e:
                    logger.warning(f"Failed to load drug {drug_id}: {e}")

            # Periodic garbage collection
            if i > 0 and i % GC_INTERVAL == 0:
                gc.collect()

        # Preload Variants
        unique_genos = self.df["geno_key"].unique().astype(str)
        logger.debug(f"Preloading {len(unique_genos)} unique variants...")

        for i, geno_str in enumerate(unique_genos):
            try:
                gene, variant = geno_str.split("_", 1)
                path = self.gene_variant_path.get(gene, {}).get(variant)
                if path:
                    self.geno_cache[geno_str] = torch.load(path, weights_only=False)
            except Exception as e:
                logger.warning(f"Failed to load variant {geno_str}: {e}")

            # Periodic garbage collection
            if i > 0 and i % GC_INTERVAL == 0:
                gc.collect()

        # Log memory estimate
        total_graphs = len(self.drug_cache) + len(self.geno_cache)
        estimated_mb = total_graphs * 0.1
        logger.info(
            f"✅ Loaded {len(self.drug_cache)} drugs and {len(self.geno_cache)} variants."
            f"Estimated memory:  ~{estimated_mb:.1f}MB"
        )

    def _get_empty_graph(self, type_data: str, graph_id: str = "") -> Data:
            """Generate dummy graph consistent with library dimensions.

            Creates 1 isolated node (no edges) with zero-tensors.

            Args:
                type_data: Type of graph ("drug" or "geno").
                graph_id: Optional ID for metadata.

            Returns:
                Empty PyG Data object.

            Raises:
                ValueError: If type_data is invalid.
            """
            # Resolve dimensions (priority: input_dims > defaults)
            if type_data == "drug":
                drug_dims = self.input_dims. get("drugs", DEFAULT_DIMENSIONS["drugs"])
                n_feats = drug_dims.get("features", DEFAULT_DIMENSIONS["drugs"]["features"])
                n_edge_feats = drug_dims. get("edges", DEFAULT_DIMENSIONS["drugs"]["edges"])
            elif type_data == "geno":
                geno_dims = self.input_dims.get("geno", DEFAULT_DIMENSIONS["geno"])
                n_feats = geno_dims.get("features", DEFAULT_DIMENSIONS["geno"]["features"])
                n_edge_feats = geno_dims.get("edges", DEFAULT_DIMENSIONS["geno"]["edges"]) # noqa
            else:
                raise ValueError(f"Unknown type_data: '{type_data}'.Must be 'drug' or 'geno'.")

            # Construct empty graph
            x = torch.zeros((1, n_feats), dtype=torch.float)
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attrs = torch.empty((0, n_edge_feats), dtype=torch.float)

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attrs)

            # Metadata (consistent for both types)
            data.cid = str(graph_id)
            data.smiles = ""

            if type_data == "drug":
                data.name = "dummy_drug"
            elif type_data == "geno":
                data.name = "dummy_variant"
                data.variant_name = str(graph_id)

            return self._sanitize_data(data)

    def _load_graph(
        self, cache: dict, key: str, path: Path | None, type_graph: str = ""
    ) -> Data:
        """Load graph from cache or disk.

        Args:
            cache: Cache dictionary (drug_cache or geno_cache).
            key: Graph identifier.
            path: Path to graph file.
            type_graph: Type of graph ("drug" or "geno").

        Returns:
            PyG Data object.
        """
        # 1.Check Cache
        if key in cache:
            self._cache_stats[f"{type_graph}_hits"] += 1
            data = cache[key]
            return data.clone() if self.inference_mode else data

        # 2.Check Disk
        self._cache_stats[f"{type_graph}_misses"] += 1

        if path and path.exists():
            try:
                data = torch.load(path, weights_only=False)

                # Handle metadata based on mode
                if self.inference_mode:
                    if not hasattr(data, "cid"):
                        data.cid = str(key)
                else:
                    # Remove metadata in training mode
                    for attr in ["cid", "name", "smiles", "variant_name"]:
                        if hasattr(data, attr):
                            delattr(data, attr)

                return self._sanitize_data(data)

            except Exception as e:
                logger.warning(f"Corrupt file {path}: {e}")
                return self._get_empty_graph(type_data=type_graph, graph_id=key)

        # 3.TRY TO CREATE IT ON THE FLY (OPTIONAL) - NOT IMPLEMENTED
        # (Placeholder for future logic)

        # 4.Return Empty
        return self._get_empty_graph(type_data=type_graph, graph_id=key)

    def _sanitize_data(self, data: Data) -> Data:
        """Ensure tensors are contiguous for DataLoader compatibility.
        Prevents 'storage not resizable' errors with multiprocessing.

        Args:
            data: PyG Data object.

        Returns:
            Sanitized Data object.
        """
        if hasattr(data, "x") and data.x is not None:
            data.x = data.x.contiguous()
        if hasattr(data, "edge_index") and data.edge_index is not None:
            data.edge_index = data.edge_index.contiguous()
        if hasattr(data, "edge_attr") and data.edge_attr is not None:
            data.edge_attr = data.edge_attr.contiguous()

        return data

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        """Get sample at index.

        Args:
            idx: Sample index.

        Returns:
            Dictionary with 'drug_data', 'geno_data', and 'targets'.
        """
        row = self.df.iloc[idx]

        # Load Drug Graph
        drug_id = str(row[self.drug_col])
        drug_path = self.drug_id_to_path.get(drug_id)
        drug_data = self._load_graph(
            self.drug_cache, drug_id, drug_path, type_graph="drug"
        )

        # Load Variant Graph
        geno_str = str(row["geno_key"])
        gene, variant = geno_str.split("_", 1)
        geno_path = self.gene_variant_path.get(gene, {}).get(variant)
        geno_data = self._load_graph(
            self.geno_cache, geno_str, geno_path, type_graph="geno"
        )

        # Optional:  Validate graphs
        try:
            GraphValidator.validate_graph_data(drug_data, "drug")
            GraphValidator.validate_graph_data(geno_data, "geno")
        except ValueError as e:
            logger.error(f"Graph validation failed at idx {idx}: {e}")
        # Targets
        target_dict = {col: self.targets[col][idx] for col in self.target_cols}

        return {
            "drug_data": drug_data,
            "geno_data": geno_data,
            "targets": target_dict,
        }

    def get_cache_stats(self) -> dict:
        """Return cache hit/miss statistics.

        Returns:
            Dictionary with cache statistics.
        """
        total_drug = self._cache_stats["drug_hits"] + self._cache_stats["drug_misses"]
        total_geno = self._cache_stats["geno_hits"] + self._cache_stats["geno_misses"]

        return {
            "drug_hit_rate": self._cache_stats["drug_hits"] / total_drug if total_drug > 0 else 0.0,
            "geno_hit_rate": self._cache_stats["geno_hits"] / total_geno if total_geno > 0 else 0.0,
            **self._cache_stats,
        }

    def _encode_targets(self, df: pd.DataFrame) -> dict[str, torch.Tensor]:
        """Encode targets into tensors.
        Handles both single-label and multi-label targets.

        Args:
            df: Input DataFrame.

        Returns:
            Dictionary mapping target names to tensors.

        Raises:
            EncoderError: If encoding fails.
        """
        encoded_targets = {}

        for col in self.target_cols:
            try:
                # Prepare data
                raw_series = df[col].fillna("Unknown").astype(str)

                if col in self.multilabel_cols:
                    # Multi-label encoding
                    encoded_targets[col] = self._encode_multilabel(col, raw_series)
                else:
                    # Single-label encoding
                    encoded_targets[col] = self._encode_singlelabel(col, raw_series)

            except Exception as e:
                raise EncoderError(f"Failed to encode target '{col}': {e}") from e

        logger.debug(f"Encoded {len(encoded_targets)} targets")
        return encoded_targets

    def _encode_multilabel(self, col: str, series: pd.Series) -> torch.Tensor:
        """Encode multi-label target.

        Args:
            col: Column name.
            series: Data series.

        Returns:
            Float tensor (for BCEWithLogitsLoss).
        """

        processed_data = series.apply(
            lambda x: x.split("|") if x != "Unknown" else []
        )

        if col in self.encoders:
            # Transform mode
            mlb = self.encoders[col]
            matrix = mlb.transform(processed_data)
        else:
            # Fit mode
            mlb = MultiLabelBinarizer()
            matrix = mlb.fit_transform(processed_data)
            self.encoders[col] = mlb
            logger.debug(f"Fitted MultiLabelBinarizer for '{col}':  {len(mlb.classes_)} classes")

        return torch.tensor(matrix, dtype=torch.float32)

    def _encode_singlelabel(self, col: str, series: pd.Series) -> torch.Tensor:
        """Encode single-label target.

        Args:
            col: Column name.
            series: Data series.

        Returns:
            Long tensor (for CrossEntropyLoss).
        """
        processed_data = series.values

        if col in self.encoders:
            # Transform mode with unknown handling
            le = self.encoders[col]
            known_classes = set(le.classes_)
            processed_data = [
                x if x in known_classes else "Unknown" for x in processed_data
            ]

            try:
                indices = le.transform(processed_data)
            except ValueError as e:
                raise EncoderError(
                    f"Failed to transform '{col}'. Ensure 'Unknown' class exists in training data."
                ) from e
        else:
            # Fit mode
            le = LabelEncoder()
            indices = le.fit_transform(processed_data) # type: ignore
            self.encoders[col] = le
            logger.debug(f"Fitted LabelEncoder for '{col}': {len(le.classes_)} classes")

        return torch.tensor(indices, dtype=torch.long)
