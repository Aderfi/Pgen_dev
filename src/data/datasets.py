"""Pharmagen - Data Handler.

Unified Data Loading, Preprocessing, and Dataset definition.
Adheres to Zen of Python: Sparse is better than dense.
"""

import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from torch.utils.data import Dataset
from torch_geometric.data import Data

from src.config.manager import LIBRARY

logger = logging.getLogger(__name__)

UNKNOWN_CATEGORY_LABEL = "__UNKNOWN__"
EMPTY_GRAPH_NODE_DIM = 5

class PGenProcessor(BaseEstimator, TransformerMixin):
    """Handles encoding of categorical and multi-label features.

    Wraps LabelEncoder and MultiLabelBinarizer.
    """

    def __init__(
        self,
        feature_cols: list[str],
        target_cols: list[str],
        multi_label_cols: list[str],
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
        self.encoders: dict[str, Any] = {}
        self.cols_to_process = set(self.feature_cols + self.target_cols)

    def fit(
        self, df: pd.DataFrame, y: None = None, # noqa: ARG002
    ) -> "PGenProcessor":
        """Fit encoders to data.

        Args:
            df: Input DataFrame.
            y: Ignored, present for sklearn compatibility.

        Returns:
            Self for method chaining.

        """
        logger.info("Fitting encoders")
        for col in self.cols_to_process:
            if col not in df.columns:
                continue
            series = df[col]

            if col in self.multi_label_cols:
                # Optimized: Use list comprehension instead of apply
                parsed = [x.split("|") if x else [] for x in series]
                enc = MultiLabelBinarizer()
                enc.fit(parsed)
                self.encoders[col] = enc
            else:
                uniques = sorted({*series.dropna().unique(), UNKNOWN_CATEGORY_LABEL})
                enc = LabelEncoder()
                enc.fit(uniques)
                self.encoders[col] = enc

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted encoders.

        Args:
            df: Input DataFrame.

        Returns:
            Transformed DataFrame.

        Raises:
            RuntimeError: If processor is not fitted.

        """
        if not self.encoders:
            raise RuntimeError("Processor not fitted.")

        df_out = df.copy()
        for col, enc in self.encoders.items():
            if col not in df_out.columns:
                continue

            if isinstance(enc, MultiLabelBinarizer):
                # Optimized: Use list comprehension instead of apply
                parsed = [
                    x.split("|") if isinstance(x, str) and x else []
                    for x in df_out[col]
                ]
                encoded = list(enc.transform(parsed))
                df_out[col] = pd.Series(encoded, index=df_out.index)
            else:
                vals = df_out[col].astype(str).to_numpy()
                mask_unknown = ~np.isin(vals, enc.classes_)
                if mask_unknown.any():
                    vals[mask_unknown] = UNKNOWN_CATEGORY_LABEL
                df_out[col] = enc.transform(vals)

        return df_out


class PGenDataset(Dataset):
    """Optimized Dataset using contiguous memory arrays for speed.

    Separates scalar features (LongTensor) from dense/multi-hot features
    (FloatTensor).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        target_cols: list[str],
        multi_label_cols: set[str],
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
            for c in (feature_cols + target_cols)
            if c in df.columns
        ]
        multi_label_cols = {c.lower() for c in multi_label_cols}

        for col in cols:
            series = df[col]
            if col in multi_label_cols:
                matrix = np.stack(series.tolist()).astype(np.float32)
                self.dense_data[col] = np.ascontiguousarray(matrix)
            else:
                self.scalar_data[col] = series.to_numpy(dtype=np.int64)

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


EMPTY_GRAPH_NODE_DIM = 5


class DoubleTowerDataset(Dataset):
    """Dataset optimizado para modelo Double Tower.

    Carga grafos de fármacos y variantes genéticas desde disco.

    Args:
        df: DataFrame con datos de entrada.
        drug_col: Nombre de la columna con IDs de fármacos.
        haplo_col: Nombre de la columna con IDs de variantes/genes.
        target_cols: Lista de columnas objetivo.
        multilabel_cols: Lista de columnas multi-label (opcional).

    Returns:
        Diccionario con:
            - drug_data: Objeto Data del fármaco (x, edge_index).
            - haplo_data: Objeto Data de la variante/genes (x, edge_index).
            - targets: Tensor con los valores objetivo.

    """

    def __init__(
        self,
        df: pd.DataFrame,
        drug_col: str,
        haplo_col: str,
        target_cols: list[str],
        multilabel_cols: list[str] | None = None,
    ) -> None:
        """Initialize DoubleTowerDataset.

        Args:
            df: Input DataFrame.
            drug_col: Column name for drug IDs.
            haplo_col: Column name for haplotype/variant IDs.
            target_cols: List of target column names.
            multilabel_cols: List of multi-label column names (optional).

        """
        self.df = df.reset_index(drop=True)
        self.drug_col = drug_col
        self.haplo_col = haplo_col
        self.drug_lib = LIBRARY / "drugs"
        self.variant_lib = LIBRARY / "gene_graphs"

        self.drug_id_to_path = self._build_drug_index()
        self.gene_variant_path = self._build_genes_index()

        self.targets = self._encode_targets(
            df, target_cols, multilabel_cols,
        )

    def _build_drug_index(self) -> dict[str, Path]:
        """Build index mapping drug compound IDs to file paths.

        Returns:
            Dictionary mapping drug IDs to file paths.

        """
        index_drugs = {}
        for file_path in self.drug_lib.glob("*.pt"):
            match = re.match(r"^(\d+)_", file_path.name)
            if match:
                drug_id = match.group(1)
                index_drugs[drug_id] = file_path
        return index_drugs

    def _build_genes_index(self) -> dict[str, dict[str, Path]]:
        """Build index mapping gene IDs and variants to file paths.

        Returns:
            Nested dictionary mapping gene IDs and variant names to paths.

        """
        index_genes: dict[str, dict[str, Path]] = {}
        for file_path in self.variant_lib.rglob("*.pt"):
            filename_clean = file_path.stem

            gene_id, variant = filename_clean.split("_", 1)

            if gene_id not in index_genes:
                index_genes[gene_id] = {}
            index_genes[gene_id][variant] = file_path
        return index_genes

    def _get_empty_graph(self) -> Data:
        """Return empty graph with consistent dimensions.

        Returns:
            Empty PyTorch Geometric Data object.

        """
        return Data(
            x=torch.zeros((1, EMPTY_GRAPH_NODE_DIM), dtype=torch.float),
            edge_index=torch.empty((2, 0), dtype=torch.long),
        )

    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Data | dict]:
        """Get item at index.

        Args:
            idx: Index of item to retrieve.

        Returns:
            Dictionary with drug data, haplo data, and targets.

        """
        row = self.df.iloc[idx]
        drug_id_df = str(row[self.drug_col])

        drug_path: Path | str | None = self.drug_id_to_path.get(
            drug_id_df,
        )

        if drug_path and drug_path.exists():
            try:
                drug_data = torch.load(drug_path, weights_only=False)
            except (OSError, RuntimeError) as exc:
                logger.exception("Error loading drug graph: %s %s",
                    drug_path, exc_info=exc)
                drug_data = self._get_empty_graph()
        else:
            logger.warning("Drug ID %s not found on disk", drug_id_df)
            drug_data = self._get_empty_graph()

        gene_id, variant_name = str(row[self.haplo_col]).split("_", 1)

        haplo_path: Path | str | None = self.gene_variant_path.get(
            gene_id, {}).get(
                variant_name, None,
                )

        if haplo_path and haplo_path.exists():
            try:
                haplo_data = torch.load(haplo_path, weights_only=False)
            except (OSError, RuntimeError) as exc:
                logger.exception("Error loading haplo graph: %s %s",
                    haplo_path, exc_info=exc)
                haplo_data = self._get_empty_graph()
        else:
            logger.warning(
                "Haplotype ID %s not found on disk",
                row[self.haplo_col],
            )
            haplo_data = self._get_empty_graph()

        target_tensor = {col: self.targets[col][idx] for col in self.targets}

        return {
            "drug_data": drug_data,
            "haplo_data": haplo_data,
            "targets": target_tensor,
        }

    def _encode_targets(
        self,
        df: pd.DataFrame,
        target_cols: list[str],
        multilabel_cols: list[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Encode targets as tensors optimized by type.

        Args:
            df: Input DataFrame.
            target_cols: List of target column names.
            multilabel_cols: List of multi-label column names (opcional).

        Returns:
            Dictionary mapping column names to encoded tensors.

        """
        encoded_targets = {}
        self.encoders = {}

        multilabel_set = set(multilabel_cols) if multilabel_cols else set()

        for col in target_cols:
            if col in multilabel_set:
                # Optimized: Use list comprehension instead of apply
                raw_series = df[col].fillna(pd.NA).astype(str)
                raw_data = [x.split("|") if x and x != 'nan' else [] for x in raw_series]

                mlb = MultiLabelBinarizer()
                matrix = mlb.fit_transform(raw_data)

                self.encoders[col] = mlb
                encoded_targets[col] = torch.tensor(
                    matrix, dtype=torch.float32,
                )

            else:
                le = LabelEncoder()
                raw_data = df[col].fillna(pd.NA).astype(str)

                indices = le.fit_transform(raw_data)

                self.encoders[col] = le
                encoded_targets[col] = torch.tensor(
                    indices, dtype=torch.long,
                )

        return encoded_targets
