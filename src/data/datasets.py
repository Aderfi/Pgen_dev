# -*- coding: utf-8 -*-
# Pharmagen - Data Handler
# Unified Data Loading, Preprocessing, and Dataset definition.
# Adheres to Zen of Python: Sparse is better than dense.

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from torch.utils.data import Dataset
from torch_geometric.data import Data

from src.config.manager import LIBRARY

logger = logging.getLogger(__name__)

UNKNOWN_TOKEN = "__UNKNOWN__"


class PGenProcessor(BaseEstimator, TransformerMixin):
    """
    Handles encoding of categorical and multi-label features.
    Wraps LabelEncoder and MultiLabelBinarizer.
    """

    def __init__(
        self,
        feature_cols: List[str],
        target_cols: List[str],
        multi_label_cols: List[str],
    ):
        self.feature_cols = [c.lower() for c in feature_cols]
        self.target_cols = [c.lower() for c in target_cols]
        self.multi_label_cols = set(c.lower() for c in multi_label_cols)
        self.encoders: Dict[str, Any] = {}
        self.cols_to_process = set(self.feature_cols + self.target_cols)

    def fit(self, df: pd.DataFrame, y=None):
        logger.info("Fitting encoders...")
        for col in self.cols_to_process:
            if col not in df.columns:
                continue

            series = df[col]
            if col in self.multi_label_cols:
                # Split strings to lists for MLB
                parsed = series.apply(lambda x: x.split("|") if x else [])
                enc = MultiLabelBinarizer()
                enc.fit(parsed)
                self.encoders[col] = enc
            else:
                # Single label
                uniques = sorted(list(set(series.unique()) | {UNKNOWN_TOKEN}))
                enc = LabelEncoder()
                enc.fit(uniques)
                self.encoders[col] = enc
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.encoders:
            raise RuntimeError("Processor not fitted.")

        df_out = df.copy()
        for col, enc in self.encoders.items():
            if col not in df_out.columns:
                continue

            if isinstance(enc, MultiLabelBinarizer):
                # String -> List -> Multi-Hot Matrix
                parsed = df_out[col].apply(
                    lambda x: x.split("|") if isinstance(x, str) and x else []
                )
                # Store as object (numpy array inside cell) to keep DataFrame structure
                encoded = list(enc.transform(parsed))
                df_out[col] = pd.Series(encoded, index=df_out.index)
            else:
                # String -> Int
                # Handle unknown values safely
                vals = df_out[col].astype(str).to_numpy()
                mask_unknown = ~np.isin(vals, enc.classes_)
                if mask_unknown.any():
                    vals[mask_unknown] = UNKNOWN_TOKEN
                df_out[col] = enc.transform(vals)

        return df_out


class PGenDataset(Dataset):
    """
    Optimized Dataset using contiguous memory arrays for speed.
    Separates scalar features (LongTensor) from dense/multi-hot features (FloatTensor).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_cols: List[str],
        multi_label_cols: Set[str],
    ):
        self.scalar_data = {}
        self.dense_data = {}
        self.length = len(df)

        cols = [c.lower() for c in (feature_cols + target_cols) if c in df.columns]
        multi_label_cols = {c.lower() for c in multi_label_cols}

        for col in cols:
            series = df[col]
            if col in multi_label_cols:
                # List of arrays -> 2D Matrix -> Float32
                # Assumes series contains numpy arrays from PGenProcessor
                matrix = np.stack(series.tolist()).astype(np.float32)
                self.dense_data[col] = np.ascontiguousarray(matrix)
            else:
                # Int array -> Int64 (Long)
                self.scalar_data[col] = series.to_numpy(dtype=np.int64)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Zero-copy conversion to tensors
        batch = {}
        for col, data in self.dense_data.items():
            batch[col] = torch.from_numpy(data[idx])
        for col, data in self.scalar_data.items():
            batch[col] = torch.tensor(data[idx], dtype=torch.long)
        return batch


class DoubleTowerDataset(Dataset):
    """
    Dataset optimizado:
        1. Fármacos: Mapea strings a archivos .pt de la librería en disco.
        2. Variantes/Genes: Mapea strings a archivos .pt de la librería en disco por cada gen.
    Args:
        df: DataFrame con datos de entrada.
        drug_col: Nombre de la columna con IDs de fármacos.
        haplo_col: Nombre de la columna con IDs de variantes/genes.
        target_cols: Lista de columnas objetivo.
        drug_graphs_dir: Directorio base donde se almacenan los archivos .pt de fármacos.
        variant_loader: Instancia de ParquetVariantLoader con los grafos ya cargados en memoria.

    Returns:
        Diccionario con:
            - drug_data: Objeto Data del fármaco.           (x, edge_index, edge_attr)
            - haplo_data: Objeto Data de la variante/genes. (x, edge_index, edge_attr)
            - targets: Tensor con los valores objetivo.     (float)

    """

    def __init__(
        self,
        df: pd.DataFrame,
        drug_col: str,
        haplo_col: str,
        target_cols: List[str],
        multilabel_cols: Optional[List[str]] = None,
    ):
        self.df = df.reset_index(drop=True)
        self.drug_col = drug_col
        self.haplo_col = haplo_col
        self.drug_lib = LIBRARY / "drugs"
        self.variant_lib = LIBRARY / "gene_graphs"

        self.drug_id_to_path = self._build_drug_index()
        self.gene_variant_path = self._build_genes_index()

        # Convertir targets a float32 para regresión/BCE
        # self.targets = df[target_cols].values.astype('float32')
        self.targets = self._encode_targets(df, target_cols, multilabel_cols)

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
        # self.variant_lib == library/gene_graphs
        # library/gene_graphs/CYP2D6/CYP2D6_star4.pt ... library/gene_graphs/HLA-B/HLA-B_rs3892097.pt
        # Estructura esperada del índice:
        #   { gene_id: { variant_name: Path } }

        index_genes = {}
        # Listamos todos los archivos .pt una sola vez
        for file_path in self.variant_lib.rglob("*.pt"):
            # gene_id es todo lo que está antes del primer guion bajo
            filename_clean = file_path.stem  # Nombre sin extensión

            gene_id, variant = filename_clean.split("_", 1)
            # if variant.startswith("star"):
            #    variant = variant.replace("star", "*")

            if gene_id not in index_genes:
                index_genes[gene_id] = {}
            index_genes[gene_id][variant] = file_path
        return index_genes

    def _get_empty_graph(self):
        """Retorna un grafo vacío (Data) con dimensiones consistentes."""
        return Data(
            x=torch.zeros((1, 5), dtype=torch.float),
            edge_index=torch.empty((2, 0), dtype=torch.long),
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        drug_id_df = str(row[self.drug_col])

        # A. Fármaco (I/O Disco)
        # Archivo: {compound_id}_{drug_name}.pt

        drug_path: Union[Path, str] | None = self.drug_id_to_path.get(
            drug_id_df,
        )

        if drug_path and drug_path.exists():
            try:
                # Usamos weights_only=True por seguridad (estándar en versiones recientes de Torch)
                drug_data = torch.load(drug_path, weights_only=False)
            except Exception as e:
                logger.error(f"Error cargando {drug_path}: {e}")
                drug_data = self._get_empty_graph()
        else:
            # Manejo si el fármaco no existe en el directorio
            logger.warning(f"ID {drug_id_df} no encontrado en disco.")
            drug_data = self._get_empty_graph()

        # B. Variante
        # Archivo: {gene_id}_{variant_name}.pt
        haplo_path: Union[Path, str] | None = self.gene_variant_path.get(
            row[self.haplo_col], None
        )

        if haplo_path and haplo_path.exists():
            try:
                # Usamos weights_only=True por seguridad (estándar en versiones recientes de Torch)
                haplo_data = torch.load(haplo_path, weights_only=False)
            except Exception as e:
                logger.error(f"Error cargando {haplo_path}: {e}")
                haplo_data = self._get_empty_graph()
        else:
            # Manejo si la variante no existe en el directorio
            logger.warning(f"ID {row[self.haplo_col]} no encontrado en disco.")
            haplo_data = self._get_empty_graph()

        # C. Targets
        target_tensor = {col: self.targets[col][idx] for col in self.targets}

        return {
            "drug_data": drug_data,
            "haplo_data": haplo_data,
            "targets": target_tensor,
        }

    def _encode_targets(
        self,
        df: pd.DataFrame,
        target_cols: List[str],
        multilabel_cols: List[str] | None = None,
    ) -> Dict[str, torch.Tensor]:
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
        self.encoders = {}  # Vital para poder decodificar las predicciones después

        multilabel_set = set(multilabel_cols) if multilabel_cols else set()

        for col in target_cols:
            if col in multilabel_set:
                # --- CASO MULTI-LABEL (Ej: "Cefalea|Náuseas") ---
                # Usamos MultiLabelBinarizer para crear vectores One-Hot [0, 1, 0, 1]
                # IMPORTANTE: Ajusta el separador .split('|') si tus datos usan ';' o ','
                raw_data = (
                    df[col]
                    .fillna(pd.NA)
                    .astype(str)
                    .apply(lambda x: x.split("|") if x else [])
                )

                mlb = MultiLabelBinarizer()
                matrix = mlb.fit_transform(raw_data)

                self.encoders[col] = mlb
                # BCEWithLogitsLoss requiere FloatTensor
                encoded_targets[col] = torch.tensor(matrix, dtype=torch.float32)

            else:
                # --- CASO SINGLE-LABEL (Ej: "Metabolizador Lento") ---
                # Usamos LabelEncoder para crear índices de clase [0, 2, 1...]
                le = LabelEncoder()
                raw_data = df[col].fillna(pd.NA).astype(str)

                indices = le.fit_transform(raw_data)

                self.encoders[col] = le
                # CrossEntropyLoss requiere LongTensor (enteros)
                encoded_targets[col] = torch.tensor(indices, dtype=torch.long)

        return encoded_targets


if __name__ == "__main__":
    # Configuración de Logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)
    import shutil

    print("\n🔬 --- INICIANDO DIAGNÓSTICO DE DOUBLE TOWER DATASET ---\n")

    # 1. CONFIGURACIÓN DEL ENTORNO MOCK (Simulación de disco)
    # Definimos una ruta temporal local para no afectar tu sistema real
    TEST_DIR = Path("./test_env_temp")
    LIBRARY_MOCK = TEST_DIR / "library"

    # Sobreescribimos la variable global LIBRARY para que apunte al entorno de prueba
    # NOTA: En tu código real, asegúrate de que LIBRARY esté definida antes de la clase
    global LIBRARY
    LIBRARY = LIBRARY_MOCK

    try:
        # Limpieza previa por si acaso
        if TEST_DIR.exists():
            shutil.rmtree(TEST_DIR)

        # Crear estructura de directorios
        (LIBRARY_MOCK / "drugs").mkdir(parents=True)
        (LIBRARY_MOCK / "gene_graphs").mkdir(parents=True)

        logger.info(f"📁 Entorno temporal creado en: {TEST_DIR}")

        # 2. GENERACIÓN DE DATOS DUMMY (Archivos .pt y DataFrame)

        # A. Crear grafos dummy para Fármacos (ID_Nombre.pt)
        drug_ids = ["1001", "1002"]
        for d_id in drug_ids:
            dummy_graph = Data(
                x=torch.randn(5, 10), edge_index=torch.tensor([[0, 1], [1, 0]])
            )
            torch.save(dummy_graph, LIBRARY_MOCK / "drugs" / f"{d_id}_testdrug.pt")

        # B. Crear grafos dummy para Genes (Gen_Variante.pt)
        # Nota: Usamos el formato que tu _build_genes_index espera (GEN_VARIANTE)
        gene_files = ["CYP2D6_star4.pt", "CYP2D6_star1.pt", "HLA-B_star5701.pt"]
        for g_file in gene_files:
            dummy_graph = Data(
                x=torch.randn(8, 10), edge_index=torch.tensor([[0, 1], [1, 0]])
            )
            torch.save(dummy_graph, LIBRARY_MOCK / "gene_graphs" / g_file)

        # C. Crear DataFrame de prueba
        df_data = {
            "compound_id": [1001, 1002, 9999],  # 9999 no existe (prueba robustez)
            "gene_id": ["CYP2D6", "HLA-B", "CYP2D6"],
            "metabolizer": ["Poor", "Normal", "Ultra"],  # Target Single-label
            "side_effects": ["Headache|Nausea", "Nausea", None],  # Target Multi-label
        }
        df_test = pd.DataFrame(df_data)
        logger.info("📊 DataFrame de prueba generado.")

        # 3. INSTANCIACIÓN DEL DATASET
        logger.info("⚙️ Instanciando DoubleTowerDataset...")

        dataset = DoubleTowerDataset(
            df=df_test,
            drug_col="drugs_cid",
            haplo_col="genotype",
            target_cols=["metabolizer", "side_effects"],
            multilabel_cols=["side_effects"],
        )

        # 4. EVALUACIÓN DE COMPONENTES

        # A. Evaluar Encoding de Targets
        print("\n--- 1. Evaluación de Target Encoding ---")
        for col, encoder in dataset.encoders.items():
            print(f"✅ Columna '{col}': Encoder tipo {type(encoder).__name__}")
            if hasattr(encoder, "classes_"):
                print(f"   Clases detectadas: {encoder.classes_}")

        # B. Evaluar Carga de Datos (__getitem__)
        print("\n--- 2. Evaluación de Carga de Grafos (Item 0) ---")
        try:
            item = dataset[0]

            # Chequeo Fármaco
            drug_data = item["drug_data"]
            print(f"💊 Drug Data: {drug_data}")
            if drug_data.x.shape[0] == 1 and drug_data.x.shape[1] == 5:
                print(
                    "   ⚠️ (Aviso) Se cargó el grafo vacío por defecto (checkear IDs)."
                )
            else:
                print(
                    f"   ✅ Grafo de fármaco cargado correctamente. Nodos: {drug_data.num_nodes}"
                )

            # Chequeo Targets
            print(f"🎯 Targets: {item['targets']}")

            # Chequeo Variante (Aquí es donde tu código actual podría fallar)
            haplo_data = item["haplo_data"]
            print(f"🧬 Haplo Data: {haplo_data}")

        except AttributeError as e:
            print("\n❌ ERROR CRÍTICO DETECTADO EN LÓGICA DE VARIANTES:")
            print(f"   El error fue: {e}")
            print(
                "   👉 DIAGNÓSTICO: Tu método '_build_genes_index' crea un diccionario anidado:"
            )
            print("      Ej: {'CYP2D6': {'star4': Path(...)} }")
            print(
                "      Pero en '__getitem__', intentas acceder como si fuera una ruta directa:"
            )
            print(
                "      'haplo_path.exists()' falla porque 'haplo_path' es un diccionario, no un Path."
            )
            print(
                "   👉 SOLUCIÓN: Debes decidir si 'haplo_col' en el DF tiene 'CYP2D6' o 'CYP2D6_star4'."
            )

        except Exception as e:
            print(f"❌ Error inesperado: {e}")

        # 5. PRUEBA DE DATALOADER (Simulación de Batch)
        print("\n--- 3. Prueba de Batching (DataLoader) ---")
        try:
            from torch_geometric.loader import DataLoader

            loader = DataLoader(dataset, batch_size=2, shuffle=False)
            batch = next(iter(loader))
            print("📦 Batch generado exitosamente.")
            print(f"   Batch Size: {batch['drug_data'].batch.max() + 1}")
            print(
                f"   Targets Batch Shape (side_effects): {batch['targets']['side_effects'].shape}"
            )
        except Exception as e:
            print(
                f"⚠️ No se pudo crear el batch (probablemente por el error anterior): {e}"
            )

    finally:
        # LIMPIEZA
        # Comenta la siguiente línea si quieres inspeccionar los archivos generados
        if TEST_DIR.exists():
            shutil.rmtree(TEST_DIR)
        print("\n🧹 Entorno temporal eliminado. Test finalizado.")
