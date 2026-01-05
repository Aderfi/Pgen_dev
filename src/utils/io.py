# Pharmagen - IO Utilities
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.config.manager import DIRS, METADATA, MULTI_LABEL_COLS, VERSION

LOGS_DIR = DIRS["logs"]
UNKNOWN_TOKEN = "__UNKNOWN__"

logger = logging.getLogger(__name__)

##############################################################################
# JSON HANDLING
##############################################################################


def save_json(data: dict[str, Any], path: str | Path):
    """Saves a dictionary to a JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(path: str | Path) -> dict[str, Any]:
    """Loads a JSON file into a dictionary."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


##############################################################################
# CONSOLE MESSAGES
##############################################################################


def welcome_message():
    msg = f"""
    ============================================
            PHARMAGEN v{VERSION}
    ============================================
    Software para farmacogenética y deep learning.

    Logs: {LOGS_DIR}
    ============================================
    """
    print(msg)


def print_gnu_notice():
    """Imprime el aviso legal"""

    # Lógica inteligente de años
    start_year = 2025
    current_year = datetime.now().year

    if current_year > start_year:
        # Si estamos en 2026 o futuro, muestra "2025-2026"
        year_str = f"{start_year}-{current_year}"
    else:
        # Si estamos en 2025, muestra solo "2025"
        year_str = str(start_year)

    author = "Adrim Hamed Outmani (@Aderfi)"
    program = METADATA.get("project_name", "Pharmagen")

    notice = f"""
    {program} Copyright (C) {year_str} {author}
    This program comes with ABSOLUTELY NO WARRANTY; for details type `show w'.
    This is free software, and you are welcome to redistribute it
    under certain conditions; type `show c' for details.
    """
    print(notice)


def print_warranty_details():
    """Texto completo para 'show w'."""
    print("\n" + "=" * 60)
    print("NO WARRANTY")
    print("=" * 60)
    print("""
    BECAUSE THE PROGRAM IS LICENSED FREE OF CHARGE, THERE IS NO WARRANTY
    FOR THE PROGRAM, TO THE EXTENT PERMITTED BY APPLICABLE LAW. EXCEPT WHEN
    OTHERWISE STATED IN WRITING THE COPYRIGHT HOLDERS AND/OR OTHER PARTIES
    PROVIDE THE PROGRAM "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER EXPRESSED
    OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
    MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE ENTIRE RISK AS
    TO THE QUALITY AND PERFORMANCE OF THE PROGRAM IS WITH YOU. SHOULD THE
    PROGRAM PROVE DEFECTIVE, YOU ASSUME THE COST OF ALL NECESSARY SERVICING,
    REPAIR OR CORRECTION.
    """)
    input("\nPresione [Enter] para volver...")


def print_conditions_details():
    """Texto completo para 'show c'."""
    print("\n" + "=" * 60)
    print("REDISTRIBUTION CONDITIONS")
    print("=" * 60)
    print("""
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <https://www.gnu.org/licenses/>.
    """)
    input("\nPresione [Enter] para volver...")


##############################################################################
# DATA LOADING UTILITIES
##############################################################################

class DataLoaderUtils:
    @staticmethod
    def load_dataframe(
        csv_path: str | Path,
        cols: list,
        stratify_col: list[str] | str | None = None,
    ) -> pd.DataFrame:
        """Carga el DataFrame desde CSV."""
        if str(csv_path).endswith(".csv"):
            df = pd.read_csv(csv_path)
        else:
            df = pd.read_csv(
                csv_path,
                sep="\t",
            )
        return DataLoaderUtils.clean_and_prepare_data(df, stratify_col=stratify_col)

    @staticmethod
    def normalize_multilabel_col(series: pd.Series, delimiter: str = "|") -> pd.Series:
        """
        Patrón: String Normalization.
        Asegura que las etiquetas multi-label sean consistentes, únicas y ordenadas.
        Optimized: Uses vectorized string operations for better performance.
        """
        # Convert to string and handle missing/empty values
        series_str = series.fillna("").astype(str)
        
        # Vectorized approach: split, clean, sort, and rejoin
        # This is faster than apply for large series
        def _clean_string(x):
            if not x or x.strip() == "" or x.lower() == "unknown":
                return ""
            parts = x.split(delimiter)
            cleaned_parts = sorted(list({p.strip() for p in parts if p.strip()}))
            return delimiter.join(cleaned_parts)
        
        # Use list comprehension which is faster than apply for this use case
        return pd.Series([_clean_string(x) for x in series_str], index=series.index)

    @staticmethod
    def add_stratify_column(df: pd.DataFrame, stratify_cols: list[str]) -> pd.DataFrame:
        """
        Agrega una columna '_stratify' al DataFrame para uso en train_test_split.
        Combina múltiples columnas en una sola etiqueta estratificada.
        Optimized: Uses vectorized string operations instead of apply() for better performance.
        """
        if not stratify_cols:
            return df

        if len(stratify_cols) == 1 and stratify_cols[0] in df.columns:
            df["_stratify"] = df[stratify_cols[0]].astype(str)
        else:
            # Vectorized approach: convert all columns to string and concatenate
            str_cols = [df[col].astype(str) for col in stratify_cols if col in df.columns]
            if str_cols:
                df["_stratify"] = str_cols[0] if len(str_cols) == 1 else str_cols[0].str.cat(str_cols[1:], sep="_")
            else:
                # Fallback if no valid columns found
                df["_stratify"] = ""
        return df

    @staticmethod
    def clean_and_prepare_data(
        df: pd.DataFrame, stratify_col: list[str] | str | None = None
    ):
        # 1. Cargar asumiendo tabuladores (TSV)
        work_df = df.copy()

        count_pre = len(work_df)
        work_df = work_df.dropna(subset=["gene", "genotype"])
        count_post = len(work_df)
        logger.info(
            f"Eliminadas {count_pre - count_post} filas con valores NaN en 'gene' o 'genotype'."
        )

        # 3. FILTRADO DEFENSIVO: Eliminar filas con genes vacíos o espacios en blanco
        work_df = work_df[work_df["gene"].str.strip() != ""]
        work_df = work_df[work_df["genotype"].str.strip() != ""]

        # 4. CONSTRUCCIÓN DE LA LLAVE
        work_df["haplo_key"] = (
            work_df["gene"].astype(str) + "_" + work_df["genotype"].astype(str)
        )

        # 5. (Opcional) Verificar que los archivos existen en el tree.txt (o disco)
        for col in MULTI_LABEL_COLS:
            if col in work_df.columns:
                work_df[col] = DataLoaderUtils.normalize_multilabel_col(work_df[col])
        if stratify_col:
            work_df = DataLoaderUtils.add_stratify_column(
                work_df,
                stratify_cols=[stratify_col]
                if isinstance(stratify_col, str)
                else stratify_col,
            )

        logger.info(
            f"Dataframe limpio: {len(work_df)} filas válidas generadas con keys tipo 'GENE_VARIANT'."
        )
        return work_df

    @staticmethod
    def _build_drug_index(drug_lib: Path) -> dict[str, Path]:
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
    def _build_genes_index(variant_lib: Path) -> dict[str, dict[str, Path]]:
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
