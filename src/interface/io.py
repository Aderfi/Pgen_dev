# Pharmagen - IO Utilities
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

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
TRAIN_DATA_SCHEMA = {
    "drugs_cid": pl.String,              # Contiene pipes "5460033|2907" -> No es Int
    "drugs": pl.String,
    "genotype": pl.String,               # Contiene pipes y letras
    "gene": pl.String,
    "alleles": pl.String,
    "phenotype_category": pl.Categorical, # Cardinalidad baja: Toxicity, Efficacy, Metabolism/PK
    "direction_of_effect": pl.Categorical,# Cardinalidad baja: increased, decreased, __UNDETERMINED__
    "effect_function": pl.Categorical,    # risk, severity, metabolism...
    "effect_type": pl.Categorical,        # side effect, disease...
    "phenotype_product": pl.String,       # Alta variabilidad
    "metabolizer_types": pl.String,
    "population_types": pl.Categorical,   # people, women, men...
    "population_phenotypes_or_diseases": pl.String,
    "comparison_allele(s)_or_genotype(s)": pl.String,
    "comparison_metabolizer_types": pl.String,
    "significance": pl.Categorical,       # yes, not stated
    "is/is_not_associated": pl.Categorical, # Associated with, Not associated with
    "variant_annotation_id": pl.Int64,    # Enteros grandes
    "pmid": pl.Int64,                     # PubMed IDs
    "sentence": pl.String,                # Texto largo
    "notes": pl.String                    # Texto largo
}

# Star allele table is now loaded from data/dicts/star_alleles.tsv via
# src.genomics.star_alleles. The legacy names below are derived from that
# catalog so existing callers keep working unchanged.
from src.genomics.star_alleles import get_default_map as _get_star_map

_STAR_MAP = _get_star_map()
STAR_ALLELE_MAP: dict[str, str] = {
    label: "|".join(_STAR_MAP[label].rsids) for label in _STAR_MAP.labels
}
RSID_TO_STAR_ALLELES: dict[str, list[str]] = _STAR_MAP.rsid_to_labels

class DataLoaderUtils:
    @staticmethod
    def load_dataframe(
        csv_path: str | Path,
        cols: list,
        stratify_col: list[str] | str | None = None,
    ) -> pl.DataFrame:
        """Carga el DataFrame desde CSV o TSV"""

        path_str = str(csv_path)
        separator = "\t" if not path_str.endswith(".csv") else ","
        try:
            df = pl.read_csv(
                path_str,
                separator=separator,
                has_header=True,
                columns=cols,
                schema=TRAIN_DATA_SCHEMA,
                null_values=["", "NA", "NaN", "null", "NA", "N/A"], # Unificación de nulos
                encoding='utf-8',
            )
        except Exception as e:
            logger.error(f"Error reading CSV {path_str}: {e}")
            raise

        return DataLoaderUtils.clean_and_prepare_data(df, stratify_col=stratify_col)

    @staticmethod
    def normalize_multilabel_col(col_name: str, delimiter: str = "|") -> pl.Expr:
        """
        Patrón: Expression Builder.
        Devuelve una EXPRESIÓN de Polars, no modifica in-place.
        Esto permite usarla dentro de un `with_columns` paralelizado.

        Lógica: Split -> Strip -> Unique -> Sort -> Join
        """
        return (
            pl.col(col_name)
            .cast(pl.String)
            .fill_null("")
            .str.split(delimiter) # Convierte a List[String]
            .list.eval(pl.element().str.strip_chars()) # Strip a cada elemento
            # Filtramos strings vacíos dentro de la lista
            .list.eval(pl.element().filter(pl.element() != ""))
            .list.unique() # Elimina duplicados
            .list.sort()   # Orden alfabético determinista
            .list.join(delimiter) # Vuelve a unir en string
        )

    @staticmethod
    def add_stratify_column(df: pl.DataFrame, stratify_cols: list[str]) -> pl.DataFrame:
        """
        Agrega una columna '_stratify' al DataFrame para uso en train_test_split.
        Combina múltiples columnas en una sola etiqueta estratificada.
        """
        if not stratify_cols:
            return df
        valid_cols = [c for c in stratify_cols if c in df.columns]
        if not valid_cols:
            return df

        return df.with_columns(
            _stratify=pl.concat_str(valid_cols, separator="_", ignore_nulls=True)
            .alias("_stratify")
        )

    @staticmethod
    def clean_and_prepare_data(
        df: pl.DataFrame, stratify_col: list[str] | str | None = None
    ) -> pl.DataFrame:
        """
        Limpia y prepara el DataFrame para entrenamiento.

        Operaciones:
        1. Elimina filas con NaN en columnas críticas
        2. Filtra filas con valores vacíos
        3. Construye geno_key (prioriza star alleles sobre rsIDs)
        4. Normaliza columnas multi-label
        5. Agrega columna de estratificación si se requiere

        Args:
            df: DataFrame de entrada con columnas 'gene', 'genotype', y opcionalmente 'alleles'
            stratify_col: Columna(s) para estratificación en train/test split

        Returns:
            DataFrame limpio con columna 'geno_key' añadida
        """
        count_pre = len(df)

        # Definimos expresiones de limpieza de strings
        clean_gene = pl.col("gene").cast(pl.String).str.strip_chars()
        clean_genotype = (
            pl.col("genotype")
            .cast(pl.String)
            .str.strip_chars()
            .str.replace(r"^REF_SEQ\|", "") # Regex replacement
        )

        # Filtramos nulos y strings vacíos APLICANDO EXPRESIONES.
        work_df = df.filter(
            pl.col("gene").is_not_null() &
            pl.col("genotype").is_not_null() &
            (clean_gene != "") &
            (clean_genotype != "")
        ).with_columns([
            clean_gene.alias("gene"),
            clean_genotype.alias("genotype"),
            # Manejo seguro de columna 'alleles' si no existe
            pl.col("alleles").fill_null("").str.strip_chars()
            if "alleles" in df.columns else pl.lit("").alias("alleles")
        ])

        logger.info(f"Eliminadas {count_pre - len(work_df)} filas inválidas.")
        count_before = len(work_df)

        # CONSTRUCCIÓN DE GENO_KEY
        def generate_keys(struct: dict) -> list[str]:
            gene = struct["gene"]
            genotype = struct["genotype"]
            alleles = struct["alleles"]

            keys = set()

            # Prioridad 1: Star alleles en columna 'alleles'
            if alleles and "*" in alleles:
                for part in alleles.split("/"):
                    part = part.strip()
                    if "*" in part:
                        keys.add(f"{gene}_{part}")

            # Prioridad 2: rsIDs en 'genotype' -> Mapeo -> Star Alleles
            parts = [p.strip() for p in genotype.split("|") if p.strip()]

            for rsid in parts:
                if rsid in RSID_TO_STAR_ALLELES:
                    for star in RSID_TO_STAR_ALLELES[rsid]:
                        if "*" in star:
                            suffix = "*" + star.split("*")[-1]
                            keys.add(f"{gene}_{suffix}")
                        else:
                            keys.add(f"{gene}_{star}")

                elif not keys:
                    # Fallback RSID
                    keys.add(f"{gene}_{rsid}")

            # Prioridad 3: Fallback Final
            if not keys and parts:
                 keys.add(f"{gene}_{parts[0]}")

            return list(keys)

        work_df = work_df.with_columns(
                geno_key = pl.struct(["gene", "genotype", "alleles"])
                    .map_elements(generate_keys, return_dtype=pl.List(pl.String))
                    .alias("geno_key")
        )

        work_df = (work_df.explode("geno_key").unique())

        logger.debug(
            f"Expansión: {count_before} -> {len(work_df)} filas "
            f"({len(work_df) - count_before:+d})"
        )

        expressions = []
        for col in MULTI_LABEL_COLS:
            if col in work_df.columns:
                expressions.append(
                    DataLoaderUtils.normalize_multilabel_col(col).alias(col)
                )

        if expressions:
            work_df = work_df.with_columns(expressions)

        # 5. Estratificación
        # ---------------------------------------------
        if stratify_col:
            cols_to_stratify = [stratify_col] if isinstance(stratify_col, str) else stratify_col
            work_df = DataLoaderUtils.add_stratify_column(work_df, cols_to_stratify)

        logger.info(f"Dataframe final: {len(work_df)} filas con geno_key.")
        return work_df

    @staticmethod
    def _build_drug_index(drug_lib: Path) -> dict[str, Path]:
        """Mapea los compound_id con sus rutas reales en disco."""
        index_drugs = {}
        for file_path in drug_lib.glob("*.pt"):
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
        for file_path in variant_lib.glob("*.pt"):
            filename_clean = file_path.stem

            gene_id, variant = filename_clean.split("_", 1)

            if gene_id not in index_genes:
                index_genes[gene_id] = {}
            index_genes[gene_id][variant] = file_path
        return index_genes
