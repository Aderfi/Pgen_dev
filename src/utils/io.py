# Pharmagen - IO Utilities
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
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

STAR_ALLELE_MAP = {
            # --- CYP2D6 (Metabolizador de antidepresivos, opioides, tamoxifeno) ---
            "CYP2D6*3": "rs35742686",   # No Function (AS: 0.0)
            "CYP2D6*4": "rs3892097",    # No Function (AS: 0.0)
            "CYP2D6*6": "rs5030656",    # No Function (AS: 0.0)
            "CYP2D6*9": "rs5030655",    # Decreased Function (AS: 0.5)
            "CYP2D6*10": "rs1065852",   # Decreased Function (AS: 0.25)
            "CYP2D6*17": "rs28371706",  # Decreased Function (AS: 0.5)
            "CYP2D6*29": "rs55811643",  # Decreased Function (AS: 0.5)
            "CYP2D6*41": "rs28371725",  # Decreased Function (AS: 0.5)
            # Nota: *5 (deleción) y *xN (duplicación) requieren análisis de CNV, no rsID único.

            # --- CYP2C19 (Clopidogrel, IBPs) ---
            "CYP2C19*2": "rs4244285",   # No Function (AS: 0.0)
            "CYP2C19*3": "rs4986893",   # No Function (AS: 0.0)
            "CYP2C19*4": "rs28399504",  # No Function (AS: 0.0)
            "CYP2C19*17": "rs12248560", # Increased Function (AS: 1.0+)

            # --- CYP2C9 (Warfarina, Fenitoína, AINEs) ---
            "CYP2C9*2": "rs1799853",    # Decreased Function (AS: 0.5)
            "CYP2C9*3": "rs1057910",    # Severe Decreased Function (AS: 0.0)
            "CYP2C9*5": "rs28371686",   # Decreased Function
            "CYP2C9*6": "rs9332131",    # No Function (AS: 0.0)
            "CYP2C9*8": "rs7900194",    # Decreased Function
            "CYP2C9*11": "rs28371685",  # Decreased Function

            # --- DPYD (Crítico: Toxicidad 5-FU y Capecitabina) ---
            "DPYD*2A": "rs3918290",     # No Function (AS: 0.0)
            "DPYD*13": "rs55886062",    # No Function (AS: 0.0)
            "DPYD*9A": "rs1801159",     # Normal Function (AS: 1.0)
            "c.2846A>T": "rs67376798",  # Decreased Function (AS: 0.5)
            "c.1236G>A": "rs75017182",  # HapB3 - Decreased Function (AS: 0.5)

            # --- SLCO1B1 (Transportador de Estatinas - Miopatía) ---
            "SLCO1B1*5": "rs4149056",   # No Function (AS: 0.0)
            "SLCO1B1*15": "rs4149056|rs2306283", # Haplotipo (AS: 0.0)
            "SLCO1B1*37": "rs2306283",  # Normal Function (AS: 1.0)

            # --- CYP3A5 (Tacrolimus) ---
            "CYP3A5*3": "rs776746",     # No Function (Non-expresser) (AS: 0.0)
            "CYP3A5*6": "rs10276036",   # No Function (AS: 0.0)
            "CYP3A5*7": "rs41303343",   # No Function (AS: 0.0)

            # --- NUDT15 (Tiopurinas en Pediatría/Onco) ---
            "NUDT15*2": "rs116855232|rs147390019", # No Function (AS: 0.0)
            "NUDT15*3": "rs116855232",  # No Function (AS: 0.0)

            # --- NAT2 (Isoniazida - Acetiladores rápidos/lentos) ---
            "NAT2*5": "rs1801280",      # Slow Acetylator (AS: 0.0)
            "NAT2*6": "rs1799930",      # Slow Acetylator (AS: 0.0)
            "NAT2*7": "rs1799931",      # Slow Acetylator (AS: 0.0)
            "NAT2*12": "rs1208",        # Rapid Acetylator (AS: 1.0)
            "NAT2*14": "rs1801279",     # Slow Acetylator (AS: 0.0)

            # --- Otros relevantes ---
            "CYP4F2*3": "rs2108622",    # Decreased Function (Afecta Vit K)
            "CYP2B6*6": "rs2279343|rs3211371", # Decreased Function (Efavirenz)
            "CYP2B6*18": "rs28399499",  # No Function (AS: 0.0)
            "CYP1A2*1F": "rs762551",    # Inducibilidad aumentada (Tabaco)
        }

RSID_TO_STAR_ALLELES:  dict[str, list[str]] = {}
for star_allele, rsids in STAR_ALLELE_MAP.items():
    for rsid in rsids. split("|"):
        rsid = rsid.strip()
        if rsid not in RSID_TO_STAR_ALLELES:
            RSID_TO_STAR_ALLELES[rsid] = []
        RSID_TO_STAR_ALLELES[rsid]. append(star_allele)

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
        """

        def _clean_string(x):
            if pd.isna(x) or str(x).strip() == "" or str(x).lower() == "unknown":
                return ""
            # 1. Split por el delimitador principal
            parts = str(x).split(delimiter)
            # 2. Limpieza de espacios, eliminación de duplicados y orden alfabético
            cleaned_parts = sorted(list({p.strip() for p in parts if p.strip()}))
            # 3. Re-unión con delimitador estándar
            return delimiter.join(cleaned_parts)

        return series.apply(_clean_string)

    @staticmethod
    def add_stratify_column(df: pd.DataFrame, stratify_cols: list[str]) -> pd.DataFrame:
        """
        Agrega una columna '_stratify' al DataFrame para uso en train_test_split.
        Combina múltiples columnas en una sola etiqueta estratificada.
        """
        if not stratify_cols:
            return df

        def _combine_stratify(row):
            return "_".join(str(row[col]) for col in stratify_cols if col in row)

        if len(stratify_cols) == 1 and stratify_cols[0] in df.columns:
            df["_stratify"] = df[stratify_cols[0]].astype(str)
        else:
            df["_stratify"] = df.apply(_combine_stratify, axis=1)
        return df

    @staticmethod
    def clean_and_prepare_data( # noqa
        df: pd.DataFrame, stratify_col: list[str] | str | None = None
    ) -> pd.DataFrame:
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
        work_df = df.copy()  # noqa

        # 1-2. Limpieza inicial (igual que antes)
        count_pre = len(work_df)
        work_df = work_df.dropna(subset=["gene", "genotype"])

        mask_valid = (
            (work_df["gene"].str.strip() != "") &
            (work_df["genotype"].str.strip() != "")
        )
        work_df = work_df[mask_valid].copy()

        logger.info(f"Eliminadas {count_pre - len(work_df)} filas inválidas.")
        count_before = len(work_df)

        # 3. CONSTRUCCIÓN DE GENO_KEY - Enfoque híbrido optimizado

        # Pre-procesar columnas
        genes = work_df["gene"].astype(str).str.strip()
        genotypes = (
            work_df["genotype"].astype(str)
            .str.strip()
            .str.replace(r"^REF_SEQ\|", "", regex=True)
        )

        has_alleles_col = "alleles" in work_df.columns
        if has_alleles_col:
            alleles = work_df["alleles"].fillna("").astype(str).str.strip()
        else:
            alleles = pd.Series("", index=work_df.index)

        # Construir lista de geno_keys para cada fila
        results = []

        for idx, (gene, genotype, allele_str) in enumerate(zip(genes, genotypes, alleles)):
            row_data = work_df.iloc[idx]. to_dict()
            geno_keys = set()

            # Prioridad 1: Star alleles directos
            if "*" in allele_str:
                for allele in allele_str.split("/"):
                    allele = allele.strip()
                    if "*" in allele:
                        geno_keys.add(f"{gene}_{allele}")

            # Prioridad 2: rsID -> star allele
            for rsid in genotype.split("|"):
                rsid = rsid.strip()
                if rsid in RSID_TO_STAR_ALLELES:
                    for star_allele in RSID_TO_STAR_ALLELES[rsid]:
                        if "*" in star_allele:
                            star_suffix = "*" + star_allele.split("*")[-1]
                            geno_keys.add(f"{gene}_{star_suffix}")
                        else:
                            geno_keys.add(f"{gene}_{star_allele}")
                elif not geno_keys:
                    # Fallback solo si no hay matches
                    geno_keys.add(f"{gene}_{rsid}")

            # Si aún vacío, usar primer rsID
            if not geno_keys:
                geno_keys.add(f"{gene}_{genotype.split('|')[0]}")

            # Crear una fila por cada geno_key
            for gk in geno_keys:
                row_copy = row_data.copy()
                row_copy["geno_key"] = gk
                results.append(row_copy)

        # Reconstruir DataFrame
        work_df = pd.DataFrame(results)
        work_df = work_df.drop_duplicates()

        logger.info(
            f"Expansión:  {count_before} -> {len(work_df)} filas "
            f"({len(work_df) - count_before:+d})"
        )

        # 4-5. Normalización y estratificación (igual que antes)
        for col in MULTI_LABEL_COLS:
            if col in work_df.columns:
                work_df[col] = DataLoaderUtils.normalize_multilabel_col(work_df[col])

        if stratify_col:
            work_df = DataLoaderUtils. add_stratify_column(
                work_df,
                stratify_cols=[stratify_col] if isinstance(stratify_col, str) else stratify_col,
            )

        logger.info(f"Dataframe final: {len(work_df)} filas con geno_key.")

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
