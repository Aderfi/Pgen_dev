# Pharmagen - IO Utilities
import json
import logging
from typing import List, Optional
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Union
from datetime import datetime
from src.cfg.manager import DIRS, METADATA, PROJECT_ROOT, VERSION

LOGS_DIR = DIRS["logs"]
UNKNOWN_TOKEN = "__UNKNOWN__"

##############################################################################
# JSON HANDLING
##############################################################################

def save_json(data: Dict[str, Any], path: Union[str, Path]):
    """Saves a dictionary to a JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    """Loads a JSON file into a dictionary."""
    with open(path, "r", encoding="utf-8") as f:
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
    print("\n" + "="*60)
    print("NO WARRANTY")
    print("="*60)
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
    print("\n" + "="*60)
    print("REDISTRIBUTION CONDITIONS")
    print("="*60)
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

def load_dataset(
    csv_path: Union[str, Path],
    cols_to_load: List[str],
    multi_label_cols: Optional[List[str]] = None,
    stratify_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Load, Cleand and Prepare Dataset from TSV/CSV.

    Args:
        csv_path (Union[str, Path]): Path to the dataset file.
        cols_to_load (List[str]): List of columns to load.
        multi_label_cols (Optional[List[str]]): Columns that are multi-label.
        stratify_col (Optional[str]): Column(s) to use for stratification during splits.
    Returns:
        pd.DataFrame: Cleaned DataFrame ready for processing.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    # Normalize column requests
    cols_lower = [c.lower() for c in cols_to_load]
    multi_label_lower = [c.lower() for c in (multi_label_cols or [])]

    logger.info(f"Loading dataset from {path.name}...")
    
    # Load
    df = pd.read_csv(path, sep="\t")
    df.columns = df.columns.str.lower().str.strip()

    # Normalize Columns
    df.columns = df.columns.str.lower().str.strip()

    # Clean Content
    for col in df.columns:
        if col in multi_label_lower:
            df[col] = df[col].apply(serialize_multilabel).str.lower()
        else:
            # Single label / Feature cleaning
            df[col] = (
                df[col].fillna(UNKNOWN_TOKEN)
                .astype(str)
                .str.replace(", ", ",")
                .str.replace(r"[,;]+", "|", regex=True)
                .str.strip()
                .str.lower()
            )

    # Stratification Helper
    if stratify_col:
        s_cols = [c.strip() for c in stratify_col.lower().split(",")]
        valid_s_cols = [c for c in s_cols if c in df.columns]
        
        if valid_s_cols:
            df["_stratify"] = df[valid_s_cols].astype(str).agg("|".join, axis=1)
            # Filter singletons to allow splitting
            counts = df["_stratify"].value_counts()
            df = df[df["_stratify"].isin(counts[counts > 1].index)].reset_index(drop=True)
        else:
            df["_stratify"] = "default"
            
    return df
