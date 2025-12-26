# Pharmagen - Data Utilities
import torch

from typing import Dict, Tuple, Union

from torch_geometric.data import Data
from sklearn.preprocessing import OneHotEncoder

from rdkit import Chem
from rdkit.Chem import rdchem
from rdkit.Chem import Atom

from rapidfuzz import process, fuzz

def normalize_drug_names(name: str) -> str:
    """Normaliza nombres de fármacos para consistencia."""
    return name.strip().lower().replace(" ", "_")

def map_drug_name(input_name: str, valid_names: list, threshold: int = 92) -> str:
    """Mapea un nombre de fármaco a la lista de nombres válidos usando fuzzy matching."""
    def normalize_drug_names(name: str) -> str:
        """Normaliza nombres de fármacos para consistencia."""
        return name.strip().lower().replace(" ", "_")

    normalized_input = normalize_drug_names(input_name)
    normalized_valids = [normalize_drug_names(name) for name in valid_names]

    match = process.extractOne(
        query=normalized_input,
        choices=normalized_valids,
        scorer=fuzz.WRatio,
        score_cutoff=threshold
    )
    if not match:
        return normalized_input  # No se encontró coincidencia adecuada

    match_name, score, _ = match

    if score >= threshold:
    # Retornar el nombre original correspondiente
        index = normalized_valids.index(match_name)
        return valid_names[index]
    else:
        return normalized_input  # No se encontró coincidencia adecuada

if __name__ == "__main__":
    ...



