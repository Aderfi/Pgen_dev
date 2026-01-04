# Pharmagen - Data Utilities

from functools import lru_cache
from rapidfuzz import fuzz, process


@lru_cache(maxsize=512)  # Reduced from 1024 to prevent memory issues
def normalize_drug_names(name: str) -> str:
    """
    Optimized: Normalize drug names with caching for repeated lookups.
    Cache size of 512 balances performance and memory usage.
    Cache stores up to 512 unique drug names which is sufficient for most databases.
    """
    return name.strip().lower().replace(" ", "_")


def map_drug_name(input_name: str, valid_names: list, threshold: int = 92) -> str:
    """
    Optimized: Map drug names using fuzzy matching with pre-normalized cache.
    Removes duplicate normalize function.
    """
    normalized_input = normalize_drug_names(input_name)
    normalized_valids = [normalize_drug_names(name) for name in valid_names]

    match = process.extractOne(
        query=normalized_input,
        choices=normalized_valids,
        scorer=fuzz.WRatio,
        score_cutoff=threshold,
    )
    
    if not match:
        return normalized_input  # No match found

    match_name, score, _ = match

    if score >= threshold:
        # Return original name corresponding to normalized match
        index = normalized_valids.index(match_name)
        return valid_names[index]
    
    return normalized_input  # No adequate match


if __name__ == "__main__":
    ...
