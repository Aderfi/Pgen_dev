import re
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import pubchempy as pcp
from rapidfuzz import fuzz, process
from tqdm import tqdm  # Import tqdm

# Constants
FILE_INPUT = "final_genomic_data_merged.tsv"
FILE_OUTPUT = "final_data_with_cid.tsv"
SEPARATORS_PATTERN = re.compile(r"[;,\|/\\]")


class DrugResolver:
    """
    Handles the logic for mapping drug names to CIDs using a hierarchy:
    1. Exact Match (Cache)
    2. Fuzzy Match (RapidFuzz)
    3. External API (PubChem)
    """

    def __init__(self, library_path: Path):
        self.library_path = library_path
        self.reference_cache: Dict[str, int] = self._load_library_cache()
        # Runtime cache to store results found during execution (avoids re-querying API)
        self.runtime_cache: Dict[str, Optional[int]] = {}

    def _load_library_cache(self) -> Dict[str, int]:
        """Loads local .pt files into a dictionary."""
        cache = {}
        if not self.library_path.exists():
            print(f"⚠️ Warning: Library path {self.library_path} does not exist.")
            return cache

        # Iterate over files (using tqdm here is optional but fast)
        files = list(self.library_path.glob("*.pt"))
        if not files:
            return cache

        print(f"   Populating reference cache from {len(files)} files...")
        for file in files:
            try:
                name_stem = file.stem.lower()
                parts = name_stem.split("_", 1)
                if len(parts) == 2 and parts[0].isdigit():
                    cid, drug_name = int(parts[0]), parts[1]
                    cache[drug_name] = cid
            except Exception:
                continue

        return cache

    def resolve_term(self, name: str) -> Optional[int]:
        """Resolves a single drug name to a CID."""
        name = name.strip().lower()
        if not name or name in ["nan", "none"]:
            return None

        # 1. Check Reference Library
        if name in self.reference_cache:
            return self.reference_cache[name]

        # 2. Check Runtime Cache (Already seen in this run)
        if name in self.runtime_cache:
            return self.runtime_cache[name]

        # 3. Fuzzy Match
        match_result = process.extractOne(
            name, self.reference_cache.keys(), scorer=fuzz.WRatio
        )

        if match_result:
            match_name, score, _ = match_result
            if score >= 92:
                cid = self.reference_cache[match_name]
                self.runtime_cache[name] = cid
                return cid

        # 4. PubChem API Fallback
        try:
            compounds = pcp.get_compounds(name, "name")
            if compounds:
                cid = compounds[0].cid
                self.runtime_cache[name] = cid
                return cid
        except Exception:
            pass

        # Cache negative result to avoid retrying
        self.runtime_cache[name] = None
        return None

    def process_row(self, raw_entry: Any) -> Optional[str]:
        """
        Parses a row entry, splits by delimiters, and resolves CIDs.
        """
        if pd.isna(raw_entry):
            return None

        text = str(raw_entry).strip()
        if not text:
            return None

        potential_names = re.split(SEPARATORS_PATTERN, text)

        resolved_cids = []
        for name in potential_names:
            cid = self.resolve_term(name)
            if cid is not None:
                resolved_cids.append(str(cid))

        return "|".join(resolved_cids) if resolved_cids else None


def main():
    # 1. Load Data
    print("📂 Loading dataset...")
    try:
        df = pd.read_csv(FILE_INPUT, sep="\t", low_memory=False)
    except FileNotFoundError:
        print(f"❌ Error: File '{FILE_INPUT}' not found.")
        return

    # 2. Clean Columns
    df.columns = [col.strip().lower() for col in df.columns]
    if "drug(s)" in df.columns:
        df.rename(columns={"drug(s)": "drugs"}, inplace=True)
    if "variant/haplotypes" in df.columns:
        df.rename(columns={"variant/haplotypes": "genotype"}, inplace=True)

    print(f"   Rows: {len(df)} | Columns: {len(df.columns)}")

    # 3. Initialize Resolver
    library_path = Path("../dev_Pharmagen/library/drugs")
    resolver = DrugResolver(library_path)
    print(f"📚 Reference Cache loaded: {len(resolver.reference_cache)} unique drugs.")

    # 4. Map Drugs (With TQDM Progress Bar)
    print("\n🔄 Mapping drugs to CIDs...")

    # Initialize tqdm for pandas
    tqdm.pandas(desc="Processing Rows", unit="row")

    work_df = df.copy()

    # Use .progress_apply() instead of .apply()
    work_df["drugs_cid"] = work_df["drugs"].progress_apply(resolver.process_row)

    # 5. Save Output
    work_df.to_csv(FILE_OUTPUT, sep="\t", index=False)

    # Stats
    total_cached = len(resolver.runtime_cache)
    print("\n✅ Processing Complete.")
    print(f"   New terms resolved & cached: {total_cached}")
    print(f"   File saved to: {FILE_OUTPUT}")


if __name__ == "__main__":
    main()
