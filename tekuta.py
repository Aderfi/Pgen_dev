import json
from pathlib import Path
from typing import Optional


def create_drug_cache(library_path: Optional[Path]) -> dict:
    drug_name_cache = {}

    library_path = Path("library/drugs") if not library_path else library_path
    for file in library_path.glob("*.pt"):
        name = file.stem.lower()
        cid, drug_name = name.split("_", 1)
        if cid.isdigit():
            cid = int(cid)
        drug_name_cache[drug_name] = cid
    return drug_name_cache


if __name__ == "__main__":

    dict_cache = create_drug_cache(Path("library/drugs"))
    print(f"Cache dictionary created with {len(dict_cache)} entries.")

    with open("drug_name_cache.json", "w") as f:
        json.dump(dict_cache, f, indent=1)
