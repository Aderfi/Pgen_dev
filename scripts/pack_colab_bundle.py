"""Pack the data the notebooks need into one zip (for Google Drive / Colab).

``data/`` and ``reports/`` are git-ignored, so a Colab clone has no library,
knowledge-graph export, splits or evaluation reports. This zips exactly what
``notebooks/`` reads, with paths relative to the repository root, so that
``notebooks/colab_setup.py`` can unpack it over the clone:

    uv run python scripts/pack_colab_bundle.py [-o pharmagen_colab_bundle.zip]

Left out: the legacy ``.pt`` drug / pharmacophore graphs (``data/library/drugs``,
``data/library/pharmacophores``), the KG raw download cache and old checkpoints
(trained on an earlier library, unusable by the current model).
"""

from __future__ import annotations

import argparse
import logging
import zipfile
from pathlib import Path

logger = logging.getLogger("pack_colab_bundle")

ROOT = Path(__file__).resolve().parents[1]

#: Files and directories (relative to the repo root) that go into the bundle.
INCLUDE = (
    # Modephere drug library + dev_Pgen's ADMET profiles and HGVS catalog
    "data/library/compounds",
    "data/library/compounds.parquet",
    "data/library/aliases.parquet",
    "data/library/conformers.parquet",
    "data/library/failures.parquet",
    "data/library/build_manifest.json",
    "data/library/admet_profile.parquet",
    "data/library/hgvs_catalog.pt",
    # Modephere pgx exports and the KG training snapshot
    "data/library/pgx",
    "data/library/kg/training_rows.parquet",
    "data/library/kg/allele_variants.parquet",
    "data/library/kg/annotation_edges.parquet",
    "data/library/kg/annotation_variants.parquet",
    # Staged ChEMBL release: offline drug resolution for the live mini-build
    "data/chembl",
    # Evaluation splits (bound to training_rows + hgvs_catalog by sha256)
    "data/splits",
    # Generated reports
    "reports/kg",
    "reports/eval",
    "reports/pharmacophores",
)

#: Already-compressed formats are stored, everything else deflated.
STORED_SUFFIXES = {".parquet", ".zip", ".gz", ".png"}


def iter_files(paths: tuple[str, ...]) -> list[Path]:
    files: list[Path] = []
    for rel in paths:
        path = ROOT / rel
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(p for p in path.rglob("*") if p.is_file()))
        else:
            logger.warning("Missing, skipped: %s", rel)
    return files


def pack(output: Path) -> Path:
    files = iter_files(INCLUDE)
    total = sum(f.stat().st_size for f in files)
    logger.info("Packing %d files (%.1f MB) into %s", len(files), total / 1e6, output)
    with zipfile.ZipFile(output, "w") as zf:
        for f in files:
            method = (
                zipfile.ZIP_STORED
                if f.suffix in STORED_SUFFIXES
                else zipfile.ZIP_DEFLATED
            )
            zf.write(f, f.relative_to(ROOT).as_posix(), compress_type=method)
    logger.info("Wrote %s (%.1f MB)", output, output.stat().st_size / 1e6)
    return output


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=ROOT / "pharmagen_colab_bundle.zip",
        help="zip to write (default: ./pharmagen_colab_bundle.zip)",
    )
    pack(parser.parse_args().output)


if __name__ == "__main__":
    main()
