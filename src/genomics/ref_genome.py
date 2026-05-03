# Pharmagen - Pharmacogenetic Prediction and Therapeutic Efficacy
# Copyright (C) 2025 Adrim Hamed Outmani
# Licensed under the GNU GPLv3. See LICENSE file in the project root.
"""Reference genome download and indexing for GRCh38."""

import gzip
import logging
import shutil
import subprocess
from datetime import datetime
from email.utils import parsedate_to_datetime
from pathlib import Path

import requests
from tqdm.auto import tqdm

from src.config.manager import REF_GENOME_DIR, REF_GENOME_FASTA


logger = logging.getLogger(__name__)

# Ensembl primary assembly URL (no alternative haplotypes).
ENSEMBL_URL = (
    "https://ftp.ensembl.org/pub/release-114/fasta/homo_sapiens/dna/"
    "Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz"
)


class ReferenceGenomeManager:
    """Download, refresh, and index the GRCh38 reference genome.

    Ensures the FASTA, the samtools .fai index, and the BWA index files
    exist for downstream alignment / variant-calling steps.
    """

    def __init__(self) -> None:
        self.target_fasta = REF_GENOME_FASTA
        self.download_dir = REF_GENOME_DIR
        self.download_dir.mkdir(parents=True, exist_ok=True)

        # Temporary .gz file used during download.
        self.local_gz = (
            self.download_dir / "Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz"
        )

    def _needs_download(self) -> bool:
        """Return True if the remote file is newer than the local copy."""
        if not self.local_gz.exists():
            return True

        try:
            response = requests.head(ENSEMBL_URL, timeout=30)
            last_modified = response.headers.get("Last-Modified")
            if last_modified is None:
                return False
            remote_time = parsedate_to_datetime(last_modified)
            local_time = datetime.fromtimestamp(self.local_gz.stat().st_mtime).astimezone()
            if remote_time > local_time:
                logger.info("New reference genome version detected on Ensembl (%s).", remote_time)
                return True
            return False
        except Exception as e:  # noqa: BLE001 — network failures are non-fatal here
            logger.warning("Could not check remote timestamp: %s. Skipping download.", e)
            return False

    def download_genome(self) -> None:
        """Download the gzipped FASTA with a progress bar, then decompress."""
        if not self._needs_download() and self.target_fasta.exists():
            logger.info("Local reference genome is up to date.")
            return

        logger.info("Downloading reference genome from %s ...", ENSEMBL_URL)
        response = requests.get(ENSEMBL_URL, stream=True, timeout=60)
        total_size = int(response.headers.get("content-length", 0))

        with (
            open(self.local_gz, "wb") as f,
            tqdm(
                desc="Downloading GRCh38",
                total=total_size,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
            ) as bar,
        ):
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                bar.update(size)
        self._decompress_genome()

    def _decompress_genome(self) -> None:
        """Decompress the .gz directly into the target FASTA path."""
        logger.info("Decompressing reference genome ...")
        with gzip.open(self.local_gz, "rb") as f_in, open(self.target_fasta, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        logger.info("Reference genome ready at: %s", self.target_fasta)

    def index_samtools(self) -> None:
        """Generate the .fai index used for random-access FASTA reads."""
        fai_path = Path(str(self.target_fasta) + ".fai")
        if (
            not fai_path.exists()
            or fai_path.stat().st_mtime < self.target_fasta.stat().st_mtime
        ):
            logger.info("Generating samtools .fai index ...")
            try:
                subprocess.run(
                    ["samtools", "faidx", str(self.target_fasta)], check=True
                )
                logger.info("samtools .fai index created.")
            except subprocess.CalledProcessError:
                logger.error(
                    "samtools faidx failed. Ensure samtools is installed and on PATH."
                )
            except FileNotFoundError:
                logger.error("samtools binary not found on PATH.")
        else:
            logger.info("samtools .fai index already exists and is up to date.")

    def index_bwa(self) -> None:
        """Generate the BWA index files (.bwt, .pac, etc.) used for alignment.

        Note: bwa index can take up to an hour and uses substantial RAM on
        large genomes; consider running it offline on smaller machines.
        """
        bwt_path = Path(str(self.target_fasta) + ".bwt")
        if not bwt_path.exists():
            logger.info("Generating BWA index (this can take ~1 hour) ...")
            try:
                subprocess.run(
                    ["bwa", "index", str(self.target_fasta)], check=True
                )
                logger.info("BWA index created.")
            except subprocess.CalledProcessError:
                logger.error("bwa index failed. Ensure bwa is installed and on PATH.")
            except FileNotFoundError:
                logger.error("bwa binary not found on PATH.")
        else:
            logger.info("BWA index already exists.")

    def run(self) -> None:
        """Full pipeline: download → samtools index → BWA index."""
        self.download_genome()
        if self.target_fasta.exists():
            self.index_samtools()
            self.index_bwa()
        else:
            logger.error("FASTA file not found; cannot index.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ReferenceGenomeManager().run()
