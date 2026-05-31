# Pharmagen - Pharmacogenetic Prediction and Therapeutic Efficacy
# Copyright (C) 2025 Adrim Hamed Outmani
# Licensed under the GNU GPLv3. See LICENSE file in the project root.
"""NGS analysis pipeline — wrappers around external bioinformatics tools.

Phases:
    1. Raw read QC + trimming         (FastQC, fastp)
    2. Reference alignment            (BWA-MEM, samtools, Picard, Qualimap)
    3. Variant calling                (Freebayes, vcftools)
    4. Variant annotation             (VEP)

Each phase has its own class so callers can run sub-pipelines (e.g. start
from already-aligned BAMs).
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from pathlib import Path

from src.config import PROJECT_ROOT
from src.config import get_settings as _get_settings

_paths = _get_settings().paths
DATA_DIR = _paths.data
REF_GENOME_FASTA = _paths.ref_genome_fasta

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shell tool executor
# ---------------------------------------------------------------------------


class BioToolExecutor:
    """Base class for bioinformatics CLI wrappers.

    Runs subprocesses with stdout/stderr captured and structured logging.
    Commands are passed as argv lists (no shell=True) — pipelines that need
    pipes use Python-side composition (e.g. ``samtools sort`` after
    ``bwa mem``) rather than shell pipes.
    """

    def __init__(self, threads: int = 4) -> None:
        self.threads = str(threads)

    def _run(self, command: list[str], description: str) -> subprocess.CompletedProcess:
        logger.info("Starting: %s", description)
        logger.debug("CMD: %s", " ".join(command))

        if sys.platform == "win32":
            logger.warning(
                "Running NGS pipeline on native Windows. If pipes fail or tools "
                "are missing, prefer WSL2."
            )

        try:
            process = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                cwd=str(PROJECT_ROOT),
                timeout=14400,  # 4 hours hard cap; long jobs should override
            )
            logger.info("Finished: %s", description)
            return process
        except subprocess.CalledProcessError as e:
            logger.error("Critical failure in %s (exit %s)", description, e.returncode)
            if e.stdout:
                logger.error("stdout (tail): %s", e.stdout[-500:])
            if e.stderr:
                logger.error("stderr: %s", e.stderr)
            msg = f"NGS pipeline step failed ({description}). See logs for details."
            raise RuntimeError(msg) from e
        except FileNotFoundError as e:
            msg = f"Tool not found on PATH for step: {description}"
            raise RuntimeError(msg) from e


# ---------------------------------------------------------------------------
# Phase 1: raw-read QC and trimming
# ---------------------------------------------------------------------------


class ProcessRawGenome(BioToolExecutor):
    """Quality control and adapter trimming. Tools: FastQC, fastp."""

    def __init__(self, output_dir: Path, threads: int = 4) -> None:
        super().__init__(threads)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_fastqc(self, fastq_files: list[Path], step_name: str = "pre_qc") -> Path:
        out_dir = self.output_dir / step_name
        out_dir.mkdir(exist_ok=True)
        cmd = ["fastqc", "-t", self.threads, "-o", str(out_dir), *map(str, fastq_files)]
        self._run(cmd, f"FastQC ({step_name})")
        return out_dir

    def run_fastp(self, r1: Path, r2: Path, sample_name: str) -> dict[str, Path]:
        clean_dir = self.output_dir / "clean_reads"
        clean_dir.mkdir(exist_ok=True)
        out_r1 = clean_dir / f"{sample_name}_R1_clean.fastq.gz"
        out_r2 = clean_dir / f"{sample_name}_R2_clean.fastq.gz"
        report_html = clean_dir / f"{sample_name}_fastp.html"
        report_json = clean_dir / f"{sample_name}_fastp.json"
        cmd = [
            "fastp",
            "-i",
            str(r1),
            "-I",
            str(r2),
            "-o",
            str(out_r1),
            "-O",
            str(out_r2),
            "--detect_adapter_for_pe",
            "-w",
            self.threads,
            "-h",
            str(report_html),
            "-j",
            str(report_json),
        ]
        self._run(cmd, f"fastp cleaning ({sample_name})")
        return {"r1": out_r1, "r2": out_r2}


# ---------------------------------------------------------------------------
# Phase 2: alignment to reference
# ---------------------------------------------------------------------------


class MappingAlignmentAnalysis(BioToolExecutor):
    """Read alignment + dedup + BAM QC. Tools: BWA, samtools, Picard, Qualimap."""

    def __init__(
        self,
        output_dir: Path,
        ref_genome: Path = REF_GENOME_FASTA,
        threads: int = 8,
    ) -> None:
        super().__init__(threads)
        self.output_dir = Path(output_dir)
        self.ref_genome = ref_genome
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._check_bwa_index()

    def _check_bwa_index(self) -> None:
        if not Path(str(self.ref_genome) + ".bwt").exists():
            logger.warning("BWA index not found; building (this can take a while) ...")
            self._run(["bwa", "index", str(self.ref_genome)], "BWA indexing")

    def map_reads(self, r1: Path, r2: Path, sample_name: str) -> Path:
        """Align with BWA-MEM and sort with samtools.

        Pipes the BWA stdout into samtools sort via Python plumbing rather
        than ``shell=True``.
        """
        bam_dir = self.output_dir / "bams"
        bam_dir.mkdir(exist_ok=True)
        sorted_bam = bam_dir / f"{sample_name}_sorted.bam"

        # Read group is required by downstream tools (GATK/Picard).
        rg_tag = f"@RG\\tID:{sample_name}\\tSM:{sample_name}\\tPL:ILLUMINA"

        bwa = subprocess.Popen(
            [
                "bwa",
                "mem",
                "-t",
                self.threads,
                "-R",
                rg_tag,
                str(self.ref_genome),
                str(r1),
                str(r2),
            ],
            stdout=subprocess.PIPE,
        )
        try:
            subprocess.run(
                ["samtools", "sort", "-@", self.threads, "-o", str(sorted_bam), "-"],
                stdin=bwa.stdout,
                check=True,
            )
        finally:
            if bwa.stdout is not None:
                bwa.stdout.close()
            bwa.wait()
        if bwa.returncode != 0:
            msg = f"bwa mem failed for sample {sample_name}"
            raise RuntimeError(msg)

        self._run(["samtools", "index", str(sorted_bam)], "samtools index BAM")
        return sorted_bam

    def preprocess_identify_duplicates(self, input_bam: Path, sample_name: str) -> Path:
        dedup_bam = self.output_dir / "bams" / f"{sample_name}_dedup.bam"
        metrics = self.output_dir / "bams" / f"{sample_name}_dedup_metrics.txt"
        cmd = [
            "picard",
            "MarkDuplicates",
            f"I={input_bam}",
            f"O={dedup_bam}",
            f"M={metrics}",
            "REMOVE_DUPLICATES=false",
            "VALIDATION_STRINGENCY=LENIENT",
        ]
        self._run(cmd, "Picard MarkDuplicates")
        self._run(["samtools", "index", str(dedup_bam)], "samtools index dedup BAM")
        return dedup_bam

    def quality_analysis(self, bam_file: Path) -> None:
        qm_dir = self.output_dir / "qualimap_report"
        # Qualimap occasionally fails in headless environments without X11.
        try:
            self._run(
                [
                    "qualimap",
                    "bamqc",
                    "-bam",
                    str(bam_file),
                    "-outdir",
                    str(qm_dir),
                    "--java-mem-size=4G",
                ],
                "Qualimap BamQC",
            )
        except RuntimeError:
            logger.warning(
                "Qualimap failed (likely a GUI/X11 issue); continuing pipeline."
            )


# ---------------------------------------------------------------------------
# Phase 3: variant calling and filtering
# ---------------------------------------------------------------------------


class VariantIdentificationAnalysis(BioToolExecutor):
    """Variant calling. Tools: Freebayes, vcftools."""

    def __init__(self, output_dir: Path, ref_genome: Path = REF_GENOME_FASTA) -> None:
        # Freebayes does not parallelise well across threads.
        super().__init__(threads=1)
        self.output_dir = Path(output_dir)
        self.ref_genome = ref_genome
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def identify_variants(self, bam_file: Path, sample_name: str) -> Path:
        vcf_raw = self.output_dir / f"{sample_name}_raw.vcf"
        # Freebayes writes to stdout — we redirect by opening a file and using stdout=.
        with vcf_raw.open("w") as out:
            subprocess.run(
                ["freebayes", "-f", str(self.ref_genome), str(bam_file)],
                check=True,
                stdout=out,
            )
        return vcf_raw

    def filter_variants(self, input_vcf: Path, sample_name: str) -> Path:
        vcf_filtered = self.output_dir / f"{sample_name}_filtered.vcf"
        out_prefix = self.output_dir / f"{sample_name}_temp"
        # Standard clinical filters: quality > 20, depth > 10.
        self._run(
            [
                "vcftools",
                "--vcf",
                str(input_vcf),
                "--minQ",
                "20",
                "--minDP",
                "10",
                "--recode",
                "--recode-INFO-all",
                "--out",
                str(out_prefix),
            ],
            "vcftools filtering",
        )
        # Rename vcftools' .recode.vcf output.
        temp_out = self.output_dir / f"{sample_name}_temp.recode.vcf"
        if temp_out.exists():
            shutil.move(str(temp_out), str(vcf_filtered))
        return vcf_filtered


# ---------------------------------------------------------------------------
# Phase 4: variant annotation
# ---------------------------------------------------------------------------


class VariantAnnotator(BioToolExecutor):
    """Functional variant annotation via Ensembl VEP.

    Requires VEP installed and a local cache configured (~/.vep).
    """

    def __init__(
        self, output_dir: Path, threads: int = 4, assembly: str = "GRCh38"
    ) -> None:
        super().__init__(threads)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.assembly = assembly

    def annotate_variants(self, input_vcf: Path, sample_name: str) -> Path:
        annotated_vcf = self.output_dir / f"{sample_name}_annotated.vcf"
        stats_file = self.output_dir / f"{sample_name}_vep_summary.html"
        cmd = [
            "vep",
            "-i",
            str(input_vcf),
            "-o",
            str(annotated_vcf),
            "--assembly",
            self.assembly,
            "--cache",
            "--offline",
            "--force_overwrite",
            "--vcf",
            "--stats_file",
            str(stats_file),
            "--pick",
            "--fork",
            self.threads,
        ]
        self._run(cmd, "VEP annotation")
        return annotated_vcf


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_full_ngs_pipeline(r1: Path, r2: Path, sample_name: str) -> None:
    """Run all four phases sequentially for a single paired-end sample."""
    base_results = Path(DATA_DIR) / "processed" / sample_name
    logger.info("Starting pharmacogenetic NGS pipeline for sample: %s", sample_name)

    try:
        step1 = ProcessRawGenome(base_results / "01_qc")
        step1.run_fastqc([r1, r2], "raw_fastqc")
        clean_files = step1.run_fastp(r1, r2, sample_name)
        step1.run_fastqc([clean_files["r1"], clean_files["r2"]], "clean_fastqc")

        step2 = MappingAlignmentAnalysis(base_results / "02_alignment")
        raw_bam = step2.map_reads(clean_files["r1"], clean_files["r2"], sample_name)
        final_bam = step2.preprocess_identify_duplicates(raw_bam, sample_name)
        step2.quality_analysis(final_bam)

        step3 = VariantIdentificationAnalysis(base_results / "03_variants")
        raw_vcf = step3.identify_variants(final_bam, sample_name)
        filtered_vcf = step3.filter_variants(raw_vcf, sample_name)

        step4 = VariantAnnotator(base_results / "04_annotation")
        final_vcf = step4.annotate_variants(filtered_vcf, sample_name)

        logger.info("Pipeline completed successfully.")
        logger.info("Annotated VCF: %s", final_vcf)
        logger.info(
            "VEP report: %s",
            base_results / "04_annotation" / f"{sample_name}_vep_summary.html",
        )
    except Exception:  # noqa: BLE001 — surface anything from external tools as a single failure
        logger.exception("NGS pipeline failed for sample %s", sample_name)
        raise
