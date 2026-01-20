import argparse
import os
import re
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import networkx as nx
import polars as pl
import torch
from pyfaidx import Fasta
from rdkit import Chem
from rdkit.Chem import rdchem
from torch_geometric.data import Data
from tqdm import tqdm

DOC_PRES = """
# =============================================================================
#  Genomic Graph Builder for Pharmacogenomics Variants
# =============================================================================
Documentation:
This script constructs a unified library of genomic graphs for pharmacogenomics
variants using PyTorch Geometric. It processes variant data from multiple
sources, validates them against a reference genome, and generates graph
representations suitable for machine learning tasks.

# Features:
- Supports SNPs, MNPs, Insertions, Deletions, and Star Alleles.
- Extracts genomic context from a reference FASTA and GFF annotation.
- Utilizes Polars for efficient data processing.
- Generates detailed graph structures with atom and bond features using RDKit.
- Saves the final library in Parquet format for easy access.

# Requirements:
    - Python 3.10+
    - PyTorch Geometric
    - RDKit
    - polars
    - pyfaidx
        ...
"""
DOC_FILES = """
# FILES FOR BUILDING THE LIBRARY:

    - DRUG_GRAPH_LIBRARY: .tsv file with AT LEAST columns: "cid"    "cmpdname"	"smiles" -- from PubChem
    - REFERENCE GENOME: FASTA file (e.g., GRCh38) and GFF annotation file. (NCBI Nomenclature prefered)
    - PGX VARIANTS: Folder structure with per-gene VCF files containing star alleles and variants. (PharmVarDB format)
    - OTHER VARIANTS: Optional TSV file with additional variants to include with cols:
        snp     --  [string(rs+id)]
        snp_id  --  [int (snp without rs)]
        chr	    -- (chrNUMBER|LETTER, NUMBER|LETTER or NC_XXXXXXXXX.X format)
        pos	    -- (1-based position)[int]
        variant	-- [string: REF>ALT, e.g., A>G, AT>- (deletion), - >AG (insertion)]
        variant_type	-- [string: SNP, MNP, INS, DEL, STAR_ALLELE]
        gene	-- [string: gene name, optional, will be inferred if missing]
        Ref_Allele	-- [string: reference allele]
        Alt_Allele	-- [string: alternate allele]
"""

DOC_OUTPUTS = """
##############
## OUTPUTS: ##
##############

ROOT_DIR/
    library/
    ├── genomic_variants.parquet         -- Parquet file with validated variant data.
    ├── graphs/   - One .pt file per variant graph. One dir per gene. || All variants for each gene in a single dir. Gene name as dir name.
    │   ├── CYP2D6/
    │   │   ├── CYP2D6_star4.pt
    │   ├── DPYD/
    │   │   ├── DPYD_rs3918290.pt
    │   ...
    └── drugs/                        -- Directory with graphs for drug molecules from SMILES.
        ├── 12345_drugname.pt
        ├── 67890_drugname2.pt
        ...
"""
DOC_USAGE = """
# USAGE:
    +
    +    USE A CODE EDITOR TO CONFIGURE FILENAMES ON "if __name__ == '__main__':" SECTION. At the end of the file.
    +    python library_creator.py
    +
    + DONE!


# =============================================================================
"""

# =============================================================================
#  GLOBAL CONFIGURATION AND MAPPINGS
# =============================================================================

BASE_DIR = Path("data")
REF_DIR = BASE_DIR / "ref_genome"
LIB_DIR = Path("src/library")

# Input Files
GENE_VAR_TSV = BASE_DIR / "snp_data_output.tsv"
DRUGS_TSV = BASE_DIR / "drugs_cid.tsv"

# Reference Files
FASTA_FILE = REF_DIR / "genome.fna"
GFF_FILE = REF_DIR / "gen_annotations.gff"
PGX_FOLDER = BASE_DIR / "haplotype_variants"

# Outputs
GENE_OUT_DIR = LIB_DIR / "gene_graphs"
DRUG_OUT_DIR = LIB_DIR / "drugs"
PARQUET_FILE = LIB_DIR / "genome_library.parquet"

NA_VALUES = ["-", ".", "N/A", "nan", "None", ""]

PGX_METADATA = {
    # --- ENZIMAS MAYORES (METABOLISMO DE FÁRMACOS) ---
    "CYP2D6": {
        "*1": {"metabolic_function": "Normal Function", "activity_score": 1.0},
        "*2": {"metabolic_function": "Normal Function", "activity_score": 1.0},
        "*3": {"metabolic_function": "No Function", "activity_score": 0.0},
        "*4": {"metabolic_function": "No Function", "activity_score": 0.0},
        "*5": {"metabolic_function": "No Function (Deletion)", "activity_score": 0.0},
        "*6": {"metabolic_function": "No Function", "activity_score": 0.0},
        "*9": {"metabolic_function": "Decreased Function", "activity_score": 0.5},
        "*10": {"metabolic_function": "Decreased Function", "activity_score": 0.25},
        "*17": {"metabolic_function": "Decreased Function", "activity_score": 0.5},
        "*29": {"metabolic_function": "Decreased Function", "activity_score": 0.5},
        "*41": {"metabolic_function": "Decreased Function", "activity_score": 0.5},
        "*1xN": {"metabolic_function": "Increased Function", "activity_score": 2.0},
        "*2xN": {"metabolic_function": "Increased Function", "activity_score": 2.0},
        "default": {"metabolic_function": "Unknown_DEF", "activity_score": None},
    },
    "CYP2C19": {
        "*1": {"metabolic_function": "Normal Function", "activity_score": 1.0},
        "*2": {"metabolic_function": "No Function", "activity_score": 0.0},
        "*3": {"metabolic_function": "No Function", "activity_score": 0.0},
        "*4": {"metabolic_function": "No Function", "activity_score": 0.0},
        "*17": {"metabolic_function": "Increased Function", "activity_score": 1.0},
        "default": {"metabolic_function": "Unknown", "activity_score": None},
    },
    "CYP2C9": {
        "*1": {"metabolic_function": "Normal Function", "activity_score": 1.0},
        "*2": {"metabolic_function": "Decreased Function", "activity_score": 0.5},
        "*3": {"metabolic_function": "Decreased Function (Severe)", "activity_score": 0.0},
        "*5": {"metabolic_function": "Decreased Function", "activity_score": None},
        "*6": {"metabolic_function": "No Function", "activity_score": 0.0},
        "*8": {"metabolic_function": "Decreased Function", "activity_score": None},
        "*11": {"metabolic_function": "Decreased Function", "activity_score": None},
        "default": {"metabolic_function": "Unknown", "activity_score": None},
    },
    "CYP2B6": {
        "*1": {"metabolic_function": "Normal Function", "activity_score": 1.0},
        "*4": {"metabolic_function": "Increased Function", "activity_score": None},
        "*6": {"metabolic_function": "Decreased Function", "activity_score": 0.0},
        "*9": {"metabolic_function": "Decreased Function", "activity_score": None},
        "*18": {"metabolic_function": "No Function", "activity_score": 0.0},
        "default": {"metabolic_function": "Unknown", "activity_score": None},
    },
    "CYP3A4": {
        "*1": {"metabolic_function": "Normal Function", "activity_score": 1.0},
        "*20": {"metabolic_function": "No Function", "activity_score": 0.0},
        "*22": {"metabolic_function": "Decreased Function", "activity_score": 0.5},
        "*26": {"metabolic_function": "No Function", "activity_score": 0.0},
        "default": {"metabolic_function": "Unknown", "activity_score": None},
    },
    "CYP3A5": {
        "*1": {"metabolic_function": "Normal Function (Expresser)", "activity_score": 1.0},
        "*3": {"metabolic_function": "No Function (Non-expresser)", "activity_score": 0.0},
        "*6": {"metabolic_function": "No Function", "activity_score": 0.0},
        "*7": {"metabolic_function": "No Function", "activity_score": 0.0},
        "default": {"metabolic_function": "Unknown", "activity_score": None},
    },
    "CYP1A2": {
        "*1F": {"metabolic_function": "Inducible/Normal", "activity_score": 1.0},
        "*1C": {"metabolic_function": "Decreased Function", "activity_score": 0.5},
        "*1K": {"metabolic_function": "Decreased Function", "activity_score": 0.5},
        "default": {"metabolic_function": "Normal Function", "activity_score": 1.0},
    },
    "CYP2C8": {
        "*1": {"metabolic_function": "Normal Function", "activity_score": 1.0},
        "*2": {"metabolic_function": "Decreased Function", "activity_score": None},
        "*3": {"metabolic_function": "Decreased/Altered", "activity_score": None},
        "*4": {"metabolic_function": "Decreased Function", "activity_score": None},
        "default": {"metabolic_function": "Unknown", "activity_score": None},
    },
    # --- TOXICIDAD / SEGURIDAD (CRÍTICOS) ---
    "DPYD": {
        "*1": {"metabolic_function": "Normal Function", "activity_score": 1.0},
        "*2A": {"metabolic_function": "No Function", "activity_score": 0.0},
        "*13": {"metabolic_function": "No Function", "activity_score": 0.0},
        "HapB3": {"metabolic_function": "Decreased Function", "activity_score": 0.5},
        "c.2846A>T": {"metabolic_function": "Decreased Function", "activity_score": 0.5},
        "c.1236G>A": {"metabolic_function": "Decreased Function", "activity_score": 0.5},
        "*9A": {"metabolic_function": "Normal Function", "activity_score": 1.0},
        "default": {"metabolic_function": "Unknown", "activity_score": None},
    },
    "NUDT15": {
        "*1": {"metabolic_function": "Normal Function", "activity_score": 1.0},
        "*2": {"metabolic_function": "No Function", "activity_score": 0.0},
        "*3": {"metabolic_function": "No Function", "activity_score": 0.0},
        "*4": {"metabolic_function": "Uncertain", "activity_score": None},
        "default": {"metabolic_function": "Unknown", "activity_score": None},
    },
    "SLCO1B1": {
        "*1": {"metabolic_function": "Normal Function", "activity_score": 1.0},
        "*5": {"metabolic_function": "No Function/Decreased", "activity_score": 0.0},
        "*15": {"metabolic_function": "No Function/Decreased", "activity_score": 0.0},
        "*37": {"metabolic_function": "Normal Function", "activity_score": 1.0},
        "default": {"metabolic_function": "Unknown", "activity_score": None},
    },
    "NAT2": {
        "*4": {"metabolic_function": "Rapid Acetylator", "activity_score": 1.0},
        "*5": {"metabolic_function": "Slow Acetylator", "activity_score": 0.0},
        "*6": {"metabolic_function": "Slow Acetylator", "activity_score": 0.0},
        "*7": {"metabolic_function": "Slow Acetylator", "activity_score": 0.0},
        "*12": {"metabolic_function": "Rapid Acetylator", "activity_score": 1.0},
        "*14": {"metabolic_function": "Slow Acetylator", "activity_score": 0.0},
        "default": {"metabolic_function": "Unknown", "activity_score": None},
    },
    "CYP4F2": {
        "*1": {"metabolic_function": "Normal Function", "activity_score": 1.0},
        "*3": {"metabolic_function": "Decreased Function", "activity_score": 0.5},
        "default": {"metabolic_function": "Unknown", "activity_score": None},
    },
    "CYP2A6": {
        "*1": {"metabolic_function": "Normal Function", "activity_score": 1.0},
        "*2": {"metabolic_function": "No Function", "activity_score": 0.0},
        "*4": {"metabolic_function": "No Function (Deletion)", "activity_score": 0.0},
        "*9": {"metabolic_function": "Decreased Function", "activity_score": 0.5},
        "*12": {"metabolic_function": "Decreased Function", "activity_score": 0.5},
        "default": {"metabolic_function": "Unknown", "activity_score": None},
    },
    "CYP2A13": {
        "*1": {"metabolic_function": "Normal Function", "activity_score": 1.0},
        "*2": {"metabolic_function": "Decreased Function", "activity_score": 0.5},
        "*3": {"metabolic_function": "Decreased Function", "activity_score": 0.5},
        "*4": {"metabolic_function": "Decreased Function", "activity_score": 0.5},
        "*7": {"metabolic_function": "No Function", "activity_score": 0.0},
        "default": {"metabolic_function": "Unknown", "activity_score": None},
    },
}

FXN_CATEGORIES = {
    "coding": [
        "missense_variant",
        "coding_sequence_variant",
        "synonymous_variant",
        "stop_gained",
        "frameshift_variant",
        "stop_lost",
    ],
    "regulatory": [
        "3_prime_UTR_variant",
        "5_prime_UTR_variant",
        "upstream_transcript_variant",
        "downstream_transcript_variant",
        "regulatory_region_variant",
    ],
    "splicing": [
        "intron_variant",
        "splice_acceptor_variant",
        "splice_donor_variant",
        "splice_region_variant",
    ],
    "intergenic": ["intergenic_variant", "__INTERGENIC__"],
}

DPYD_RS_MAP = {
    "rs3918290": "*2A",
    "rs55886062": "*13",
    "rs67376798": "c.2846A>T",
    "rs75017182": "HapB3",
    "rs1801265": "*4",
}

# Global Variables for Worker access logic (adapted for Polars map_elements)
GLOBAL_GENOME = None
GLOBAL_GENE_TREES = None
GLOBAL_CHROM_MAPPING = {
    "1": "NC_000001.11",
    "2": "NC_000002.12",
    "3": "NC_000003.12",
    "4": "NC_000004.12",
    "5": "NC_000005.10",
    "6": "NC_000006.12",
    "7": "NC_000007.14",
    "8": "NC_000008.11",
    "9": "NC_000009.12",
    "10": "NC_000010.11",
    "11": "NC_000011.10",
    "12": "NC_000012.12",
    "13": "NC_000013.11",
    "14": "NC_000014.9",
    "15": "NC_000015.10",
    "16": "NC_000016.10",
    "17": "NC_000017.11",
    "18": "NC_000018.10",
    "19": "NC_000019.10",
    "20": "NC_000020.11",
    "21": "NC_000021.9",
    "22": "NC_000022.11",
    "X": "NC_000023.11",
    "Y": "NC_000024.10",
    "M": "NC_012920.1",
    "MT": "NC_012920.1",
}


# =============================================================================
#  HELPER FUNCTIONS
# =============================================================================


def smiles_to_graph_complete(smiles: str):
    if not isinstance(smiles, str):
        return None
    mol: rdchem.Mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # 1. NODES (Atoms)
    atom_features = []
    for atom in mol.GetAtoms():
        features = [atom.GetAtomicNum() / 100.0]
        features.append(atom.GetDegree())


        features += one_hot_encoding(atom.GetDegree(), [0, 1, 2, 3, 4])
        features += one_hot_encoding(atom.GetFormalCharge(), [-2, -1, 0, 1, 2])
        features += one_hot_encoding(atom.GetHybridization(), [rdchem.HybridizationType.SP, rdchem.HybridizationType.SP2, rdchem.HybridizationType.SP3])
        features += one_hot_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
        features += one_hot_encoding(atom.GetChiralTag(), [rdchem.ChiralType.CHI_UNSPECIFIED, rdchem.ChiralType.CHI_TETRAHEDRAL_CW, rdchem.ChiralType.CHI_TETRAHEDRAL_CCW])
        features.append(1 if atom.GetIsAromatic() else 0)
        features.append(atom.GetMass() * 0.01)
        atom_features.append(features)

    x = torch.tensor(atom_features, dtype=torch.float)

    # 2. EDGES (Bonds)
    edge_indices, edge_attrs = [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_feats = one_hot_encoding(bond.GetBondType(), [rdchem.BondType.SINGLE, rdchem.BondType.DOUBLE, rdchem.BondType.TRIPLE, rdchem.BondType.AROMATIC])
        bond_feats += [1 if bond.GetIsConjugated() else 0, 1 if bond.IsInRing() else 0, 1 if bond.GetStereo() != rdchem.BondStereo.STEREONONE else 0]
        # Bond_Type[SINGLE, DOUBLE, TRIPLE, AROMATIC], Conjugated[1,0], In_Ring[1,0], Stereo[1,0], [start, end], [end, start]
        edge_indices += [[start, end], [end, start]]
        edge_attrs += [bond_feats, bond_feats]

    if not edge_indices:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 7), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def one_hot_encoding(value, possible_values):
    encoding = [0] * len(possible_values)
    if value in possible_values:
        encoding[possible_values.index(value)] = 1
    return encoding


def parse_fxn_class(fxn_str):
    """Parses FXN_CLASS string into a dictionary of flags."""
    flags = {"coding": 0.0, "regulatory": 0.0, "splicing": 0.0, "intergenic": 0.0}

    if fxn_str is None or fxn_str == "" or str(fxn_str).lower() in ["nan", "none"]:
        return flags

    terms = [x.strip() for x in str(fxn_str).split(",")]

    for term in terms:
        for cat, keywords in FXN_CATEGORIES.items():
            if term in keywords:
                flags[cat] = 1.0

    return flags


def worker_validate_genome_sequence(row_dict: Mapping[str, Any]) -> dict[str, Any]: # noqa
    """
    Función de validación ligera. Solo realiza la búsqueda en FASTA externo.
    Los inputs ya vienen limpios desde Polars.
    """
    output: dict[str, Any] = {"validated": False, "validation_error": None}

    chrom = row_dict["CHROM"]
    try:
        pos_val = int(row_dict["POS"])
        pos_0 = pos_val - 1
    except (TypeError, ValueError):
        output["validation_error"] = "Invalid POS format"
        return output

    ref = row_dict["REF"]

    if pos_0 < 0:
        output["validation_error"] = "Bad POS (<=0)"
        return output

    if GLOBAL_GENOME:
        if chrom not in GLOBAL_GENOME:
            output["validation_error"] = f"Chr missing: {chrom}"
            return output
        try:
            # Lógica: Si hay REF, validamos que coincida con el genoma
            if len(ref) > 0:
                # Obtener secuencia del Fasta
                # Convertimos a string y mayúsculas para asegurar comparación
                ref_fasta = str(GLOBAL_GENOME[chrom][pos_0 : pos_0 + len(ref)].seq).upper()

                if ref_fasta != ref and "N" not in ref_fasta:
                    output["validation_error"] = f"Ref Mismatch: TSV={ref} vs FASTA={ref_fasta}"
                    return output
        except IndexError:
            output["validation_error"] = "Out of bounds"
            return output
        except Exception as e:
            output["validation_error"] = str(e)
            return output

    output["validated"] = True
    return output

# =============================================================================
#  CLASS 1: GENOMIC GRAPH BUILDER
# =============================================================================


class GenomicGraphBuilderNEXTGEN:
    def __init__(self, fasta_path: Path, gff_path: Path, pgx_dir: Path):
        self.fasta_path = Path(fasta_path) if isinstance(fasta_path, str) else fasta_path
        self.gff_path = gff_path
        self.pgx_dir = pgx_dir

    def run_pipeline(
        self, tsv_input: Path, output_parquet: Path, output_graph_dir: Path
    ):
        print("\n🧬 [GENOMICS] Starting genomic pipeline...")

        # 1. Build Parquet Library
        if self.fasta_path.exists():
            clean_df = self._build_library(tsv_input)
            if clean_df is not None and not clean_df.is_empty():
                clean_df.write_parquet(output_parquet)
                print(f"✅ Library saved at: {output_parquet} ({len(clean_df)} variants)")
                self._generate_graphs(clean_df, output_graph_dir)
                self._organize_files_os_specific(output_graph_dir)
            else:
                print("⚠️ No clean variants dataframe generated.")
        else:
            print(f"❌ Missing reference files (FASTA) in {self.fasta_path.parent}")

    def _build_library(self, tsv_input: Path) -> pl.DataFrame:
        print("   🔹 Indexing genome...")
        global GLOBAL_GENOME #noqa
        # key_function handles fasta headers like ">chr1 extra_info"
        GLOBAL_GENOME = Fasta(str(self.fasta_path), key_function=lambda x: x.split()[0])

        dfs = []
        # 1. Load Main TSV
        if tsv_input.exists():
            print(f"   🔹 Loading TSV: {tsv_input.name}...")
            try:
                t_df = pl.read_csv(tsv_input, separator="\t", infer_schema_length=10000, ignore_errors=True)
                # Ensure core columns exist
                req = ["chr", "start_pos", "Ref_Allele", "Alt_Allele"]
                if not all(c in t_df.columns for c in req):
                    print(f"❌ TSV missing columns. Found: {t_df.columns}")
                else:
                    # Enforce string types for keys to prevent merge issues
                    t_df = t_df.with_columns([pl.col("start_pos").cast(pl.Utf8), pl.col("chr").cast(pl.Utf8)])
                    dfs.append(t_df)
            except Exception as e:
                print(f"❌ Error reading TSV: {e}")

        # 2. Load PGx Data
        pgx_df = self._load_pgx_folder()
        if not pgx_df.is_empty():
            pgx_df = pgx_df.rename({
                "POS": "start_pos", "CHROM": "chr", "REF": "Ref_Allele",
                "ALT": "Alt_Allele", "gene_provided": "gene", "haplotype_label": "snp"
            }).with_columns([
                pl.col("start_pos").cast(pl.Utf8),
                pl.col("chr").cast(pl.Utf8),
                pl.lit("pharmacogenomic_variant").alias("FXN_CLASS")
            ])
            dfs.append(pgx_df)

        if not dfs:
            return pl.DataFrame()
        master_df = pl.concat(dfs, how="diagonal")

        print(f"   ⚡ Validating and Enriching {len(master_df)} variants...")

        # ===========================
        #  VECTORIZED PRE-PROCESSING
        # ===========================
        master_df = master_df.with_columns([
            pl.col("chr").str.strip_chars().alias("chr_clean"),
            pl.col("start_pos").cast(pl.Int64, strict=False).alias("POS"), # Safe cast
            pl.col("Ref_Allele").str.strip_chars().str.to_uppercase().fill_null("").alias("REF"),
            pl.col("Alt_Allele").str.strip_chars().str.to_uppercase().fill_null("").alias("ALT"),
            pl.col("gene").fill_null("Intergenic").alias("gene_context"),
            pl.col("snp").fill_null(pl.lit("")).alias("snp_tmp")
        ])

        master_df = master_df.with_columns(
             pl.col("chr_clean").str.replace_many(["chr", "Chr"], "").replace(GLOBAL_CHROM_MAPPING, default=pl.col("chr_clean")).alias("CHROM")
        )
        master_df = master_df.with_columns(
            pl.when(pl.col("snp_tmp") != "")
            .then(pl.col("snp_tmp"))
            .otherwise(pl.format("var_{}_{}", pl.col("CHROM"), pl.col("POS")))
            .alias("variant_name")
        )

        len_ref = pl.col("REF").str.len_bytes()
        len_alt = pl.col("ALT").str.len_bytes()

        master_df = master_df.with_columns(
            pl.when(pl.col("variant_type").is_not_null())
            .then(pl.col("variant_type"))
            .when(len_ref == len_alt).then(pl.lit("snv"))
            .when(len_ref > len_alt).then(pl.lit("del"))
            .otherwise(pl.lit("ins"))
            .alias("variant_type_calc")
        )

        fxn_exprs = []
        for cat, keywords in FXN_CATEGORIES.items():
            pattern = "|".join([re.escape(k) for k in keywords])
            fxn_exprs.append(
                pl.col("FXN_CLASS").str.contains(pattern).fill_null(False).cast(pl.Float64).alias(f"is_{cat}")
            )

        master_df = master_df.with_columns(fxn_exprs).with_columns(pl.lit(0.5).alias("activity_score"))

        # ====================
        #  ROW-WISE VALIDATION
        # ====================

        master_df = master_df.filter(pl.col("POS").is_not_null())

        validation_schema = pl.Struct({"validated": pl.Boolean, "validation_error": pl.Utf8})

        processed: pl.DataFrame = master_df.with_columns(
            pl.struct(["CHROM", "POS", "REF"])
            .map_elements(worker_validate_genome_sequence, return_dtype=validation_schema)
            .alias("val_result")
        ).unnest("val_result")

        GLOBAL_GENOME = None

        clean: pl.DataFrame = processed.filter(pl.col("validated") == True).unique(  #noqa
            subset=["CHROM", "POS", "REF", "ALT", "gene_context"],
            keep="first"
        )

        # Select final columns to match expected output structure
        return clean.select([
            "CHROM", "POS", "REF", "ALT", "gene_context", "variant_name",
            "variant_type_calc", "activity_score",
            "is_coding", "is_regulatory", "is_splicing", "is_intergenic"
        ]).rename({"variant_type_calc": "variant_type"})

    def _load_pgx_folder(self) -> pl.DataFrame:
        all_variants = []
        if not self.pgx_dir.exists():
            return pl.DataFrame()

        for gene_folder in self.pgx_dir.iterdir():
            if gene_folder.is_dir():
                gene_name = gene_folder.name
                for vcf_file in gene_folder.glob("*.vcf"):
                    haplo_label = self._parse_haplo_name(gene_name, vcf_file.stem)
                    try:
                        vcf_df = pl.read_csv(
                            vcf_file, separator="\t", comment_prefix="#", has_header=False,
                            new_columns=["CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO"],
                            schema_overrides={"CHROM": pl.Utf8, "REF": pl.Utf8, "ALT": pl.Utf8},
                            ignore_errors=True
                        )
                        if vcf_df.is_empty():
                            continue

                        all_variants.append(vcf_df.select([
                            "CHROM", "POS", "REF", "ALT",
                            pl.lit(gene_name).alias("gene_provided"),
                            pl.lit(haplo_label).alias("haplotype_label")
                        ]))
                    except Exception:
                        continue

        if all_variants:
            return pl.concat(all_variants)
        return pl.DataFrame()

    def _generate_graphs(self, library_df: pl.DataFrame, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        genes = library_df["gene_context"].drop_nulls().unique().to_list()
        print(f"   🚀 Generating PyG graphs for {len(genes)} genes...")

        for gene in tqdm(genes, desc="Processing Genes"):
            df_gene = library_df.filter(pl.col("gene_context") == gene)
            variants = df_gene["variant_name"].drop_nulls().unique().to_list()

            for var_name in variants:
                if str(var_name).strip() == "":
                    continue
                df_variant = df_gene.filter(pl.col("variant_name") == var_name)

                G = self._build_nx_graph(df_variant, gene, var_name)
                if G:
                    pyg_data = self._to_pyg(G)
                    safe_var = (str(var_name)
                            .replace(":", "_")
                            .replace("/", "_")
                            .replace("|", "_")
                            .replace("*", "star")
                            )
                    torch.save(pyg_data, output_dir / f"{gene}_{safe_var}.pt")

    def _build_nx_graph(self, df: pl.DataFrame, gene_name, var_name):
        G = nx.MultiDiGraph(name=f"{gene_name}_{var_name}")
        pos_val = df["POS"][0]

        # Fixed Nodes
        G.add_node("start", type="backbone", pos=pos_val - 100)
        G.add_node(f"bb_{pos_val}", type="backbone", pos=pos_val)
        G.add_edge("start", f"bb_{pos_val}", type="backbone_link")

        split, merge = f"split_{pos_val}", f"merge_{pos_val + 1}"
        G.add_node(split, type="split", pos=pos_val)
        G.add_edge(f"bb_{pos_val}", split, type="link")
        G.add_node(merge, type="merge", pos=pos_val + 1)

        # Ref Path
        ref_seq = df["REF"][0]
        ref_n = f"ref_{pos_val}"
        G.add_node(ref_n, type="allele_ref", seq=ref_seq)
        G.add_edge(split, ref_n, attr="ref")
        G.add_edge(ref_n, merge, attr="join")

        # Alt Paths (Polars Iteration)
        for idx, row in enumerate(df.iter_rows(named=True)):
            if row["ALT"] == ref_seq:
                continue
            alt_n = f"alt_{pos_val}_{idx}"
            G.add_node(alt_n, type="allele_alt", seq=row["ALT"], score=row["activity_score"], variant_name=var_name,
                       is_coding=row["is_coding"], is_regulatory=row["is_regulatory"],
                       is_splicing=row["is_splicing"], is_intergenic=row["is_intergenic"])
            G.add_edge(split, alt_n, attr="alt")
            G.add_edge(alt_n, merge, attr="join")

        G.add_node("end", type="backbone_end", pos=pos_val + 100)
        G.add_edge(merge, "end", type="backbone_link")
        return G

    def _to_pyg(self, G: nx.MultiDiGraph) -> Data:
        nodes = list(G.nodes(data=True))
        node_idx = {n: i for i, (n, _) in enumerate(nodes)}
        x_list = []
        variant_name_str = "Unknown"

        for _, d in nodes:
            t = d.get("type", "")
            score = d.get("score", 0.5)
            if "variant_name" in d:
                variant_name_str = d["variant_name"]

            # [Backbone, SplitMerge, Ref, Alt, Score, Coding, Regulatory, Splicing, Intergenic]
            vec = [0.0] * 9
            if "backbone" in t:
                vec[0] = 1.0
            elif "split" in t or "merge" in t:
                vec[1] = 1.0
            elif "ref" in t:
                vec[2] = 1.0
            elif "allele_alt" in t:
                vec[3] = 1.0
                vec[4] = float(score)
                vec[5] = float(d.get("is_coding", 0.0))
                vec[6] = float(d.get("is_regulatory", 0.0))
                vec[7] = float(d.get("is_splicing", 0.0))
                vec[8] = float(d.get("is_intergenic", 0.0))
            x_list.append(vec)

        edge_index = [[node_idx[u], node_idx[v]] for u, v, _ in G.edges(data=True)]
        edge_attr = []
        for _, _, data in G.edges(data=True):
            vec = [0.0, 0.0, 0.0]
            attr = data.get("attr", "")
            if "ref" in attr:
                vec[1] = 1.0
            elif "alt" in attr:
                vec[2] = 1.0
            else:
                vec[0] = 1.0
            edge_attr.append(vec)

        data = Data(
            x=torch.tensor(x_list, dtype=torch.float32),
            edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
            edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
        )
        data.variant_name = variant_name_str
        return data

    def _organize_files_os_specific(self, graph_dir: Path):
        print(f"   📂 Organizing folder structure in {graph_dir}...")
        if os.name == "posix":
            script = """#!/bin/bash
set -e
cd $(dirname "$0")
mkdir -p UGT1A
find . -maxdepth 1 -name "*.pt" -type f | while read filename; do
    base=$(basename "$filename")
    gene_name=$(echo "$base" | cut -d'_' -f1)
    if [[ "$gene_name" =~ ^UGT1A ]]; then mv "$filename" "UGT1A/"; else mkdir -p "$gene_name"; mv "$filename" "$gene_name/"; fi
done
"""
            script_path = graph_dir / "organize_genes.sh"
            with open(script_path, "w") as f:
                f.write(script)
            try:
                os.chmod(script_path, 0o755)
            except: #noqa
                pass
            subprocess.run(["bash", script_path.name], cwd=graph_dir, check=True)

        elif os.name == "nt":
            # Logic preserved from original
            script = """
$ErrorActionPreference = "Continue"
$files = Get-ChildItem -Filter *.pt -File
foreach ($file in $files) {
    $rawGene = $file.BaseName.Split('_')[0]
    $geneName = $rawGene.Split(';')[0]
    if ($geneName -match '^UGT1A([1-9]|10)$') { $targetDirName = "UGT1A" } else { $targetDirName = $geneName }
    $targetPath = Join-Path -Path $PWD -ChildPath $targetDirName
    if (-not (Test-Path -Path $targetPath)) { New-Item -Path $targetPath -ItemType Directory -Force | Out-Null }
    $file | Move-Item -Destination $targetPath -Force
}
"""
            script_path = graph_dir / "organize_genes.ps1"
            with open(script_path, "w") as f:
                f.write(script)
            subprocess.run(["powershell", "-ExecutionPolicy", "Bypass", "-File", script_path.name], cwd=graph_dir, check=True)

    def _parse_haplo_name(self, gene, fname):
        if fname.startswith("rs"):
            return DPYD_RS_MAP.get(fname, fname)
        clean = fname.replace(f"{gene}_", "").replace(gene, "")
        base = clean.split(".")[0] if "." in clean else clean
        return f"*{base}" if base.isdigit() or not base.startswith("*") else base

# =============================================================================
#  CLASS 2: DRUG GRAPH BUILDER
# =============================================================================

class DrugGraphBuilder:
    def __init__(self):
        self.illegal_chars = re.compile(r'[<>:"/\\|?*]')

    def run_pipeline(self, tsv_input: Path, output_dir: Path):
        print("\n💊 [DRUGS] Starting drug pipeline...")
        if not tsv_input.exists():
            print(f"❌ Error: File not found {tsv_input}")
            return
        if smiles_to_graph_complete is None:
            print("❌ Missing `smiles_to_graph_complete`.")
            return

        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            df = pl.read_csv(tsv_input, separator="\t", columns=["cid", "smiles", "cmpd_name_cleaned"], ignore_errors=True)
        except Exception as e:
            print(f"❌ Error reading drug TSV columns: {e}")
            return

        failed, count = [], 0
        for row in tqdm(df.iter_rows(named=True), total=len(df), desc="Creating Drug Graphs"):
            raw_name = str(row["cmpd_name_cleaned"]).strip()
            safe_name = self.illegal_chars.sub("_", raw_name).replace(" ", "_")[:100]
            out_f = output_dir / f"{row['cid']}_{safe_name}.pt"

            if out_f.exists():
                continue
            try:
                graph = smiles_to_graph_complete(str(row["smiles"]).strip())
                if graph:
                    graph.cid, graph.name, graph.smiles = row["cid"], row["cmpd_name_cleaned"], str(row["smiles"]).strip()
                    torch.save(graph, out_f)
                    count += 1
                else:
                    failed.append((row["cid"], raw_name, "Invalid SMILES"))
            except Exception as e:
                failed.append((row["cid"], raw_name, str(e)))

        print(f"✨ Drugs processed: {count}")
        if failed:
            self._log_errors(failed)

    def _log_errors(self, failed_list):
        with open("drug_generation_errors.log", "a", encoding="utf-8") as log:
            log.write("CID\tName\tReason\n")
            for cid, name, reason in failed_list:
                log.write(f"{cid}\t{name}\t{reason}\n")
        print(f"⚠️ {len(failed_list)} errors logged.")

# =============================================================================
#  MAIN EXECUTION
# =============================================================================


def help_DOC():
    print("\n--- HELP ---")
    while True:
        c = input("\n[1] Presentation [2] Input [3] Output [4] Usage [5] All [0] Exit\nChoice: ").strip()
        DOCS = [DOC_PRES, DOC_FILES, DOC_OUTPUTS, DOC_USAGE]
        if c == "0":
            break
        elif c in "1234":
            print(f"\n{'-'*20}\n{DOCS[int(c)-1]}\n{'-'*20}")
        elif c == "5":
            print("\n".join(DOCS))

def args_parser():
    parser = argparse.ArgumentParser(
        description="Unified Genomic and Drug Graph Library Generator",
        add_help=False,
    )
    parser.add_argument(
        "-h",
        "--help",
        action="store_true",
        help="Show interactive documentation menu and exit.",
    )
    parser.add_argument(
        "--verify",
        type=str,
        default=None,
        help="Generate output for a specific gene for testing.",
    )
    return parser.parse_args()


def main(args):
    print("\nStarting Unified Graph Library Generator...\n")
    if not BASE_DIR.exists():
        print(f"❌ Error: {BASE_DIR} not found.")
        return
    for d in [LIB_DIR, GENE_OUT_DIR, DRUG_OUT_DIR]:
        d.mkdir(exist_ok=True)

    if args.verify:
        print(f"\n🔍 Test mode for: {args.verify}")
        gene_builder = GenomicGraphBuilderNEXTGEN(FASTA_FILE, GFF_FILE, PGX_FOLDER)
        gene_builder.run_pipeline(GENE_VAR_TSV, PARQUET_FILE, GENE_OUT_DIR)
        return

    print(f"\n #### ➡️ Processing genome from: {REF_DIR}")
    GenomicGraphBuilderNEXTGEN(FASTA_FILE, GFF_FILE, PGX_FOLDER).run_pipeline(GENE_VAR_TSV, PARQUET_FILE, GENE_OUT_DIR)

    print(f"\n #### ➡️ Processing drugs from: {DRUGS_TSV}")
    DrugGraphBuilder().run_pipeline(DRUGS_TSV, DRUG_OUT_DIR)

    print(f"\n✅ DONE. Library: {LIB_DIR.resolve()}")


if __name__ == "__main__":
    args = args_parser()
    if args.help:
        help_DOC()
    else:
        main(args)
