"""Gene-variant graph builder.

Produces one PyG ``Data`` per (gene, variant) combination:

    backbone --> bb_pos --> split --[ref]--> ref_pos --> merge --> end
                              \\--[alt0]--> alt_pos_0 ---/
                              \\--[alt1]--> alt_pos_1 ---/

Output schema (frozen — must stay in sync with the trained genotype tower):
    Node features (9):  one-hot of {backbone, split/merge, ref, alt}
                        + activity_score (alt only; real per-allele value from
                          the star-allele table, no longer a 0.5 placeholder)
                        + 4 functional flags (is_coding, is_regulatory,
                          is_splicing, is_intergenic)
    Edge features (3):  one-hot of {backbone_link, ref_path, alt_path}
    Global (27):        per-variant ``geno_global_feats`` [1, 27] — PGx allele
                        function (6) + Sequence Ontology consequence (13) + HGVS
                        protein-change physicochemistry (8). Decoupled from node
                        features; see :mod:`src.data.library.geno_func`.

The validator runs in pure Polars + pyfaidx; no module-level globals.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

import networkx as nx
import polars as pl
import torch
from pyfaidx import Fasta
from torch_geometric.data import Data as PyGData
from tqdm.auto import tqdm

from src.data.library.chromosome import CHROM_TO_REFSEQ
from src.data.library.geno_func import GenoFuncProvider
from src.data.library.pgx import load_pgx_folder

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from src.data.library.manifest import BuildManifest

logger = logging.getLogger(__name__)


# Schema dimensions — DO NOT CHANGE without retraining.
GENE_NODE_DIM: int = 9
GENE_EDGE_DIM: int = 3


# Functional consequence categories used to feature-flag variants. Any term
# from the input TSV that contains one of the listed keywords sets the flag.
FXN_CATEGORIES: dict[str, list[str]] = {
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


_VARIANT_NAME_UNSAFE = re.compile(r"[:/|]")


def safe_variant_filename(name: str) -> str:
    """Make a variant identifier safe for use in a filename.

    Replaces ':' '/' '|' with '_' and converts a leading '*' to 'star' so
    the result survives Windows filesystems (which forbid '*'). Round-trips
    via ``GraphIndexBuilder`` which translates 'star' back to '*'.
    """
    cleaned = _VARIANT_NAME_UNSAFE.sub("_", str(name))
    if cleaned.startswith("*"):
        cleaned = "star" + cleaned[1:]
    return cleaned


def _validate_row(
    chrom: str,
    pos: int | None,
    ref: str,
    *,
    fasta: Fasta,
    fasta_keys: set[str],
) -> tuple[bool, str | None, str | None]:
    """Confirm a variant's REF allele matches the reference FASTA.

    Returns ``(validated, used_chrom_key, error)``. ``used_chrom_key`` is the
    actual key found in the FASTA (with or without ``chr`` prefix) so the
    caller can persist it.
    """
    if pos is None or pos <= 0:
        return False, None, "Invalid POS"

    canonical = chrom
    fasta_key: str | None = None
    if canonical in fasta_keys:
        fasta_key = canonical
    elif canonical in CHROM_TO_REFSEQ and CHROM_TO_REFSEQ[canonical] in fasta_keys:
        fasta_key = CHROM_TO_REFSEQ[canonical]
    else:
        return False, None, f"Chromosome {chrom!r} not in FASTA"

    if not ref:
        return True, fasta_key, None

    pos_0 = int(pos) - 1
    try:
        fasta_seq = str(fasta[fasta_key][pos_0 : pos_0 + len(ref)].seq).upper()  # type: ignore[union-attr]
    except (KeyError, IndexError, ValueError) as e:
        return False, fasta_key, f"FASTA fetch failed: {e}"

    if fasta_seq != ref and "N" not in fasta_seq:
        return False, fasta_key, f"REF mismatch: TSV={ref} vs FASTA={fasta_seq}"
    return True, fasta_key, None


class GenomicGraphBuilder:
    """Build per-variant graphs for every (gene, variant) in the input.

    Construction:
        ``GenomicGraphBuilder(fasta_path, pgx_dir)``

    Use:
        ``builder.build(variants_tsv, output_dir, manifest=manifest)``
    """

    def __init__(
        self,
        fasta_path: Path,
        pgx_dir: Path,
        *,
        only_gene: str | None = None,
        force: bool = False,
        func_provider: GenoFuncProvider | None = None,
    ) -> None:
        self.fasta_path = fasta_path
        self.pgx_dir = pgx_dir
        self.only_gene = only_gene
        self.force = force
        # Per-variant PGx-function + pathogenicity profile attached as
        # ``geno_global_feats``. A null provider yields zero vectors, keeping the
        # graph schema complete even for a function-free build.
        self.func = (
            func_provider if func_provider is not None else GenoFuncProvider.null()
        )

    # ----- public -----------------------------------------------------------

    def build(
        self,
        variants_tsv: Path,
        output_dir: Path,
        *,
        manifest: BuildManifest,
    ) -> tuple[int, int]:
        """Build all variant graphs. Returns (built, failed)."""
        if not self.fasta_path.exists():
            msg = f"FASTA not found: {self.fasta_path}"
            raise FileNotFoundError(msg)

        output_dir.mkdir(parents=True, exist_ok=True)

        fasta = Fasta(str(self.fasta_path), key_function=lambda x: x.split()[0])
        try:
            fasta_keys = set(fasta.keys())
            cleaned = self._build_clean_dataframe(variants_tsv, fasta, fasta_keys)
            if cleaned.is_empty():
                logger.warning("No clean variants produced — nothing to build.")
                return 0, 0
            return self._render_graphs(cleaned, output_dir, manifest=manifest)
        finally:
            fasta.close()

    # ----- pipeline ---------------------------------------------------------

    def _build_clean_dataframe(
        self,
        tsv_input: Path,
        fasta: Fasta,
        fasta_keys: set[str],
    ) -> pl.DataFrame:
        """Merge TSV + PGx folder, validate against FASTA, return a clean frame."""
        frames: list[pl.DataFrame] = []
        if tsv_input.exists():
            tsv_df = self._read_variants_tsv(tsv_input)
            if tsv_df is not None:
                frames.append(tsv_df)
        else:
            logger.warning("Variants TSV not found: %s", tsv_input)

        pgx_df = load_pgx_folder(self.pgx_dir)
        if not pgx_df.is_empty():
            pgx_df = pgx_df.rename(
                {
                    "POS": "start_pos",
                    "CHROM": "chr",
                    "REF": "Ref_Allele",
                    "ALT": "Alt_Allele",
                    "gene_provided": "gene",
                    "haplotype_label": "snp",
                }
            ).with_columns(
                [
                    pl.col("start_pos").cast(pl.Utf8),
                    pl.col("chr").cast(pl.Utf8),
                    pl.lit("pharmacogenomic_variant").alias("FXN_CLASS"),
                ]
            )
            frames.append(pgx_df)

        if not frames:
            return pl.DataFrame()

        master = pl.concat(frames, how="diagonal")
        master = self._normalize_columns(master)

        if self.only_gene:
            master = master.filter(pl.col("gene_context") == self.only_gene)
            logger.info(
                "only_gene filter: %d rows retained for %s.",
                len(master),
                self.only_gene,
            )

        master = master.filter(pl.col("POS").is_not_null())
        master = self._validate_against_fasta(master, fasta, fasta_keys)
        clean = master.filter(pl.col("validated") == True).unique(  # noqa: E712
            subset=["CHROM", "POS", "REF", "ALT", "gene_context"], keep="first"
        )
        logger.info(
            "Validation: %d/%d variants kept after FASTA check.",
            len(clean),
            len(master),
        )

        return clean.select(
            [
                "CHROM",
                "POS",
                "REF",
                "ALT",
                "gene_context",
                "variant_name",
                "variant_type_calc",
                "activity_score",
                "FXN_CLASS",
                "is_coding",
                "is_regulatory",
                "is_splicing",
                "is_intergenic",
            ]
        ).rename({"variant_type_calc": "variant_type"})

    def _read_variants_tsv(self, path: Path) -> pl.DataFrame | None:
        try:
            df = pl.read_csv(
                path, separator="\t", infer_schema_length=10_000, ignore_errors=True
            )
        except Exception as e:  # noqa: BLE001
            logger.error("Could not read variants TSV %s: %s", path, e)
            return None
        required = ["chr", "start_pos", "Ref_Allele", "Alt_Allele"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            logger.error("Variants TSV missing columns: %s", missing)
            return None
        return df.with_columns(
            [pl.col("start_pos").cast(pl.Utf8), pl.col("chr").cast(pl.Utf8)]
        )

    @staticmethod
    def _normalize_columns(df: pl.DataFrame) -> pl.DataFrame:
        """Polars-vectorized column normalization."""
        df = df.with_columns(
            [
                pl.col("chr").str.strip_chars().alias("chr_clean"),
                pl.col("start_pos").cast(pl.Int64, strict=False).alias("POS"),
                pl.col("Ref_Allele")
                .str.strip_chars()
                .str.to_uppercase()
                .fill_null("")
                .alias("REF"),
                pl.col("Alt_Allele")
                .str.strip_chars()
                .str.to_uppercase()
                .fill_null("")
                .alias("ALT"),
                pl.col("gene").fill_null("Intergenic").alias("gene_context"),
                pl.col("snp").fill_null(pl.lit("")).alias("snp_tmp"),
            ]
        )
        df = df.with_columns(
            pl.col("chr_clean")
            .str.replace_many(["chr", "Chr"], "")
            .replace(CHROM_TO_REFSEQ, default=pl.col("chr_clean"))
            .alias("CHROM")
        )
        df = df.with_columns(
            pl.when(pl.col("snp_tmp") != "")
            .then(pl.col("snp_tmp"))
            .otherwise(pl.format("var_{}_{}", pl.col("CHROM"), pl.col("POS")))
            .alias("variant_name")
        )

        len_ref = pl.col("REF").str.len_bytes()
        len_alt = pl.col("ALT").str.len_bytes()

        existing_type = (
            pl.col("variant_type") if "variant_type" in df.columns else pl.lit(None)
        )
        df = df.with_columns(
            pl.when(existing_type.is_not_null())
            .then(existing_type)
            .when(len_ref == len_alt)
            .then(pl.lit("snv"))
            .when(len_ref > len_alt)
            .then(pl.lit("del"))
            .otherwise(pl.lit("ins"))
            .alias("variant_type_calc")
        )

        fxn_exprs = []
        for cat, keywords in FXN_CATEGORIES.items():
            pattern = "|".join(re.escape(k) for k in keywords)
            fxn_exprs.append(
                pl.col("FXN_CLASS")
                .str.contains(pattern)
                .fill_null(False)
                .cast(pl.Float64)
                .alias(f"is_{cat}")
            )
        return df.with_columns(fxn_exprs).with_columns(
            pl.lit(0.5).alias("activity_score")
        )

    @staticmethod
    def _validate_against_fasta(
        df: pl.DataFrame, fasta: Fasta, fasta_keys: set[str]
    ) -> pl.DataFrame:
        validation_schema = pl.Struct(
            {"validated": pl.Boolean, "validation_error": pl.Utf8}
        )

        def _worker(row: Mapping[str, Any]) -> dict[str, Any]:
            ok, _, err = _validate_row(
                row["CHROM"],
                row["POS"],
                row["REF"],
                fasta=fasta,
                fasta_keys=fasta_keys,
            )
            return {"validated": ok, "validation_error": err}

        return df.with_columns(
            pl.struct(["CHROM", "POS", "REF"])
            .map_elements(_worker, return_dtype=validation_schema)
            .alias("val_result")
        ).unnest("val_result")

    def _render_graphs(
        self,
        library_df: pl.DataFrame,
        output_dir: Path,
        *,
        manifest: BuildManifest,
    ) -> tuple[int, int]:
        genes = sorted(library_df["gene_context"].drop_nulls().unique().to_list())
        logger.info("Rendering graphs for %d genes.", len(genes))

        built = failed = 0
        for gene in tqdm(genes, desc="Genes"):
            gene_dir = output_dir / gene
            gene_dir.mkdir(exist_ok=True)
            df_gene = library_df.filter(pl.col("gene_context") == gene)
            variants = df_gene["variant_name"].drop_nulls().unique().to_list()

            for var_name in variants:
                if not str(var_name).strip():
                    continue
                if manifest.has_gene(gene, str(var_name)) and not self.force:
                    continue
                df_variant = df_gene.filter(pl.col("variant_name") == var_name)
                try:
                    row0 = df_variant.row(0, named=True)
                    # Real per-allele activity score (replaces the 0.5 placeholder).
                    activity = self.func.activity_for(str(var_name))
                    graph_nx = self._build_nx_graph(
                        df_variant, gene, str(var_name), activity_override=activity
                    )
                    pyg = self._to_pyg(graph_nx, str(var_name))
                    pyg.geno_global_feats = self.func.vector_for(
                        str(var_name), row0.get("FXN_CLASS")
                    )
                except Exception as e:  # noqa: BLE001
                    failed += 1
                    manifest.mark_gene_failed(
                        gene, str(var_name), f"{type(e).__name__}: {e}"
                    )
                    logger.warning("Gene %s variant %s failed: %s", gene, var_name, e)
                    continue

                safe_var = safe_variant_filename(var_name)
                out_path = gene_dir / f"{gene}_{safe_var}.pt"
                try:
                    torch.save(pyg, out_path)
                except OSError as e:
                    failed += 1
                    manifest.mark_gene_failed(gene, str(var_name), f"save error: {e}")
                    continue
                built += 1
                manifest.mark_gene_done(gene, str(var_name))

        manifest.stats.genes_built = built
        manifest.stats.genes_failed = failed
        manifest.stats.variants_total = built + failed
        logger.info("Gene builder done: %d built, %d failed.", built, failed)
        return built, failed

    # ----- graph construction -----------------------------------------------

    @staticmethod
    def _build_nx_graph(
        df: pl.DataFrame,
        gene: str,
        var_name: str,
        *,
        activity_override: float | None = None,
    ) -> nx.MultiDiGraph:
        g = nx.MultiDiGraph(name=f"{gene}_{var_name}")
        pos_val = df["POS"][0]

        g.add_node("start", type="backbone", pos=pos_val - 100)
        g.add_node(f"bb_{pos_val}", type="backbone", pos=pos_val)
        g.add_edge("start", f"bb_{pos_val}", type="backbone_link")

        split, merge = f"split_{pos_val}", f"merge_{pos_val + 1}"
        g.add_node(split, type="split", pos=pos_val)
        g.add_edge(f"bb_{pos_val}", split, type="link")
        g.add_node(merge, type="merge", pos=pos_val + 1)

        ref_seq = df["REF"][0]
        ref_n = f"ref_{pos_val}"
        g.add_node(ref_n, type="allele_ref", seq=ref_seq)
        g.add_edge(split, ref_n, attr="ref")
        g.add_edge(ref_n, merge, attr="join")

        for idx, row in enumerate(df.iter_rows(named=True)):
            if row["ALT"] == ref_seq:
                continue
            alt_n = f"alt_{pos_val}_{idx}"
            score = (
                activity_override
                if activity_override is not None
                else row["activity_score"]
            )
            g.add_node(
                alt_n,
                type="allele_alt",
                seq=row["ALT"],
                score=score,
                variant_name=var_name,
                is_coding=row["is_coding"],
                is_regulatory=row["is_regulatory"],
                is_splicing=row["is_splicing"],
                is_intergenic=row["is_intergenic"],
            )
            g.add_edge(split, alt_n, attr="alt")
            g.add_edge(alt_n, merge, attr="join")

        g.add_node("end", type="backbone_end", pos=pos_val + 100)
        g.add_edge(merge, "end", type="backbone_link")
        return g

    @staticmethod
    def _to_pyg(g: nx.MultiDiGraph, variant_name: str) -> PyGData:
        nodes = list(g.nodes(data=True))
        node_idx = {n: i for i, (n, _) in enumerate(nodes)}
        x_list: list[list[float]] = []
        captured_variant_name = variant_name

        for _, data in nodes:
            t = data.get("type", "")
            score = data.get("score", 0.5)
            if "variant_name" in data:
                captured_variant_name = data["variant_name"]

            vec = [0.0] * GENE_NODE_DIM
            if "backbone" in t:
                vec[0] = 1.0
            elif "split" in t or "merge" in t:
                vec[1] = 1.0
            elif "ref" in t:
                vec[2] = 1.0
            elif "allele_alt" in t:
                vec[3] = 1.0
                vec[4] = float(score)
                vec[5] = float(data.get("is_coding", 0.0))
                vec[6] = float(data.get("is_regulatory", 0.0))
                vec[7] = float(data.get("is_splicing", 0.0))
                vec[8] = float(data.get("is_intergenic", 0.0))
            x_list.append(vec)

        edge_index_list = [[node_idx[u], node_idx[v]] for u, v, _ in g.edges(data=True)]
        edge_attr: list[list[float]] = []
        for _, _, edge_data in g.edges(data=True):
            vec = [0.0, 0.0, 0.0]
            attr = edge_data.get("attr", "")
            if "ref" in attr:
                vec[1] = 1.0
            elif "alt" in attr:
                vec[2] = 1.0
            else:
                vec[0] = 1.0
            edge_attr.append(vec)

        out = PyGData(
            x=torch.tensor(x_list, dtype=torch.float32),
            edge_index=torch.tensor(edge_index_list, dtype=torch.long).t().contiguous(),
            edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
        )
        out.variant_name = captured_variant_name
        return out
