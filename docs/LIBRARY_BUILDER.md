# Library Builder

> **Module:** `src.data.library`
> **CLI:** `python -m src.data.library`
>
> The library builder pre-computes the drug and variant graphs that the
> training pipeline lazy-loads from disk. Building offline is the project's
> answer to limited training-time compute — the build runs once and the
> trained model reads from disk thereafter.

This document covers runtime behaviour, input contracts, output schemas, and
the CLI surface. For internal architecture, see
[`docs/ARCHITECTURE.md`](ARCHITECTURE.md).

---

## What it produces

```
src/library/
├── drugs/
│   ├── 2244_aspirin.pt           PubChem CID + safe-name → PyG Data
│   ├── 2519_caffeine.pt
│   └── …
├── gene_graphs/
│   ├── CYP2D6/
│   │   ├── CYP2D6_star4.pt       gene_<variant>.pt; "star" = "*"
│   │   ├── CYP2D6_rs1065852.pt
│   │   └── …
│   ├── DPYD/
│   └── …
└── build_manifest.json           Resume tracking (atomic JSON)
```

**Schema** (frozen — must stay in sync with the trained TwoTowerGAT model):

| Graph kind | Node features | Edge features |
| ---------- | ------------- | ------------- |
| Drug       | 25            | 7             |
| Gene       | 9             | 3             |

The dimensions live in `src/data/library/{drugs,genes}.py` as module-level constants and are pinned by `tests/unit/data/test_library_{drugs,genes}.py` so accidental changes fail CI before silently invalidating every trained model.

---

## Inputs

### 1. Variants TSV (`data/snp_data_output.tsv`)

| Column | Type | Required | Notes |
| ------ | ---- | :---: | ----- |
| `chr` | str | ✓ | `1`, `chr1`, `X`, `NC_000001.11` — all normalized internally. |
| `start_pos` | int | ✓ | **1-based** genomic position. |
| `Ref_Allele` | str | ✓ | Reference allele; validated against the FASTA. Mismatch → row skipped + warning. |
| `Alt_Allele` | str | ✓ | Alternate allele. |
| `gene` | str |   | Used to group output graphs into per-gene subdirs. Defaults to "Intergenic" if missing. |
| `snp` | str |   | rsID or variant name; falls back to `var_<chrom>_<pos>` if missing. |
| `variant_type` | str |   | `snv`, `del`, `ins`, `del`, `mnp`, … inferred from REF/ALT lengths if absent. |
| `FXN_CLASS` | str |   | Comma-joined functional terms. Used to set the `is_coding`/`is_regulatory`/`is_splicing`/`is_intergenic` node features. |

### 2. Drugs TSV (`data/drugs_cid.tsv`)

| Column | Type | Required | Notes |
| ------ | ---- | :---: | ----- |
| `cid` | int / str | ✓ | PubChem Compound ID. Becomes the filename prefix. |
| `smiles` | str | ✓ | Canonical SMILES; must parse via RDKit. |
| `cmpd_name_cleaned` | str | ✓ | Display name. Filesystem-unsafe characters get replaced with `_` before being baked into the filename. |

### 3. Reference FASTA (`data/ref_genome/HSapiens_GChr38.fa`)

Indexed (`*.fai`) — needed for the variant validator. If you don't have one yet:

```bash
python -c "from src.genomics.ref_genome import ReferenceGenomeManager; ReferenceGenomeManager().run()"
```

…will download GRCh38 from Ensembl, decompress, and run `samtools faidx` + `bwa index`.

### 4. Per-gene VCF folder (`data/haplotype_variants/`)

```
data/haplotype_variants/
├── CYP2D6/
│   ├── CYP2D6_4.vcf       → labelled  *4
│   ├── CYP2D6_10.vcf      → labelled  *10
│   └── …
├── DPYD/
│   ├── rs3918290.vcf      → resolved via data/dicts/star_alleles.tsv → *2A
│   └── …
└── …
```

Filename → haplotype label rules:

* `GENE_<n>` → `*<n>`        (digits become star alleles)
* `<n>` → `*<n>`             (bare digits)
* `rsXXXX` → looked up in `data/dicts/star_alleles.tsv`. Falls back to the rsID itself when not in the catalog.
* `c.2846A>T` and similar HGVS strings are kept verbatim.

---

## CLI

```bash
$ python -m src.data.library --help

usage: python -m src.data.library [-h]
                                  [--variants-tsv VARIANTS_TSV]
                                  [--drugs-tsv DRUGS_TSV]
                                  [--force] [--only-gene SYMBOL]
                                  [--skip-drugs] [--skip-genes] [--verbose]

Build the offline drug + variant graph library.

  --variants-tsv PATH   default: data/snp_data_output.tsv
  --drugs-tsv PATH      default: data/drugs_cid.tsv
  --force               Overwrite existing .pt files instead of skipping them.
  --only-gene SYMBOL    Build only this gene's variants (verification mode).
  --skip-drugs          Skip the drug pipeline.
  --skip-genes          Skip the gene pipeline.
  --verbose, -v         DEBUG-level logging.
```

Defaults are derived from the project `Settings`, so a zero-argument invocation just works.

### Common workflows

```bash
# Full build from scratch
python -m src.data.library

# Resume after an interrupted run — already-built .pt files are skipped.
python -m src.data.library

# Force-rebuild everything (e.g. after a schema change)
python -m src.data.library --force

# Verify the genomic pipeline against a single gene before launching the full job.
python -m src.data.library --only-gene CYP2D6 --skip-drugs

# Iterate on the drug pipeline without touching gene graphs.
python -m src.data.library --skip-genes
```

### Programmatic API

```python
from src.data.library import LibraryBuilder, LibraryBuildConfig

cfg = LibraryBuildConfig.from_settings(force=False, only_gene="CYP2D6")
summary = LibraryBuilder(cfg).run()

print(f"drugs:  built={summary.drugs_built} skipped={summary.drugs_skipped}")
print(f"genes:  built={summary.genes_built} failed={summary.genes_failed}")
```

For per-piece access:

```python
from src.data.library.drugs import DrugGraphBuilder, smiles_to_graph
from src.data.library.genes import GenomicGraphBuilder
from src.data.library.manifest import BuildManifest
```

---

## Resume support

Every successful save updates `src/library/build_manifest.json`. The next run reads it on startup, so:

* An interrupted build (Ctrl-C, kernel OOM, power loss) just resumes — no rebuilding the first 4000 drugs you already had.
* Failures are recorded too: ``manifest.failed["drug:2244"] = "Invalid SMILES"`` so you can grep the JSON to triage.
* The manifest is written atomically (write-temp-then-rename) so a crash mid-write leaves the previous version intact.

To start fresh: `--force`, or delete the manifest manually.

---

## Troubleshooting

| Symptom | Cause | Fix |
| ------- | ----- | --- |
| `FASTA not found: …/HSapiens_GChr38.fa` | Reference genome not downloaded yet. | Run `ReferenceGenomeManager().run()` (see § Inputs). |
| `Chromosome 'X' not in FASTA` | TSV chromosome label doesn't appear in FASTA index. Most often happens with mixed `chr1` / `1` / `NC_000001.11` conventions. | The validator already normalizes; if you still see this, check `pyfaidx` indexed the FASTA correctly. |
| `REF mismatch: TSV=A vs FASTA=G` | Wrong genome build (GRCh37 vs GRCh38) or the TSV's `start_pos` is 0-based instead of 1-based. | Confirm the build; remember `start_pos` is **1-based**. |
| `Invalid SMILES` for many drugs | TSV contains non-canonical SMILES strings or salts/mixtures. | Pre-canonicalize via RDKit before feeding the builder. Failures land in `<library_root>/build_failures.log`. |
| Drug graphs have 24 features instead of 25 | Old artifacts produced before the schema was finalized. | `python -m src.data.library --force` (or selectively delete `src/library/drugs/` and rerun). |
| Gene subdirs are empty after a run | The PGx folder schema didn't match. | `--verbose` to see which VCFs got parsed; check filenames match the rules in § Inputs. |
| Need only one gene's graphs for testing | | `--only-gene CYP2D6 --skip-drugs` |

---

## What changed vs the pre-refactor builder

The previous single-file 883-line script (`library_creator_polars.py`) lives in
`BACKUPS/dev_Pharmagen_snapshot/` (also reachable via the
`pre-refactor-2026-05` git tag). The Phase-4.5 rewrite:

- Removed module-level globals (`GLOBAL_GENOME`, `GLOBAL_CHROM_MAPPING`, …).
- Replaced hardcoded `BASE_DIR = Path("data")` with `Settings.paths`.
- Replaced bash / PowerShell organize scripts with pure Python
  (`pathlib.Path.rename`).
- Centralised rsID → star-allele lookups in `data/dicts/star_alleles.tsv`.
- Added the `build_manifest.json` resume layer.
- Translated all Spanish comments and docstrings.
- Adopted Pydantic-validated build config; CLI built on `argparse`.
- Added 47 unit tests covering schema dimensions, filename safety, manifest
  atomicity, UGT1A merging, and rsID resolution.
