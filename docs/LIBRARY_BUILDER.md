# Library Builder

`python -m src.data.library` builds everything the training pipeline loads that
is not a training row: the drug graphs, their ADMET profiles and the HGVS
variant catalog. Drugs are built by **Modephere** (`../Modephere`,
`modephere.library`); the catalog is assembled from **Modephere's
pharmacogenomic exports**. dev_Pgen adds what is its own: the ADMET-AI profile
of each compound and the star-allele function table.

Design and decisions: `docs/superpowers/specs/2026-11-07-modephere-library-design.md`
and, for drug entities and the pharmacophore branch,
`docs/superpowers/specs/2026-11-07-drug-sets-and-pharmacophores-design.md`.

## What it produces

Under `Settings.paths.library` (`data/library/`):

| Output | Written by | Content |
|---|---|---|
| `compounds/<InChIKey>.safetensors` | Modephere `LibraryBuilder` | every representation of one drug (molecule + pharmacophore graphs, conformers, 3D pharmacophores) |
| `compounds.parquet` | Modephere | registry: InChIKey, ChEMBL ID, SMILES forms, toolkit version |
| `aliases.parquet` | Modephere (0.1.3) | ChEMBL ID -> InChIKey of every query whose compound is stored, incl. queries built once for another ID (nabiximols) |
| `conformers.parquet`, `failures.parquet`, `build_manifest.json` | Modephere | conformer rows, latest build's failures (and `stored_2d_only` downgrades), build options + versions |
| `admet_profile.parquet` | `src/data/library/admet.py` | 41 ADMET-AI endpoints per InChIKey (`admet_feats`) |
| `hgvs_catalog.pt` | `src/data/genotype/catalog.py` | label / annotation → HGVS links + per-variant gene, SO terms, protein HGVS |
| `pgx/*.parquet` | Modephere `pgx export` (`--refresh-pgx`) | `annotation_rows` (drugs with `atc_codes` / `components`), `atc_members`, `allele_variants`, `annotation_variants`, `annotation_genotypes`, `variants`, `haplotypes` |

The drug tower reads a compound through `modephere.featurize.molecule_data`
(2D atom graph: `x` 61, `edge_attr` 18, `global_feats` 1038, `z`) plus
`admet_feats` 41 — Modephere's `MOLECULE_GRAPH_V1` layout — and, with
`use_pharmacophore`, `modephere.featurize.pharmacophore_data` (points 11,
point pairs 13, global 258: `PHARMACOPHORE_GRAPH_V1`) with the points' 3D
positions in one conformer (`feature_positions`); see `GraphDims` and
`DrugData` in `src/data/cache.py`. A compound stored 2D-only (its conformer
stage failed) reads with zero positions and `has_3d = 0`.

## Identity

Drugs are keyed by their **ChEMBL parent ID** across dev_Pgen (training rows'
`drugs_chembl`, the API's `drug_chembl_id` / `drug_ids`, the DDI artifact). The
library keys compounds by InChIKey; `src.data.library.bridge.chembl_index`
joins the two from `compounds.parquet` and `aliases.parquet`. Which drugs are
built — and which ChEMBL ID each annotation's drugs have — comes from
Modephere's `annotation_rows` export.

A training row's `drugs_chembl` is a `|`-joined set of **drug entities**:

| Annotation drug | Entity keys | Read as |
|---|---|---|
| a drug with a ChEMBL ID | `CHEMBL25` | that molecule |
| a combination (`sulfamethoxazole / trimethoprim`, FOLFOX) | one `CHEMBL…` per component | the components, attention-pooled with the row's other entities |
| a drug class (`opioids`, `corticosteroids`) | `ATC:N02A` (`ATC:A01AC,H02A` for several codes) | the mean of its member molecules (`atc_members`, members built; up to 64 sampled per item in training) |

Rows with a class entity have `is_class_row = true`: they train only, never
in val / test, and are dropped from a fold's training set when they would leak
(`src/evaluation/class_rows.py`).

## Steps

1. `--refresh-pgx` (optional): export Modephere's pgx tables from its Neo4j
   (`Settings.modephere.kg`) into `Settings.modephere.pgx_export`.
2. Drugs: the ChEMBL IDs the rows read — single drugs, combination
   components and the members of every drug class — go through Modephere's
   `LibraryBuilder`, offline, against the newest staged ChEMBL release under
   `Settings.modephere.chembl`. Stored compounds are skipped unless `--force`,
   except compounds built with another RDKit than the running one, which are
   always rebuilt (a forced Modephere run of their IDs). The log reports built
   / stale-rebuilt / 2D-only / failed per category, and one line per class
   whose members are not all built (`class ATC:N02A: 25 of 28 members built`).
   The first build after this change adds ~850 class members and rebuilds the
   ~1,100 stale ones: expect 3–4 h.
3. ADMET: every stored compound the ADMET table lacks is predicted (one
   ADMET-AI batch) on its `working_smiles`; `--force-admet` recomputes all.
4. Genes: `hgvs_catalog.pt` from the pgx export and `data/dicts/star_alleles.tsv`.

## Configuration

All of it is `Settings.modephere` (`[modephere]` in
`src/config/data/settings.toml`, overridden by `PHARMAGEN_MODEPHERE__*`).
`src/data/library/bridge.py` hands these to Modephere explicitly; Modephere's
own settings and `MODEPHERE_*` variables are never read from dev_Pgen.

```toml
[modephere]
library = "data/library"           # LibraryStore root
cache = "data/cache/modephere"     # PubChem raw cache (drugs ChEMBL cannot resolve)
chembl = "data/chembl"             # staged ChEMBL releases (chembl_<N>/)
pgx_export = "data/library/pgx"    # Modephere pgx export tables
logs = "logs/library"              # build failure / saturation reports
max_heavy_atoms = 300              # standardizer cap (exenatide 295, liraglutide 266)

[modephere.kg]                     # read by --refresh-pgx
uri = "bolt://localhost:7688"
user = "neo4j"
database = "neo4j"                 # password: PHARMAGEN_MODEPHERE__KG__PASSWORD
```

## CLI

```text
python -m src.data.library [--refresh-pgx] [--force] [--skip-drugs]
                           [--skip-admet] [--force-admet] [--skip-genes] [-v]
```

```bash
python -m src.data.library --refresh-pgx               # everything, fresh exports
python -m src.data.library --skip-drugs --skip-admet   # catalog only
python -m src.data.library --skip-genes                # drugs + ADMET only
```

Exit code 0 when every step that ran produced its output (individual drug
failures — unsupported chemistry, embedding failures — are expected and listed
in `logs/library/build_failures_<date>.log`); 2 when no drug could be built or
the catalog failed.

### Programmatic API

```python
from src.data.library import LibraryBuilder, LibraryBuildConfig

summary = LibraryBuilder(LibraryBuildConfig.from_settings(refresh_pgx=True)).run()

from src.data.cache import GraphCache
cache = GraphCache.from_settings()           # ChEMBL ID -> DrugData (2D + pharmacophore)
graph = cache.get_drug("CHEMBL1771")         # a sampled conformer's point positions
kind, molecules = cache.get_entity("ATC:N02A")   # a class: its built members
fixed = cache.with_conformer(0)              # lowest-energy conformer (inference)
```

## What changed (2026-11)

dev_Pgen's own drug and pharmacophore graph builders (`drugs/*.pt`,
`pharmacophores/*.pt`), the SNP-table / PharmVar / VCF ingest and the CID drug
key were replaced by Modephere. The catalog no longer reads
`data/snp_data_output.tsv`, `data/haplotype_variants/` or
`data/library/kg/{allele,annotation}_variants.parquet`; rsIDs known only to the
SNP table are no longer catalog labels.
