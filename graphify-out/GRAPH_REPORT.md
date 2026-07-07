# Graph Report - dev_Pharmagen  (2026-07-07)

## Corpus Check
- 109 files · ~63,513 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1396 nodes · 2685 edges · 81 communities (70 shown, 11 thin omitted)
- Extraction: 94% EXTRACTED · 6% INFERRED · 0% AMBIGUOUS · INFERRED: 149 edges (avg confidence: 0.51)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `43927710`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_Graph Cache Layer|Graph Cache Layer]]
- [[_COMMUNITY_Cleaning & Geno Keys|Cleaning & Geno Keys]]
- [[_COMMUNITY_ADMET Provider|ADMET Provider]]
- [[_COMMUNITY_Library Build Design Notes|Library Build Design Notes]]
- [[_COMMUNITY_Engine & Collator|Engine & Collator]]
- [[_COMMUNITY_Setup Orchestration|Setup Orchestration]]
- [[_COMMUNITY_Geno Library Builder|Geno Library Builder]]
- [[_COMMUNITY_Gene Graph & Consequence|Gene Graph & Consequence]]
- [[_COMMUNITY_GATv2 Two-Tower Model|GATv2 Two-Tower Model]]
- [[_COMMUNITY_Target Encoders|Target Encoders]]
- [[_COMMUNITY_Genomic HGVS Build|Genomic HGVS Build]]
- [[_COMMUNITY_GFF Annotation Reader|GFF Annotation Reader]]
- [[_COMMUNITY_Loss & Optimizer Factories|Loss & Optimizer Factories]]
- [[_COMMUNITY_Gene & Star-Allele Models|Gene & Star-Allele Models]]
- [[_COMMUNITY_Raw HGVS Adapter|Raw HGVS Adapter]]
- [[_COMMUNITY_API Model Catalog|API Model Catalog]]
- [[_COMMUNITY_NGS Pipeline|NGS Pipeline]]
- [[_COMMUNITY_Library Builder Orchestrator|Library Builder Orchestrator]]
- [[_COMMUNITY_Training Loop Internals|Training Loop Internals]]
- [[_COMMUNITY_Exception Hierarchy|Exception Hierarchy]]
- [[_COMMUNITY_FastAPI App & DI|FastAPI App & DI]]
- [[_COMMUNITY_Double-Graph Visualization|Double-Graph Visualization]]
- [[_COMMUNITY_Build Manifest|Build Manifest]]
- [[_COMMUNITY_Config Hyperparam Specs|Config Hyperparam Specs]]
- [[_COMMUNITY_Optuna & Standard Trainers|Optuna & Standard Trainers]]
- [[_COMMUNITY_Drug Atom Features|Drug Atom Features]]
- [[_COMMUNITY_DoubleTower Dataset|DoubleTower Dataset]]
- [[_COMMUNITY_Graph Visualization Script|Graph Visualization Script]]
- [[_COMMUNITY_HGVS Parser & Protein Pos|HGVS Parser & Protein Pos]]
- [[_COMMUNITY_Pydantic Settings|Pydantic Settings]]
- [[_COMMUNITY_HGVS Domain Models|HGVS Domain Models]]
- [[_COMMUNITY_Checkpoint Manager|Checkpoint Manager]]
- [[_COMMUNITY_Variant Domain Models|Variant Domain Models]]
- [[_COMMUNITY_CLI Entry Point|CLI Entry Point]]
- [[_COMMUNITY_dbSNP Domain & Fetch|dbSNP Domain & Fetch]]
- [[_COMMUNITY_Chromosome Mapping|Chromosome Mapping]]
- [[_COMMUNITY_Drug Record Loading|Drug Record Loading]]
- [[_COMMUNITY_Geno Functional Profile|Geno Functional Profile]]
- [[_COMMUNITY_Protein-Change Featurizer|Protein-Change Featurizer]]
- [[_COMMUNITY_Tabular Loader & Predictor|Tabular Loader & Predictor]]
- [[_COMMUNITY_Drug Domain Model|Drug Domain Model]]
- [[_COMMUNITY_Genomic Graph Builder|Genomic Graph Builder]]
- [[_COMMUNITY_Star-Allele Loading Tests|Star-Allele Loading Tests]]
- [[_COMMUNITY_SO Consequence Featurizer|SO Consequence Featurizer]]
- [[_COMMUNITY_Drug Graph Builder|Drug Graph Builder]]
- [[_COMMUNITY_Gene-Variant Graph & PGx|Gene-Variant Graph & PGx]]
- [[_COMMUNITY_CSVTSV Loader Tests|CSV/TSV Loader Tests]]
- [[_COMMUNITY_Star-Allele Domain Model|Star-Allele Domain Model]]
- [[_COMMUNITY_Prediction RequestResponse|Prediction Request/Response]]
- [[_COMMUNITY_SMILES-to-Graph Bonds|SMILES-to-Graph Bonds]]
- [[_COMMUNITY_Geno Library Assembly|Geno Library Assembly]]
- [[_COMMUNITY_VCF Variant Ingestion|VCF Variant Ingestion]]
- [[_COMMUNITY_Interactive Predict CLI|Interactive Predict CLI]]
- [[_COMMUNITY_Core Validators|Core Validators]]
- [[_COMMUNITY_GenoFunc Provider|GenoFunc Provider]]
- [[_COMMUNITY_Variant Model Tests|Variant Model Tests]]
- [[_COMMUNITY_Reference Genome Handling|Reference Genome Handling]]
- [[_COMMUNITY_ADMET-AI Eval Script|ADMET-AI Eval Script]]
- [[_COMMUNITY_Feature Saturation Tracking|Feature Saturation Tracking]]
- [[_COMMUNITY_Console IO Helpers|Console IO Helpers]]
- [[_COMMUNITY_Medication Matrix Script|Medication Matrix Script]]
- [[_COMMUNITY_Genotype Resolver|Genotype Resolver]]
- [[_COMMUNITY_ConsoleIO Widget|ConsoleIO Widget]]
- [[_COMMUNITY_Predictor Inference Internals|Predictor Inference Internals]]
- [[_COMMUNITY_Integration Smoke Tests|Integration Smoke Tests]]
- [[_COMMUNITY_Multi-Task Uncertainty Loss|Multi-Task Uncertainty Loss]]
- [[_COMMUNITY_Project Design Principles|Project Design Principles]]

## God Nodes (most connected - your core abstractions)
1. `get_settings()` - 50 edges
2. `StarAllele` - 44 edges
3. `Gene` - 29 edges
4. `Variant` - 28 edges
5. `PGenPredictor` - 28 edges
6. `TrainingLoop` - 25 edges
7. `train_pipeline()` - 25 edges
8. `PredictionResult` - 23 edges
9. `Position` - 23 edges
10. `StarAlleleMap` - 21 edges

## Surprising Connections (you probably didn't know these)
- `test_settings_paths_resolves()` --calls--> `get_settings()`  [EXTRACTED]
  tests/integration/test_pipeline_smoke.py → src/config/settings.py
- `TestPredictionRequest` --uses--> `StarAllele`  [INFERRED]
  tests/unit/domain/test_prediction.py → src/domain/gene.py
- `TestPredictionResult` --uses--> `StarAllele`  [INFERRED]
  tests/unit/domain/test_prediction.py → src/domain/gene.py
- `TestTargetPrediction` --uses--> `StarAllele`  [INFERRED]
  tests/unit/domain/test_prediction.py → src/domain/gene.py
- `test_resolve_device_returns_torch_device()` --calls--> `resolve_device()`  [EXTRACTED]
  tests/integration/test_pipeline_smoke.py → src/model/engine/base.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Two-Tower GNN Architecture** — gatv2tower, doubletowerdataset, pgenpredictor [INFERRED 0.90]

## Communities (81 total, 11 thin omitted)

### Community 0 - "Graph Cache Layer"
Cohesion: 0.06
Nodes (52): Element, build_from_accession(), DbSnpGene, DbSnpSummary, dbSNP E-utilities ESummary domain models.  Structured shape of a single ``<Docum, One ``<DocumentSummary>`` record, reduced to the PGx-relevant fields., All SPDI alleles convertible to project ``Variant`` models., Infer the genome build from a RefSeq chromosome accession version.      Defaults (+44 more)

### Community 1 - "Cleaning & Geno Keys"
Cohesion: 0.06
Nodes (31): AlleleFunction, Gene, Gene and star-allele models.  Star-allele naming follows the PharmGKB/PharmVar c, The canonical 'GENE*allele' string., Construct from a ``GENE*allele`` string., Functional classification of a star allele.      Aligned with the CPIC/PharmGKB, A gene identified by its HGNC symbol, optionally an Ensembl ID., A pharmacogenomic star allele (e.g. CYP2D6*4).      The combined ``GENE*allele`` (+23 more)

### Community 2 - "ADMET Provider"
Cohesion: 0.05
Nodes (60): Enum, BioinformaticsError, Genomic or VCF data is malformed or build-mismatched., HGVSVariant, MolecularType, NucleotideChange, ProteinChange, ProteinPosition (+52 more)

### Community 3 - "Library Build Design Notes"
Cohesion: 0.07
Nodes (55): BaseModel, default_model_name(), The model the API serves by default. For now, the first model in the     catalog, health(), RegistryDep, SettingsDep, Liveness and readiness probes., Liveness probe — always returns 200 if the process is up. (+47 more)

### Community 4 - "Engine & Collator"
Cohesion: 0.06
Nodes (37): BaseSettings, FastAPI, get_registry(), PredictorRegistry, FastAPI dependency-injection helpers.  The predictor is loaded lazily — the API, Per-process cache of loaded predictors keyed by model name.      The PGenPredict, Return the predictor for ``model_name``, loading on first request.          Rais, Pharmagen FastAPI service.  Public entrypoint:     >>> from src.api.main import (+29 more)

### Community 5 - "Setup Orchestration"
Cohesion: 0.06
Nodes (32): GenotypeResolver, LabelEncoder, MultiLabelBinarizer, PGenPredictor, Any, DataFrame, DataLoader, DoubleTowerDataset (+24 more)

### Community 6 - "Geno Library Builder"
Cohesion: 0.09
Nodes (24): Public API for project configuration.  Prefer this entry point in new code:, _BaseSpec, CategoricalSpec, FloatSpec, get_available_models(), get_model_config(), IntSpec, _load_models_toml() (+16 more)

### Community 7 - "Gene Graph & Consequence"
Cohesion: 0.09
Nodes (25): Pharmagen Project, Sequential, GATv2Tower, PharmagenTwoTower, Data, Tensor, GATv2-based graph encoding tower., Project a per-molecule descriptor vector into the drug embedding space. (+17 more)

### Community 8 - "GATv2 Two-Tower Model"
Cohesion: 0.09
Nodes (20): CondaStrategy, ConfigManager, DirectoryManager, EnvManagerType, EnvStrategy, MambaStrategy, Path, Creates the standard directory tree if it doesn't exist. (+12 more)

### Community 9 - "Target Encoders"
Cohesion: 0.09
Nodes (24): Fasta, _assemble(), gene_sequence(), GeneAnnotation, GeneModel, _GffAccumulator, _parse_attrs(), Path (+16 more)

### Community 10 - "Genomic HGVS Build"
Cohesion: 0.06
Nodes (34): Additional Resources, Avoid Repeated Lookups, Bare Except, Beautiful is better than ugly, Code Quality Guidelines, Code Style Guidelines, Common Anti-Patterns to Avoid, Dependency Inversion Principle (DIP) (+26 more)

### Community 11 - "GFF Annotation Reader"
Cohesion: 0.15
Nodes (15): CompletedProcess, BioToolExecutor, MappingAlignmentAnalysis, ProcessRawGenome, Path, Read alignment + dedup + BAM QC. Tools: BWA, samtools, Picard, Qualimap., Align with BWA-MEM and sort with samtools.          Pipes the BWA stdout into sa, Variant calling. Tools: Freebayes, vcftools. (+7 more)

### Community 12 - "Loss & Optimizer Factories"
Cohesion: 0.11
Nodes (26): Exception, ConvergenceError, DataError, EncoderError, GraphError, HardwareError, OptimizationError, PharmagenException (+18 more)

### Community 13 - "Gene & Star-Allele Models"
Cohesion: 0.15
Nodes (18): _ComponentFactory, LossFactory, OptimizerFactory, Component factories for the Pharmagen training pipeline.  Provides :class:`Optim, Registry base: maps string keys to constructors., SchedulerFactory, AdaptiveFocalLoss, AsymmetricLoss (+10 more)

### Community 14 - "Raw HGVS Adapter"
Cohesion: 0.11
Nodes (16): Any, DataLoader, device, Module, MutableSequence, Optimizer, Tensor, One epoch of training with mixed-precision + gradient clipping. (+8 more)

### Community 15 - "API Model Catalog"
Cohesion: 0.11
Nodes (26): build_reference_graph(), infer_edge_type_genomic(), infer_node_type_genomic(), plot_all_variants(), plot_gene_subgraph(), plot_genomic_graph(), plot_molecular_graph(), plot_molecule_reconstructed() (+18 more)

### Community 16 - "NGS Pipeline"
Cohesion: 0.14
Nodes (15): MonkeyPatch, OptunaTrialTrainer, Trainer used by the Optuna tuner — minimal logging, no checkpointing., The full-featured trainer used by the production pipeline., StandardTrainer, Any, device, Module (+7 more)

### Community 17 - "Library Builder Orchestrator"
Cohesion: 0.16
Nodes (21): _batch_predict_flow(), _get_predictor_class(), _interactive_predict_loop(), main_menu_loop(), Interactive Prediction Workflow with cached import., Single prediction interactive loop., Batch prediction from file., Advanced analysis workflow (placeholder). (+13 more)

### Community 18 - "Training Loop Internals"
Cohesion: 0.12
Nodes (22): build_reference_graph(), infer_edge_type_genomic(), infer_node_type_genomic(), plot_all_variants(), plot_gene_subgraph(), plot_genomic_graph(), plot_molecular_graph(), Data (+14 more)

### Community 19 - "Exception Hierarchy"
Cohesion: 0.15
Nodes (18): DoubleTowerCollater, ConfigurationError, ModelError, Invalid or missing configuration (hyperparameters, model settings)., Model creation or loading failure (bad architecture, missing weights)., ConfigValidator, Checks that configuration objects satisfy runtime constraints., build_gnn_model() (+10 more)

### Community 20 - "FastAPI App & DI"
Cohesion: 0.14
Nodes (21): build_train_val_loaders(), build_two_tower_datasets(), infer_dataset_dimensions(), DataFrame, DataLoader, DoubleTowerDataset, Split ``df`` into (train, val), stratifying on ``_stratify`` if present., Construct paired train/val DoubleTowerDatasets.      The val dataset reuses the (+13 more)

### Community 21 - "Double-Graph Visualization"
Cohesion: 0.19
Nodes (9): CheckpointManager, load_model_only(), Any, Module, Optimizer, Path, Training checkpoint management.  :class:`CheckpointManager` handles atomic saves, Manages training checkpoints with atomic saving and automatic cleanup. (+1 more)

### Community 22 - "Build Manifest"
Cohesion: 0.18
Nodes (15): ArgumentParser, arguments_parser(), main(), Execute training in headless mode., Execute prediction in headless mode., Main entry point for Pharmagen CLI.      Args:         args: Parsed command line, _run_headless_prediction(), _run_headless_training() (+7 more)

### Community 23 - "Config Hyperparam Specs"
Cohesion: 0.10
Nodes (19): 1. What matters for our case, 2. Substrate ≠ inhibitor (read this first), 3. Tool comparison, 4. Per-tool notes, 5. Recommendation for Pharmagen, 6. Integration notes (when we build it), 7. Starting points (verify URLs before use), 8. Evaluation on the real catalog (2026-06-11) (+11 more)

### Community 24 - "Optuna & Standard Trainers"
Cohesion: 0.10
Nodes (19): 10. Cross-cutting concerns, 11. Artifact & data-flow summary, 1. System overview, 2. Entry points & dispatch, 3. The shared engine backbone, 4.1 Drug graphs (SMILES → molecular graph), 4.2 Gene-variant graphs (VCF/TSV → topology graph), 4. Workflow — Library building (offline) (+11 more)

### Community 25 - "Drug Atom Features"
Cohesion: 0.15
Nodes (8): PyGData, Drug, Any, Drug model — small molecule with SMILES and (optionally) a graph encoding.  The, A drug molecule keyed by PubChem CID.      `molecule` (RDKit Mol) and `graph` (P, Build a Drug from a SMILES string, validating with RDKit.          Raises ValueE, Tests for src.domain.drug., TestDrug

### Community 26 - "DoubleTower Dataset"
Cohesion: 0.11
Nodes (18): 0. Guiding Principles, 1. Target Layout (after the refactor), 2. Phasing, 3. Risks & Mitigations, 4. Out of Scope, 5. Execution Order in This Session, Pharmagen — `src/` Refactor Plan, Phase 0 — Triage & Backup *(target: ~30 min, blocks everything)* (+10 more)

### Community 27 - "Graph Visualization Script"
Cohesion: 0.17
Nodes (8): Decompress the .gz directly into the target FASTA path., Generate the .fai index used for random-access FASTA reads., Generate the BWA index files (.bwt, .pac, etc.) used for alignment.          Not, Full pipeline: download → samtools index → BWA index., Download, refresh, and index the GRCh38 reference genome.      Ensures the FASTA, Return True if the remote file is newer than the local copy., Download the gzipped FASTA with a progress bar, then decompress., ReferenceGenomeManager

### Community 28 - "HGVS Parser & Protein Pos"
Cohesion: 0.18
Nodes (13): Logger, load_sample(), main(), DataFrame, Path, Evaluate ADMET-AI on a sample of the real Pharmagen SMILES.  Step (c) of the dru, Return ``n`` random ``(cid, smiles)`` pairs from the SMILES dictionary., Run ADMET-AI over ``smiles`` and return the prediction DataFrame. (+5 more)

### Community 29 - "Pydantic Settings"
Cohesion: 0.16
Nodes (12): _get_optuna_study(), _get_train_pipeline(), Path, Execute standard training workflow with cached import., Execute Optuna optimization workflow with cached import., Lazy import with automatic cache of train_pipeline., Lazy import with automatic cache of run_optuna_study., _run_optuna_training() (+4 more)

### Community 30 - "HGVS Domain Models"
Cohesion: 0.14
Nodes (14): 1. Memory management, 2. SOLID applied to the codebase, 3. Error model, 4. Validation framework, 5. Pydantic at every boundary, 6. Subprocess safety, 7. Documentation surface, 8. What still needs doing (+6 more)

### Community 31 - "Checkpoint Manager"
Cohesion: 0.14
Nodes (14): 1. Install, 2. Build the offline graph library, 3. Train a model, 4. Inference (FastAPI), 5. Inference (CLI menu), 6. Configuration, 7. Domain models — the canonical types, 8. Testing (+6 more)

### Community 32 - "Variant Domain Models"
Cohesion: 0.24
Nodes (13): build_adjacency_matrix(), build_edge_table(), build_node_table(), load_graph(), main(), print_summary(), DataFrame, Path (+5 more)

### Community 33 - "CLI Entry Point"
Cohesion: 0.15
Nodes (10): DataValidator, DataFrame, Runtime validation utilities for Pharmagen.  Provides lightweight guards for dat, Return class counts; warn about rare classes below *min_samples_per_class*., Raise ``ValueError`` if the DataFrame is missing required columns., Inspects DataFrame quality at runtime., Return per-column missing-value fractions; warn if any exceed *threshold*., load_and_clean_data() (+2 more)

### Community 34 - "dbSNP Domain & Fetch"
Cohesion: 0.16
Nodes (13): load_json(), print_conditions_details(), print_gnu_notice(), print_warranty_details(), Any, Path, Console-facing IO helpers — JSON utilities and GPL notices., Write ``data`` as JSON with 2-space indent. (+5 more)

### Community 35 - "Chromosome Mapping"
Cohesion: 0.15
Nodes (12): 3-Layer Query Rule, Context Navigation (Graphify), Conventions to keep, Do NOT, Engine contract (training ↔ inference), Environment & Commands, Layout, Outstanding tech debt (+4 more)

### Community 36 - "Drug Record Loading"
Cohesion: 0.15
Nodes (13): 1. Variants TSV (`data/snp_data_output.tsv`), 2. Drugs TSV (`data/drugs_cid.tsv`), 3. Reference FASTA (`data/ref_genome/HSapiens_GChr38.fa`), 4. Per-gene VCF folder (`data/haplotype_variants/`), CLI, Common workflows, Inputs, Library Builder (+5 more)

### Community 37 - "Geno Functional Profile"
Cohesion: 0.22
Nodes (7): LRScheduler, Any, device, Module, MutableSequence, Optimizer, Return one loss module per target column, moved to *device*.

### Community 38 - "Protein-Change Featurizer"
Cohesion: 0.24
Nodes (4): TestClient, Tests for /v1/predict.  Predictions can't actually run without trained model art, TestBatchPredict, TestSinglePredict

### Community 39 - "Tabular Loader & Predictor"
Cohesion: 0.20
Nodes (6): ProgressBar, Any, Simple progress bar for console (alternative to tqdm for minimal dependencies)., Update progress by n steps., Render the progress bar., Add color to the progress bar

### Community 40 - "Drug Domain Model"
Cohesion: 0.18
Nodes (5): Animation loop running in separate thread., Prompt for a choice from a list. (Enumerated input)          Args:             p, Context Manager for console loading animations.      Usage:     >>> with Spinner, Check if terminal supports ANSI colors., Spinner

### Community 41 - "Genomic Graph Builder"
Cohesion: 0.24
Nodes (6): Any, Return ``True`` if all Optuna search-space entries look sane.          Accepts b, PGenTuner, Any, Trial, Orchestrator for Optuna-based Hyperparameter Optimization.

### Community 42 - "Star-Allele Loading Tests"
Cohesion: 0.20
Nodes (7): Any, DataLoader, device, Module, MutableSequence, Optimizer, Compile the model with ``torch.compile`` for inference speed.

### Community 44 - "SO Consequence Featurizer"
Cohesion: 0.20
Nodes (10): API surface, Configuration flow, Conventions, Data flow during training, Engine bootstrap contract, High-level picture, Module map, Outstanding tech debt (+2 more)

### Community 45 - "Drug Graph Builder"
Cohesion: 0.20
Nodes (10): 1. Memory accumulates across Optuna trials, 2. Dataset preloading on large corpora, 3. Batch size too large for VRAM, 4. Gradient accumulation as a workaround, Causes of OOM and what to do, Hardware budgets, Memory Optimization, Practical checks (+2 more)

### Community 46 - "Gene-Variant Graph & PGx"
Cohesion: 0.20
Nodes (10): At a glance, Author, Documentation, Install, Key features, License, Pharmagen, Project layout (+2 more)

### Community 47 - "CSV/TSV Loader Tests"
Cohesion: 0.20
Nodes (10): Autor, Características principales, Documentación, Estructura del proyecto, Instalación, Licencia, Pharmagen, Requisitos (+2 more)

### Community 48 - "Star-Allele Domain Model"
Cohesion: 0.28
Nodes (5): ABC, Trainers for the Pharmagen training pipeline.  from src.model.training import St, Shared training-loop primitives.  ``TrainingLoop`` owns the parts that don't cha, Optuna-trial trainer.  Skips ``torch.compile`` (overhead is wasted on pruned tri, Standard (non-Optuna) trainer.  Uses ``CheckpointManager`` for best-checkpoint p

### Community 49 - "Prediction Request/Response"
Cohesion: 0.22
Nodes (6): ConsoleIO, Path, Static helper for Console Input/Output operations with validation., Prompt for a file/directory path with validation.          Args:             pro, Prompt for a float with validation., Clear the console screen (cross-platform).

### Community 50 - "SMILES-to-Graph Bonds"
Cohesion: 0.22
Nodes (7): Any, DataLoader, device, Module, MutableSequence, Optimizer, Trial

### Community 51 - "Geno Library Assembly"
Cohesion: 0.22
Nodes (8): CI, Coverage, Markers, Pharmagen Test Suite, Running tests, Shared fixtures (`tests/conftest.py`), Structure, Writing new tests

### Community 52 - "VCF Variant Ingestion"
Cohesion: 0.33
Nodes (3): TestClient, Tests for /v1/library/{drugs,genes}., TestLibrary

### Community 53 - "Interactive Predict CLI"
Cohesion: 0.31
Nodes (4): TestClient, Tests for /v1/models., TestGetModel, TestListModels

### Community 54 - "Core Validators"
Cohesion: 0.33
Nodes (4): TestClient, Tests for /health and /ready., TestHealth, TestReady

### Community 55 - "GenoFunc Provider"
Cohesion: 0.33
Nodes (5): device(), mock_encoder(), Shared pytest fixtures for Pharmagen test suite., Appropriate torch device for testing., Mock sklearn LabelEncoder.

### Community 57 - "Reference Genome Handling"
Cohesion: 0.40
Nodes (3): analyze_graph_directory(), Path, Parses a directory to calculate total and average graph statistics.     :param r

## Knowledge Gaps
- **165 isolated node(s):** `dev_Pharmagen`, `ref_genome_utils.sh script`, `Project Overview`, `Refactor Status`, `Layout` (+160 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **11 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `get_settings()` connect `Engine & Collator` to `CLI Entry Point`, `ADMET Provider`, `dbSNP Domain & Fetch`, `Setup Orchestration`, `Geno Library Builder`, `Geno Functional Profile`, `Star-Allele Loading Tests`, `GFF Annotation Reader`, `Star-Allele Domain Model`, `Library Builder Orchestrator`, `Exception Hierarchy`, `FastAPI App & DI`, `Double-Graph Visualization`, `Build Manifest`, `Graph Visualization Script`, `HGVS Parser & Protein Pos`?**
  _High betweenness centrality (0.182) - this node is a cross-community bridge._
- **Why does `PGenPredictor` connect `Setup Orchestration` to `Engine & Collator`, `Library Builder Orchestrator`, `Exception Hierarchy`, `Double-Graph Visualization`, `Build Manifest`?**
  _High betweenness centrality (0.039) - this node is a cross-community bridge._
- **Why does `TrainingLoop` connect `Raw HGVS Adapter` to `Star-Allele Domain Model`, `NGS Pipeline`, `Variant Model Tests`, `Gene & Star-Allele Models`?**
  _High betweenness centrality (0.034) - this node is a cross-community bridge._
- **Are the 23 inferred relationships involving `StarAllele` (e.g. with `BatchPredictRequest` and `BatchPredictResponse`) actually correct?**
  _`StarAllele` has 23 INFERRED edges - model-reasoned connections that need verification._
- **Are the 8 inferred relationships involving `Gene` (e.g. with `StarAlleleMap` and `StarAlleleRecord`) actually correct?**
  _`Gene` has 8 INFERRED edges - model-reasoned connections that need verification._
- **Are the 7 inferred relationships involving `Variant` (e.g. with `DbSnpGene` and `DbSnpSummary`) actually correct?**
  _`Variant` has 7 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Execute training in headless mode.`, `Execute prediction in headless mode.`, `Main entry point for Pharmagen CLI.      Args:         args: Parsed command line` to the rest of the system?**
  _483 weakly-connected nodes found - possible documentation gaps or missing edges._