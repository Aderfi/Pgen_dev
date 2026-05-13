# Pharmagen Test Suite

The test suite mirrors `src/` and is configured through
`pyproject.toml` (`[tool.pytest.ini_options]`). All paths below are relative
to the repository root.

## Structure

```
tests/
├── conftest.py            Shared pytest fixtures (device, mock_encoder)
├── fixtures/              Sample data files used by tests
│
├── unit/                  Unit tests (231 tests; ~3 s)
│   ├── api/               FastAPI routers + DI
│   │   ├── conftest.py
│   │   ├── test_health.py
│   │   ├── test_library.py
│   │   ├── test_models.py
│   │   └── test_predict.py
│   ├── config/            Pydantic Settings + ModelConfig
│   │   ├── test_models_config.py
│   │   └── test_settings.py
│   ├── data/              Loaders, encoders, cache, normalize, cleaning,
│   │                      datasets, and the offline library builder
│   │   ├── test_cache.py
│   │   ├── test_cleaning.py
│   │   ├── test_datasets.py
│   │   ├── test_encoders.py
│   │   ├── test_library_chromosome.py
│   │   ├── test_library_drugs.py        (pins drug 25/7 schema)
│   │   ├── test_library_genes.py        (pins gene 9/3 schema)
│   │   ├── test_library_manifest.py     (atomic resume)
│   │   ├── test_library_organize.py
│   │   ├── test_library_pgx.py
│   │   ├── test_loaders.py
│   │   ├── test_migration.py
│   │   └── test_normalize.py
│   ├── domain/            Pydantic v2 domain models
│   │   ├── test_drug.py
│   │   ├── test_gene.py
│   │   ├── test_prediction.py
│   │   └── test_variant.py
│   ├── genomics/          StarAlleleMap and friends
│   │   └── test_star_alleles.py
│   └── modeling/          Trainer factory + training-loop concerns
│       └── test_trainer_factory.py
│
├── integration/           Currently empty (smoke tests are Phase 8 work)
│
└── benchmarks/            Performance budgets
    ├── test_batch_size.py
    ├── test_graph_size.py
    └── test_memory_estim.py
```

## Running tests

```bash
pytest tests/unit/                       # all unit tests (231 tests)
pytest tests/unit/ -q                    # quieter output
pytest tests/unit/api/                   # only the FastAPI tests
pytest tests/unit/data/test_library_*.py # only the library-builder schema tests
pytest tests/unit/ -v                    # verbose
```

Filter by marker (defined in `pyproject.toml`):

```bash
pytest tests/ -m "not slow"              # skip slow tests
pytest tests/ -m unit                    # unit only
pytest tests/ -m integration             # integration only (currently empty)
```

Run a single test:

```bash
pytest tests/unit/data/test_library_drugs.py::test_smiles_to_graph_shape
```

## Coverage

Coverage is enabled by default via `pyproject.toml`
(`addopts = ["--cov=src", "--cov-report=term-missing", ...]`). For an HTML
report:

```bash
pytest tests/unit/ --cov=src --cov-report=html
open htmlcov/index.html
```

## Shared fixtures (`tests/conftest.py`)

- `device` — picks `cuda` if available, otherwise `cpu`.
- `mock_encoder` — a `MagicMock` shaped like `sklearn.preprocessing.LabelEncoder`.

API-specific fixtures live in `tests/unit/api/conftest.py`.

## Markers

- `slow` — tests that take significant time.
- `integration` — end-to-end tests (currently no occurrences).
- `unit` — unit tests (default for everything under `tests/unit/`).

## Writing new tests

1. Mirror the `src/` path under `tests/unit/`. A new module under
   `src/foo/bar.py` gets `tests/unit/foo/test_bar.py`.
2. Reuse `conftest.py` fixtures rather than re-creating them in every file.
3. Pin any contract that future regressions would silently invalidate — see
   `test_library_drugs.py` for the 25/7 feature-count assertion.
4. Use markers (`@pytest.mark.slow`, `@pytest.mark.integration`) sparingly.
5. Follow the standard naming convention: `test_*.py` files, `test_*`
   functions, `Test*` classes.

## CI

Phase 8 (pending) will add `.github/workflows/` to run the suite on every PR.
Until then, run locally before opening a PR.
