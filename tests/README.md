# Pharmagen Test Suite

This directory contains the organized test suite for Pharmagen following Python testing best practices.

## Structure

```
tests/
├── conftest.py                 # Shared pytest fixtures
├── pytest.ini                  # Pytest configuration
│
├── unit/                       # Unit tests
│   ├── config/                 # Configuration management tests
│   │   └── test_manager.py
│   ├── data/                   # Data processing and dataset tests
│   │   └── test_datasets.py
│   ├── modeling/               # Model architecture tests
│   │   └── test_deepfm.py
│   └── utils/                  # Utility function tests
│       └── test_data_processing.py
│
├── integration/                # Integration tests
│   ├── test_pipeline.py        # Pipeline integration tests
│   └── test_predict.py         # Prediction workflow tests
│
├── benchmarks/                 # Performance benchmarks
│   └── test_batch_size.py      # Batch size optimization tests
│
└── fixtures/                   # Test data and fixtures
```

## Running Tests

### Run all tests
```bash
pytest tests/
```

### Run unit tests only
```bash
pytest tests/unit/
```

### Run integration tests
```bash
pytest tests/integration/
```

### Run tests with specific markers
```bash
# Skip slow tests
pytest tests/ -m "not slow"

# Run only integration tests
pytest tests/ -m integration

# Run only benchmark tests
pytest tests/ -m benchmark

# Skip CUDA-required tests
pytest tests/ -m "not cuda"
```

### Run tests with verbose output
```bash
pytest tests/ -v
```

### Run specific test file
```bash
pytest tests/unit/config/test_manager.py
```

## Test Markers

- `unit`: Unit tests (default for tests/unit/)
- `integration`: Integration tests (default for tests/integration/)
- `benchmark`: Performance benchmarks
- `slow`: Tests that take significant time to run
- `cuda`: Tests that require CUDA/GPU

## Shared Fixtures

The `conftest.py` file provides shared fixtures:

- `sample_dataframe`: Basic DataFrame for testing
- `double_tower_dataframe`: DataFrame for DoubleTowerDataset
- `model_params`: Default DeepFM model parameters
- `dummy_graph`: PyTorch Geometric dummy graph
- `mock_config`: Mock configuration for pipeline
- `temp_library`: Temporary directory for tests
- `device`: Appropriate torch device
- `mock_encoder`: Mock sklearn encoder

## Writing New Tests

1. Place unit tests in `tests/unit/` matching the source structure
2. Place integration tests in `tests/integration/`
3. Place benchmarks in `tests/benchmarks/`
4. Use shared fixtures from `conftest.py`
5. Add appropriate markers (`@pytest.mark.slow`, `@pytest.mark.cuda`, etc.)
6. Follow naming convention: `test_*.py` for files, `test_*` for functions

## CI/CD Integration

Tests can be run in CI/CD pipelines with:
```bash
pytest tests/ --tb=short --color=yes
```

For coverage reports (requires pytest-cov):
```bash
pytest tests/ --cov=src --cov-report=html --cov-report=term-missing
```
