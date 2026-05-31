# Code Quality Guidelines

Coding standards for the Pharmagen project, grounded in the Zen of Python and
SOLID principles. The examples below use the post-refactor module layout — see
[`docs/ARCHITECTURE.md`](ARCHITECTURE.md) for the full map.

## Zen of Python in Practice

### Beautiful is better than ugly

```python
# ❌ BAD
def f(x,y,z):
    return x*y+z if x>0 else z-y

# ✅ GOOD
def calculate_result(multiplier: float, value: float, offset: float) -> float:
    """Calculate result based on multiplier sign."""
    if multiplier > 0:
        return multiplier * value + offset
    return offset - value
```

### Explicit is better than implicit

```python
# ❌ BAD
def load_data(path):
    # Returns different types based on file extension
    if path.endswith('.csv'):
        return pd.read_csv(path)
    return pd.read_json(path)

# ✅ GOOD
def load_csv(path: Path) -> pd.DataFrame:
    """Load data from CSV file."""
    return pd.read_csv(path)

def load_json(path: Path) -> pd.DataFrame:
    """Load data from JSON file."""
    return pd.read_json(path)
```

### Simple is better than complex

```python
# ❌ BAD: Over-engineered
class DataProcessorFactory:
    _instances = {}
    def __new__(cls, *args, **kwargs):
        key = (cls, args, tuple(kwargs.items()))
        if key not in cls._instances:
            cls._instances[key] = super().__new__(cls)
        return cls._instances[key]

# ✅ GOOD: Simple and clear
class DataProcessor:
    """Process data for training."""
    def __init__(self, config: dict):
        self.config = config
```

### Flat is better than nested

```python
# ❌ BAD: Deeply nested
def process(data):
    if data:
        if len(data) > 0:
            if validate(data):
                return transform(data)
    return None

# ✅ GOOD: Early returns
def process(data):
    """Process validated data."""
    if not data or len(data) == 0:
        return None
    if not validate(data):
        return None
    return transform(data)
```

### Readability counts

```python
# ❌ BAD: Unclear abbreviations
def calc_mse(y_t, y_p):
    return ((y_t - y_p) ** 2).mean()

# ✅ GOOD: Clear names
def calculate_mean_squared_error(
    targets: torch.Tensor,
    predictions: torch.Tensor
) -> torch.Tensor:
    """Calculate mean squared error between targets and predictions."""
    squared_errors = (targets - predictions) ** 2
    return squared_errors.mean()
```

### Errors should never pass silently

```python
# ❌ BAD: Silent failure
def load_graph(path):
    try:
        return torch.load(path)
    except:
        return None

# ✅ GOOD: Explicit error handling
from src.core import GraphError

def load_graph(path: Path) -> Data:
    """Load graph from file.
    
    Raises:
        GraphError: If file is corrupt or missing.
    """
    if not path.exists():
        raise GraphError(f"Graph file not found: {path}")
    
    try:
        return torch.load(path, weights_only=False)
    except Exception as e:
        raise GraphError(f"Failed to load graph from {path}: {e}")
```

## SOLID Principles

### Single Responsibility Principle (SRP)

Each class should have one clear purpose:

```python
# ✅ GOOD: Separate responsibilities
class GraphIndexBuilder:
    """Only builds indices."""
    @staticmethod
    def build_drug_index(path: Path) -> dict:
        ...

class GraphLoader:
    """Only loads graphs."""
    def load_graph(self, graph_id: str) -> Data:
        ...

class GraphValidator:
    """Only validates graphs."""
    @staticmethod
    def validate(graph: Data) -> bool:
        ...
```

### Open/Closed Principle (OCP)

Open for extension, closed for modification:

```python
# ✅ GOOD: Use factory pattern
class LossFactory:
    _registry = {
        "cross_entropy": nn.CrossEntropyLoss,
        "focal": FocalLoss,
    }
    
    @classmethod
    def register(cls, name: str, loss_cls):
        """Register new loss function."""
        cls._registry[name] = loss_cls
    
    @classmethod
    def create(cls, name: str, **kwargs):
        """Create loss function by name."""
        return cls._registry[name](**kwargs)

# Adding new loss doesn't modify factory
LossFactory.register("custom_loss", CustomLoss)
```

### Liskov Substitution Principle (LSP)

Subtypes must be substitutable for base types:

```python
# ✅ GOOD: Consistent interface
class BaseDataset(Dataset):
    def __getitem__(self, idx: int) -> dict:
        raise NotImplementedError

class DoubleTowerDataset(BaseDataset):
    def __getitem__(self, idx: int) -> dict:
        """Returns dict with same structure as base."""
        return {
            "drug_data": ...,
            "geno_data": ...,
            "targets": ...
        }
```

### Interface Segregation Principle (ISP)

Many specific interfaces better than one general:

```python
# ✅ GOOD: Focused interfaces
class Trainable(Protocol):
    def train_epoch(self, loader: DataLoader) -> dict:
        ...

class Validatable(Protocol):
    def validate(self, loader: DataLoader) -> dict:
        ...

class Checkpointable(Protocol):
    def save_checkpoint(self, path: Path):
        ...
    def load_checkpoint(self, path: Path):
        ...
```

### Dependency Inversion Principle (DIP)

Depend on abstractions, not concretions:

```python
# ✅ GOOD: Inject dependencies
class Trainer:
    def __init__(
        self,
        model: nn.Module,  # Abstract: any nn.Module
        optimizer: Optimizer,  # Abstract: any optimizer
        loss_fn: nn.Module,  # Abstract: any loss
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
```

## Code Style Guidelines

### Imports

```python
# Standard library
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

# Third-party
import numpy as np
import pandas as pd
import torch

# Local
from src.config import get_settings
from src.core import setup_logging
```

### Docstrings

Use Google style:

```python
def train_model(
    model: nn.Module,
    data: DataLoader,
    epochs: int = 10
) -> dict:
    """Train a model on provided data.
    
    Args:
        model: Neural network model to train.
        data: DataLoader with training batches.
        epochs: Number of training epochs.
        
    Returns:
        Dictionary with training metrics.
        
    Raises:
        ValueError: If epochs < 1.
        RuntimeError: If CUDA out of memory.
        
    Example:
        >>> model = create_model()
        >>> loader = DataLoader(dataset)
        >>> metrics = train_model(model, loader, epochs=5)
    """
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}")
    ...
```

### Type Hints

Always use type hints for public APIs:

```python
from typing import Optional, Union, List, Dict, Any
from pathlib import Path

def process_file(
    path: Union[str, Path],
    config: Dict[str, Any],
    verbose: bool = False
) -> Optional[pd.DataFrame]:
    """Process file with configuration."""
    ...
```

### Error Messages

Make them actionable:

```python
# ❌ BAD: Vague
raise ValueError("Invalid configuration")

from src.core import ConfigurationError

# ✅ GOOD: Specific and actionable
raise ConfigurationError(
    f"Model '{model_name}' requires 'learning_rate' in params. "
    f"Add 'learning_rate = 0.001' to src/config/data/models.toml."
)
```

### Logging

Use appropriate levels:

```python
logger.debug("Loading graph from cache")  # Development info
logger.info("Starting training on 1000 samples")  # User info
logger.warning("Batch size reduced due to memory")  # Potential issue
logger.error("Failed to load model checkpoint")  # Error occurred
logger.critical("CUDA out of memory, aborting")  # Fatal error
```

## Common Anti-Patterns to Avoid

### Magic Numbers

```python
# ❌ BAD
if score > 0.5:
    ...

# ✅ GOOD
CLASSIFICATION_THRESHOLD = 0.5
if score > CLASSIFICATION_THRESHOLD:
    ...
```

### Mutable Default Arguments

```python
# ❌ BAD
def append_item(item, items=[]):
    items.append(item)
    return items

# ✅ GOOD
def append_item(item, items=None):
    if items is None:
        items = []
    items.append(item)
    return items
```

### Bare Except

```python
# ❌ BAD
try:
    process()
except:
    pass

# ✅ GOOD
try:
    process()
except FileNotFoundError:
    logger.warning("File not found, using defaults")
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    raise
```

### String Concatenation in Loops

```python
# ❌ BAD
result = ""
for item in items:
    result += str(item)

# ✅ GOOD
result = "".join(str(item) for item in items)
```

## Performance Tips

### Use List Comprehensions

```python
# ❌ SLOW
squares = []
for x in range(100):
    squares.append(x**2)

# ✅ FAST
squares = [x**2 for x in range(100)]
```

### Avoid Repeated Lookups

```python
# ❌ SLOW
for i in range(len(data)):
    process(data[i])

# ✅ FAST
for item in data:
    process(item)
```

### Use Generators for Large Data

```python
# ❌ MEMORY INTENSIVE
def load_all_files(directory):
    return [load_file(f) for f in directory.glob("*.pt")]

# ✅ MEMORY EFFICIENT
def load_all_files(directory):
    for f in directory.glob("*.pt"):
        yield load_file(f)
```

## Testing Guidelines

### Write Testable Code

```python
# ✅ GOOD: Easy to test
def calculate_loss(predictions, targets):
    """Pure function, deterministic, no side effects."""
    return (predictions - targets).pow(2).mean()

# Can test with:
def test_calculate_loss():
    preds = torch.tensor([1.0, 2.0, 3.0])
    targets = torch.tensor([1.0, 2.0, 3.0])
    loss = calculate_loss(preds, targets)
    assert loss.item() == 0.0
```

### Use Fixtures

```python
@pytest.fixture
def sample_config():
    return {
        "model_name": "test_model",
        "params": {"lr": 0.001}
    }

def test_model_creation(sample_config):
    model = create_model(sample_config)
    assert model is not None
```

## Summary Checklist

- [ ] Follow Zen of Python principles
- [ ] Apply SOLID principles
- [ ] Use type hints for public APIs
- [ ] Write comprehensive docstrings
- [ ] Handle errors explicitly
- [ ] Use meaningful variable names
- [ ] Keep functions focused (SRP)
- [ ] Make code testable
- [ ] Log at appropriate levels
- [ ] Avoid anti-patterns

## Additional Resources

- [PEP 8](https://peps.python.org/pep-0008/) - Python Style Guide
- [PEP 257](https://peps.python.org/pep-0257/) - Docstring Conventions
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [The Zen of Python](https://peps.python.org/pep-0020/)
