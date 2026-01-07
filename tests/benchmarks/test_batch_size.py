# Pharmagen - Pharmacogenetic Prediction and Therapeutic Efficacy
# Copyright (C) 2025 Adrim Hamed Outmani
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Benchmark tests for optimal batch size determination."""
import pytest
import torch
import torch.nn

pytest.importorskip("src.model")
pytest.importorskip("src.performance_monitor")

from src.model import DeepFM_PGenModel
from src.performance_monitor import estimate_optimal_batch_size


@pytest.mark.benchmark
@pytest.mark.cuda
@pytest.mark.skip(reason="Requires CUDA and may cause OOM on test runners")
def test_hardware_capacity():
    """Test determination of optimal batch size for hardware."""
    n_features = {"drug": 100, "gene": 50, "allele": 120, "genalle": 200}
    target_dims = {"outcome": 2, "type": 5, "variant": 10}

    model = DeepFM_PGenModel(
        n_features=n_features,
        target_dims=target_dims,
        embedding_dim=128,
        hidden_dim=256,
        dropout_rate=0.1,
        n_layers=2,
    )

    # Create sample input (one row is enough, function will replicate)
    sample_input = {
        "drug": torch.tensor([1], dtype=torch.long),
        "gene": torch.tensor([5], dtype=torch.long),
        "allele": torch.tensor([120], dtype=torch.long),
        "genalle": torch.tensor([200], dtype=torch.long),
    }

    # Execute estimation
    # Tests batch sizes like 256, 128, 64... until finding OOM limit
    optimal_bs = estimate_optimal_batch_size(
        model=model,
        sample_input=sample_input,
        max_batch_size=4096,
        device=torch.device("cuda"),
    )

    assert optimal_bs > 0, "Optimal batch size should be positive"
    assert optimal_bs <= 4096, "Optimal batch size should not exceed max"
    print(f"--> Configure 'batch_size' in config.toml to: {optimal_bs}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
