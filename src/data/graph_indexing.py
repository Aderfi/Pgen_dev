# src/data/graph_indexing.py
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

"""Graph indexing utilities for Pharmagen datasets.

Maps drug IDs to their on-disk graph files (``<cid>_<name>.pt``). The genotype
side needs no index: genotypes are HGVS variant graphs (``src.data.genotype``).
"""

import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


class GraphIndexBuilder:
    """Builds and maintains indices for graph files.
    Only responsible of indexing.
    """

    @staticmethod
    def build_drug_index(drug_lib: Path) -> dict[str, Path]:
        """Build index mapping drug IDs to file paths.

        Args:
            drug_lib: Directory containing drug graph files (.pt).

        Returns:
            Dictionary mapping drug IDs to their file paths.

        Example:
            >>> index = GraphIndexBuilder.build_drug_index(Path("data/library/drugs"))
            >>> index["10007"]  # Returns Path to 10007_chlorphentermine.pt
        """
        if not drug_lib.exists():
            logger.warning(f"Drug library not found: {drug_lib}")
            return {}

        index_drugs = {}
        for file_path in drug_lib.glob("*.pt"):
            match = re.match(r"^(\d+)_", file_path.name)
            if match:
                drug_id = match.group(1)
                index_drugs[drug_id] = file_path
            else:
                logger.debug(f"Skipping file with unexpected name: {file_path.name}")

        logger.debug(f"Indexed {len(index_drugs)} drug graphs from {drug_lib}")
        return index_drugs
