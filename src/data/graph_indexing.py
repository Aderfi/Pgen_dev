# src/data/graph_indexing.py - VERSIÓN LIMPIA
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

This module provides functionality to build indexes for drug and gene variant graph files.
Follows Single Responsibility Principle.
"""

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)


class GraphIndexBuilder:
    """Builds and maintains indices for graph files.
    Only responsible of indexing.
    """

    @staticmethod
    def build_drug_index(drug_lib:  Path) -> dict[str, Path]:
        """Build index mapping drug IDs to file paths.

        Args:
            drug_lib: Directory containing drug graph files (. pt).

        Returns:
            Dictionary mapping drug IDs to their file paths.

        Example:
            >>> index = GraphIndexBuilder.build_drug_index(Path("src/library/drugs"))
            >>> index["10007"]  # Returns Path to 10007_chlorphentermine.pt
        """
        if not drug_lib.exists():
            logger.warning(f"Drug library not found:  {drug_lib}")
            return {}

        index_drugs = {}
        for file_path in drug_lib.glob("*.pt"):
            # Extract ID from filename (e.g., '10007' from '10007_chlorphentermine.pt')
            match = re.match(r"^(\d+)_", file_path.name)
            if match:
                drug_id = match.group(1)
                index_drugs[drug_id] = file_path
            else:
                logger.debug(f"Skipping file with unexpected name: {file_path. name}")

        logger.debug(f"Indexed {len(index_drugs)} drug graphs from {drug_lib}")
        return index_drugs

    @staticmethod
    def build_gene_variant_index(variant_lib: Path) -> dict[str, dict[str, Path]]:
        """Build nested index for gene variants.

        Args:
            variant_lib: Directory containing gene variant subdirectories.

        Returns:
            Nested dictionary:  {gene_id: {variant_name: Path}}

        Example:
            >>> index = GraphIndexBuilder.build_gene_variant_index(Path("src/library/gene_graphs"))
            >>> index["CYP2D6"]["*4"]  # Returns Path to CYP2D6_star4.pt
        """
        if not variant_lib.exists():
            logger.warning(f"Variant library not found: {variant_lib}")
            return {}

        index_genes:  dict[str, dict[str, Path]] = {}

        # Initialize gene directories
        for dir_path in variant_lib.rglob("**/"):
            if dir_path.is_dir() and dir_path.name:
                index_genes[dir_path.name] = {}

        # Index all variant files
        for file_path in variant_lib.glob("**/*. pt"):
            filename_clean = file_path.stem  # Name without extension

            try:
                gene_id, variant = filename_clean.split("_", 1)

                # Convert "star" notation to "*"
                if variant.startswith("star"):
                    variant = variant. replace("star", "*")

                if gene_id not in index_genes:
                    index_genes[gene_id] = {}

                index_genes[gene_id][variant] = file_path

            except ValueError:
                logger.debug(f"Skipping file with unexpected name: {file_path.name}")
                continue

        # Log statistics
        total_variants = sum(len(variants) for variants in index_genes. values())
        logger.debug(
            f"Indexed {len(index_genes)} genes with {total_variants} variants from {variant_lib}"
        )

        return index_genes
