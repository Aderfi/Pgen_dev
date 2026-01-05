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

This module provides functionality to build and maintain indices
for drug and gene variant graph files, following SRP (Single Responsibility Principle).
"""

import logging
import re
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)


class GraphIndexBuilder:
    """Builds and maintains indices for graph files.
    
    Follows Single Responsibility Principle: only responsible for indexing.
    """

    @staticmethod
    def build_drug_index(drug_lib: Path) -> Dict[str, Path]:
        """Build index mapping drug IDs to file paths.
        
        Args:
            drug_lib: Directory containing drug graph files (.pt).
            
        Returns:
            Dictionary mapping drug IDs to their file paths.
            
        Example:
            >>> index = GraphIndexBuilder.build_drug_index(Path("src/library/drugs"))
            >>> index["10007"]  # Returns Path to 10007_chlorphentermine.pt
        """
        if not drug_lib.exists():
            logger.warning(f"Drug library not found: {drug_lib}")
            return {}
            
        index_drugs = {}
        for file_path in drug_lib.glob("*.pt"):
            # Extract ID from filename (e.g., '10007' from '10007_chlorphentermine.pt')
            match = re.match(r"^(\d+)_", file_path.name)
            if match:
                drug_id = match.group(1)
                index_drugs[drug_id] = file_path
            else:
                logger.debug(f"Skipping file with unexpected name format: {file_path.name}")
                
        logger.debug(f"Indexed {len(index_drugs)} drug graphs from {drug_lib}")
        return index_drugs

    @staticmethod
    def build_gene_variant_index(variant_lib: Path) -> Dict[str, Dict[str, Path]]:
        """Build nested index for gene variants.
        
        Args:
            variant_lib: Directory containing gene variant subdirectories.
            
        Returns:
            Nested dictionary: {gene_id: {variant_name: Path}}
            
        Example:
            >>> index = GraphIndexBuilder.build_gene_variant_index(Path("src/library/gene_graphs"))
            >>> index["CYP2D6"]["*4"]  # Returns Path to CYP2D6_star4.pt
        """
        if not variant_lib.exists():
            logger.warning(f"Variant library not found: {variant_lib}")
            return {}
            
        index_genes = {}
        
        # Initialize gene directories
        for dir_path in variant_lib.rglob("**/"):
            if dir_path.is_dir() and dir_path.name:
                index_genes[dir_path.name] = {}

        # Index all variant files
        for file_path in variant_lib.glob("**/*.pt"):
            filename_clean = file_path.stem  # Name without extension
            
            try:
                gene_id, variant = filename_clean.split("_", 1)
                
                # Convert "star" notation to "*"
                if variant.startswith("star"):
                    variant = variant.replace("star", "*")

                if gene_id not in index_genes:
                    index_genes[gene_id] = {}
                    
                index_genes[gene_id][variant] = file_path
                
            except ValueError:
                logger.debug(f"Skipping file with unexpected name format: {file_path.name}")
                continue
                
        # Log statistics
        total_variants = sum(len(variants) for variants in index_genes.values())
        logger.debug(
            f"Indexed {len(index_genes)} genes with {total_variants} variants from {variant_lib}"
        )
        
        return index_genes


class GraphValidator:
    """Validates graph data structures.
    
    Follows SRP: only responsible for validation logic.
    """

    @staticmethod
    def validate_graph_dimensions(
        graph_data,
        expected_node_features: int,
        expected_edge_features: int,
        graph_type: str = "unknown"
    ) -> bool:
        """Validate that a graph has expected dimensions.
        
        Args:
            graph_data: PyTorch Geometric Data object.
            expected_node_features: Expected number of node features.
            expected_edge_features: Expected number of edge features.
            graph_type: Type of graph for logging (e.g., "drug", "variant").
            
        Returns:
            True if valid, False otherwise.
        """
        if not hasattr(graph_data, "x"):
            logger.error(f"{graph_type} graph missing node features")
            return False
            
        if graph_data.x.shape[1] != expected_node_features:
            logger.warning(
                f"{graph_type} graph has {graph_data.x.shape[1]} node features, "
                f"expected {expected_node_features}"
            )
            return False
            
        if expected_edge_features > 0:
            if not hasattr(graph_data, "edge_attr"):
                logger.error(f"{graph_type} graph missing edge attributes")
                return False
                
            if graph_data.edge_attr.shape[1] != expected_edge_features:
                logger.warning(
                    f"{graph_type} graph has {graph_data.edge_attr.shape[1]} edge features, "
                    f"expected {expected_edge_features}"
                )
                return False
                
        return True

    @staticmethod
    def check_graph_consistency(graph_data, graph_id: str = "") -> bool:
        """Check if graph data is consistent and well-formed.
        
        Args:
            graph_data: PyTorch Geometric Data object.
            graph_id: Identifier for logging.
            
        Returns:
            True if consistent, False otherwise.
        """
        if not hasattr(graph_data, "x") or not hasattr(graph_data, "edge_index"):
            logger.error(f"Graph {graph_id} missing required attributes")
            return False
            
        num_nodes = graph_data.x.shape[0]
        
        if hasattr(graph_data, "edge_index") and graph_data.edge_index.numel() > 0:
            max_edge_idx = graph_data.edge_index.max().item()
            if max_edge_idx >= num_nodes:
                logger.error(
                    f"Graph {graph_id} has edge index {max_edge_idx} "
                    f"but only {num_nodes} nodes"
                )
                return False
                
        return True
