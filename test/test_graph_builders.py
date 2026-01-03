from pathlib import Path
from typing import Optional, Union

import matplotlib.lines as lns
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from src.utils.library_creator import GenomicGraphBuilderNEXTGEN


class GraphVisualization:
    """
    Handles all visual representation logic: Linear Genome Plots and Exports.
    Separated from the Builder to keep logic clean.
    """

    CLINICAL_COLORS = {
        "no_function": "#e74c3c",  # Red
        "decreased": "#f1c40f",  # Yellow
        "normal": "#2ecc71",  # Green
        "unknown": "#bdc3c7",  # Grey
        "reference": "#3498db",  # Blue
        "backbone": "#34495e",  # Dark Blue/Grey
    }

    ROLE_COLORS = {
        "promoter": "#ffeaa7",
        "five_prime_UTR": "#fab1a0",
        "three_prime_UTR": "#ff7675",
        "gene": "#f1f2f6",
    }

    @staticmethod
    def _get_color_by_score(score: float) -> str:
        """Maps numerical activity score to CPIC clinical colors."""
        if pd.isna(score) or score < 0:
            return GraphVisualization.CLINICAL_COLORS["unknown"]
        if score == 0.0:
            return GraphVisualization.CLINICAL_COLORS["no_function"]
        if score == 0.5:
            return GraphVisualization.CLINICAL_COLORS["decreased"]
        if score >= 1.0:
            return GraphVisualization.CLINICAL_COLORS["normal"]
        return GraphVisualization.CLINICAL_COLORS["unknown"]

    @staticmethod
    def plot_linear_genome(
        G: nx.DiGraph, gene_name: str, output_file: Optional[Union[str, Path]] = None
    ):
        """
        Advanced Pharmacogenomic Visualization:
        Plots the graph topologically with relative scaling and metabolic heatmaps.
        """
        plt.figure(figsize=(22, 8))
        ax = plt.gca()

        # A. LAYOUT CALCULATION (Artificial X-Index based on Genomic Position)
        # Sort nodes by genomic position to linearize the graph
        nodes_sorted = sorted(G.nodes(data=True), key=lambda x: x[1].get("pos", 0))
        x_map = {node[0]: i for i, node in enumerate(nodes_sorted)}

        pos_layout = {}
        node_colors = []
        node_sizes = []

        # B. NODE PROCESSING
        for node, data in G.nodes(data=True):
            node_type = data.get("type", "backbone")
            x_coord = x_map[node]
            y_coord = 0.0
            size = 300
            color = GraphVisualization.CLINICAL_COLORS["backbone"]

            if node_type in ["backbone", "split", "merge"]:
                y_coord = 0
                if node_type == "backbone":
                    size = 100
                else:
                    color = "#dcdde1"  # Lighter for connectors

            elif node_type == "allele_ref":
                y_coord = 0.5
                color = GraphVisualization.CLINICAL_COLORS["reference"]
                size = 500

            elif node_type == "allele_alt":
                y_coord = -0.5
                score = data.get("activity_score", -1.0)
                color = GraphVisualization._get_color_by_score(score)
                # Star Alleles get larger bubbles
                size = 700 if data.get("variant_type") == "STAR_ALLELE" else 500

            pos_layout[node] = (x_coord, y_coord)
            node_colors.append(color)
            node_sizes.append(size)

        # C. BACKGROUND: FUNCTIONAL CONTEXT
        # Draws colored bands for promoters, UTRs, etc.
        for node, data in G.nodes(data=True):
            if (
                data.get("type") == "backbone"
                and data.get("role") in GraphVisualization.ROLE_COLORS
            ):
                x_start = x_map[node] - 0.5
                role = data["role"]
                rect = patches.Rectangle(
                    (x_start, -1.2),
                    1,
                    2.4,
                    color=GraphVisualization.ROLE_COLORS[role],
                    alpha=0.3,
                    zorder=0,
                )
                ax.add_patch(rect)
                plt.text(
                    x_start + 0.5, 1.3, role.upper(), fontsize=7, ha="center", alpha=0.5
                )

        # D. DRAWING
        nx.draw_networkx_edges(
            G,
            pos_layout,
            edge_color="#7f8c8d",
            alpha=0.4,
            connectionstyle="arc3,rad=0.1",
            arrowsize=15,
        )

        nx.draw_networkx_nodes(
            G,
            pos_layout,
            node_color=node_colors,
            node_size=node_sizes,
            edgecolors="white",
            linewidths=1.5,
        )

        # Labels for Variants Only
        labels = {}
        for node, data in G.nodes(data=True):
            if data.get("type") == "allele_alt":
                haplo = data.get("haplotype", "")
                score = data.get("activity_score", "N/A")
                labels[node] = f"{haplo}\n(S:{score})"

        nx.draw_networkx_labels(G, pos_layout, labels, font_size=9, font_weight="bold")

        # E. FINAL TOUCHES
        plt.title(
            f"Clinical Variant Interpretation: Gene {gene_name}", fontsize=16, pad=20
        )
        plt.ylim(-1.5, 1.5)
        plt.axis("off")

        # Legend
        legend_elements = [
            patches.Patch(
                color=GraphVisualization.CLINICAL_COLORS["no_function"],
                label="Activity: 0.0 (PM)",
            ),
            patches.Patch(
                color=GraphVisualization.CLINICAL_COLORS["decreased"],
                label="Activity: 0.5 (IM)",
            ),
            patches.Patch(
                color=GraphVisualization.CLINICAL_COLORS["normal"],
                label="Activity: ≥1.0 (NM/UM)",
            ),
            lns.Line2D(
                [0],
                [0],
                color=GraphVisualization.CLINICAL_COLORS["reference"],
                marker="o",
                label="Reference",
                markersize=10,
            ),
        ]
        ax.legend(
            handles=legend_elements, loc="lower right", title="Metabolic Status (CPIC)"
        )

        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            print(f"📊 Plot saved to: {output_file}")

        # Note: plt.show() should generally be called by the user or main script if interactive
        # plt.show()


if __name__ == "__main__":
    base_dir = Path(__file__).parent / "data"
    parquet_file = base_dir / "genome_library.parquet"

    if not parquet_file.exists():
        print(f"❌ Library file not found: {parquet_file}")
    else:
        # 1. Instantiate
        builder = GenomicGraphBuilderNEXTGEN(parquet_file)

        # 2. Select Target Gene (Must exist in the 'gene_context' column of your library)
        gene_target = "CYP2D6"  # Example Pharmacogene

        # 3. Build Graph
        output_json = base_dir / f"{gene_target}_graph.json"
        G = builder.build_gene_graph(gene_target, output_json_path=output_json)

        # 4. Visualize & Export
        if G:
            # A. Linear Plot (Matplotlib)
            plot_file = base_dir / f"{gene_target}_linear.png"
            GraphVisualization.plot_linear_genome(G, gene_target, output_file=plot_file)

            # B. GraphML Export (Cytoscape)
            graphml_file = base_dir / f"{gene_target}_graph.graphml"
            builder.export_to_graphml(G, output_path=graphml_file)

            # C. Stats
            print("\n--- Graph Statistics ---")
            print(f"Total Nodes: {G.number_of_nodes()}")
            n_variants = len(
                [n for n, d in G.nodes(data=True) if d.get("type") == "allele_alt"]
            )
            print(f"Variant Bubbles: {n_variants}")

            # D. Data Inspection
            print("\n--- Sample Variant Node ---")
            for n, d in G.nodes(data=True):
                if d.get("type") == "allele_alt":
                    print(f"ID: {n}")
                    print(f"Metadata: {d}")
                    break
