import pandas as pd
import networkx as nx
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lns
from pathlib import Path
from typing import Dict, Any, Optional, List, Union


# --- 1. DATA ENCODING & SERIALIZATION ---

class GraphEncoders(json.JSONEncoder):
    """
    Custom JSON Encoder to handle NumPy data types (int32, float32, ndarrays)
    commonly found in Pandas and Scientific Computing.
    """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(GraphEncoders, self).default(obj)

    @staticmethod
    def clean_nans_for_json(data: Dict) -> Dict:
        """Recursively cleans NaNs from dictionary to ensure valid JSON."""
        for node in data.get('nodes', []):
            for key, val in node.items():
                if isinstance(val, float) and np.isnan(val):
                    node[key] = None  # Standard JSON null
        return data


# --- 2. VISUALIZATION ENGINE ---

class GraphVisualization:
    """
    Handles all visual representation logic: Linear Genome Plots and Exports.
    Separated from the Builder to keep logic clean.
    """

    CLINICAL_COLORS = {
        'no_function': '#e74c3c',  # Red
        'decreased': '#f1c40f',  # Yellow
        'normal': '#2ecc71',  # Green
        'unknown': '#bdc3c7',  # Grey
        'reference': '#3498db',  # Blue
        'backbone': '#34495e'  # Dark Blue/Grey
    }

    ROLE_COLORS = {
        'promoter': '#ffeaa7',
        'five_prime_UTR': '#fab1a0',
        'three_prime_UTR': '#ff7675',
        'gene': '#f1f2f6'
    }

    @staticmethod
    def _get_color_by_score(score: float) -> str:
        """Maps numerical activity score to CPIC clinical colors."""
        if pd.isna(score) or score < 0: return GraphVisualization.CLINICAL_COLORS['unknown']
        if score == 0.0: return GraphVisualization.CLINICAL_COLORS['no_function']
        if score == 0.5: return GraphVisualization.CLINICAL_COLORS['decreased']
        if score >= 1.0: return GraphVisualization.CLINICAL_COLORS['normal']
        return GraphVisualization.CLINICAL_COLORS['unknown']

    @staticmethod
    def plot_linear_genome(G: nx.DiGraph, gene_name: str, output_file: Optional[Union[str, Path]] = None):
        """
        Advanced Pharmacogenomic Visualization:
        Plots the graph topologically with relative scaling and metabolic heatmaps.
        """
        plt.figure(figsize=(22, 8))
        ax = plt.gca()

        # A. LAYOUT CALCULATION (Artificial X-Index based on Genomic Position)
        # Sort nodes by genomic position to linearize the graph
        nodes_sorted = sorted(G.nodes(data=True), key=lambda x: x[1].get('pos', 0))
        x_map = {node[0]: i for i, node in enumerate(nodes_sorted)}

        pos_layout = {}
        node_colors = []
        node_sizes = []

        # B. NODE PROCESSING
        for node, data in G.nodes(data=True):
            node_type = data.get('type', 'backbone')
            x_coord = x_map[node]
            y_coord = 0.0
            size = 300
            color = GraphVisualization.CLINICAL_COLORS['backbone']

            if node_type in ['backbone', 'split', 'merge']:
                y_coord = 0
                if node_type == 'backbone':
                    size = 100
                else:
                    color = '#dcdde1'  # Lighter for connectors

            elif node_type == 'allele_ref':
                y_coord = 0.5
                color = GraphVisualization.CLINICAL_COLORS['reference']
                size = 500

            elif node_type == 'allele_alt':
                y_coord = -0.5
                score = data.get('activity_score', -1.0)
                color = GraphVisualization._get_color_by_score(score)
                # Star Alleles get larger bubbles
                size = 700 if data.get('variant_type') == 'STAR_ALLELE' else 500

            pos_layout[node] = (x_coord, y_coord)
            node_colors.append(color)
            node_sizes.append(size)

        # C. BACKGROUND: FUNCTIONAL CONTEXT
        # Draws colored bands for promoters, UTRs, etc.
        for node, data in G.nodes(data=True):
            if data.get('type') == 'backbone' and data.get('role') in GraphVisualization.ROLE_COLORS:
                x_start = x_map[node] - 0.5
                role = data['role']
                rect = patches.Rectangle(
                    (x_start, -1.2), 1, 2.4,
                    color=GraphVisualization.ROLE_COLORS[role],
                    alpha=0.3, zorder=0
                )
                ax.add_patch(rect)
                plt.text(x_start + 0.5, 1.3, role.upper(), fontsize=7, ha='center', alpha=0.5)

        # D. DRAWING
        nx.draw_networkx_edges(G, pos_layout, edge_color='#7f8c8d', alpha=0.4,
                               connectionstyle='arc3,rad=0.1', arrowsize=15)

        nx.draw_networkx_nodes(G, pos_layout, node_color=node_colors,
                               node_size=node_sizes, edgecolors='white', linewidths=1.5)

        # Labels for Variants Only
        labels = {}
        for node, data in G.nodes(data=True):
            if data.get('type') == 'allele_alt':
                haplo = data.get('haplotype', '')
                score = data.get('activity_score', 'N/A')
                labels[node] = f"{haplo}\n(S:{score})"

        nx.draw_networkx_labels(G, pos_layout, labels, font_size=9, font_weight='bold')

        # E. FINAL TOUCHES
        plt.title(f"Clinical Variant Interpretation: Gene {gene_name}", fontsize=16, pad=20)
        plt.ylim(-1.5, 1.5)
        plt.axis('off')

        # Legend
        legend_elements = [
            patches.Patch(color=GraphVisualization.CLINICAL_COLORS['no_function'], label='Activity: 0.0 (PM)'),
            patches.Patch(color=GraphVisualization.CLINICAL_COLORS['decreased'], label='Activity: 0.5 (IM)'),
            patches.Patch(color=GraphVisualization.CLINICAL_COLORS['normal'], label='Activity: ≥1.0 (NM/UM)'),
            lns.Line2D([0], [0], color=GraphVisualization.CLINICAL_COLORS['reference'], marker='o', label='Reference',
                       markersize=10)
        ]
        ax.legend(handles=legend_elements, loc='lower right', title="Metabolic Status (CPIC)")

        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"📊 Plot saved to: {output_file}")

        # Note: plt.show() should generally be called by the user or main script if interactive
        # plt.show()


# --- 3. GRAPH BUILDER ---

class GenomeGraphBuilder:
    """
    Constructs a Directed Acyclic Graph (DAG) representing the pangenome of a specific gene.
    Nodes = Conserved Regions (Backbone) & Alleles (Bubbles).
    Edges = 5' -> 3' Sequence Flow.
    """

    def __init__(self, parquet_path: Union[str, Path]):
        print(f"📂 Loading Genomic Library: {parquet_path}")
        self.library = pd.read_parquet(parquet_path)

    def build_gene_graph(self, gene_name: str, output_json_path: Optional[Union[str, Path]] = None) -> Optional[
        nx.DiGraph]:
        # 1. Filter and Sort Variants
        df_gene = self.library[self.library['gene_context'] == gene_name].copy()

        if df_gene.empty:
            print(f"⚠️ No variants found for gene: {gene_name}")
            return None

        df_gene = df_gene.sort_values('POS').reset_index(drop=True)

        # Initialize Graph
        G = nx.DiGraph(name=gene_name)

        # 2. Initialization (5' Anchor)
        # Start slightly before first variant to provide context
        current_pos = max(1, df_gene['POS'].min() - 100)
        start_node_id = f"start_{current_pos}"

        G.add_node(start_node_id, type="backbone", pos=current_pos, seq="N/A")
        last_node_id = start_node_id

        print(f"🧬 Building graph for {gene_name} with {len(df_gene)} bubbles...")

        # 3. Iterate through Variants
        for idx, row in df_gene.iterrows():
            var_pos = int(row['POS'])
            ref_seq = str(row['REF'])
            alt_seq = str(row['ALT'])
            var_type = str(row['variant_type'])

            # --- A. CONSERVED PATH (Backbone) ---
            if var_pos > current_pos:
                dist = var_pos - current_pos
                inter_node_id = f"seq_{current_pos}_{var_pos}"

                # Add Backbone Node
                G.add_node(inter_node_id, type="backbone", pos=current_pos, length=dist)
                G.add_edge(last_node_id, inter_node_id)

                last_node_id = inter_node_id

            # --- B. VARIANT BUBBLE OPENING (Split) ---
            split_node = f"split_{var_pos}"
            G.add_node(split_node, type="split", pos=var_pos)
            G.add_edge(last_node_id, split_node)

            # --- C. BRANCH 1: REFERENCE ---
            ref_node = f"ref_{var_pos}"
            G.add_node(ref_node, type="allele_ref", seq=ref_seq, length=len(ref_seq))
            G.add_edge(split_node, ref_node, weight=1.0, label="reference")

            # --- D. BRANCH 2: ALTERNATIVE (Variant) ---
            alt_node = f"var_{var_pos}"

            # Safe attribute extraction
            act_score = float(row['activity_score']) if pd.notna(row.get('activity_score')) else -1.0

            features = {
                "type": "allele_alt",
                "seq": alt_seq,
                "length": len(alt_seq),
                "variant_type": var_type,
                "frameshift": bool(row['is_frameshift']) if pd.notna(row['is_frameshift']) else False,
                "star_allele": (var_type == 'STAR_ALLELE'),
                "haplotype": str(row.get('haplotype_label', 'Unknown')),
                "metabolic_function": str(row.get('metabolic_function', 'Unknown')),
                "activity_score": act_score
            }
            G.add_node(alt_node, **features)
            G.add_edge(split_node, alt_node, weight=1.0, label="variant")

            # --- E. BUBBLE CLOSING (Merge) ---
            # Merge position depends on Reference length
            len_ref_on_genome = len(ref_seq) if ref_seq != "" else 0
            merge_pos = var_pos + len_ref_on_genome
            merge_node = f"merge_{merge_pos}"

            if merge_node not in G:
                G.add_node(merge_node, type="merge", pos=merge_pos)

            G.add_edge(ref_node, merge_node)
            G.add_edge(alt_node, merge_node)

            # Update cursors
            last_node_id = merge_node
            current_pos = merge_pos

        # 4. Finalization (3' Anchor)
        end_node = f"end_{current_pos}"
        G.add_node(end_node, type="backbone_end", pos=current_pos)
        G.add_edge(last_node_id, end_node)

        print(f"✅ Graph completed: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")

        # 5. Optional JSON Export
        if output_json_path:
            self._save_to_json(G, output_json_path)

        return G

    def _save_to_json(self, G: nx.DiGraph, path: Union[str, Path]):
        """Internal helper to save graph to JSON with correct formatting."""
        data = nx.node_link_data(G, edges="links")
        data = GraphEncoders.clean_nans_for_json(data)

        with open(path, 'w') as f:
            json.dump(data, f, indent=2, cls=GraphEncoders)
        print(f"💾 Graph JSON saved to: {path}")

    def export_to_graphml(self, G: nx.DiGraph, output_path: Union[str, Path]):
        """
        Exports to GraphML for tools like Cytoscape or Gephi.
        Flattens complex attributes to strings.
        """
        G_export = G.copy()
        for node, data in G_export.nodes(data=True):
            for k, v in data.items():
                # GraphML doesn't support None/Lists/Dicts well, convert to str
                if v is None:
                    data[k] = ""
                else:
                    data[k] = str(v)

        nx.write_graphml(G_export, output_path)
        print(f"👁️ Visualizable GraphML saved to: {output_path}")

    def to_pytorch_geometric(self, G: nx.DiGraph, patient_genotype_map: Dict[str, float]):
        """
        Converts the NetworkX graph into a PyTorch Geometric Data object
        suitable for GNN input.
        """
        try:
            from torch_geometric.utils import from_networkx
            # NOTE: torch imports are local to avoid hard dependency if not used
        except ImportError:
            print("⚠️ PyTorch Geometric not installed. Skipping conversion.")
            return None

        # Mappings for One-Hot/Label Encoding
        type_map = {'backbone': 0, 'split': 1, 'merge': 2, 'allele_ref': 3, 'allele_alt': 4}
        var_map = {'SNP': 1, 'INS': 2, 'DEL': 3, 'STAR_ALLELE': 4, 'backbone': 0}

        for node, data in G.nodes(data=True):
            # 1. Feature: Gene Dosage (Zygosity)
            if data.get('type') == 'allele_alt':
                # Patient has this variant?
                data['x_dosage'] = float(patient_genotype_map.get(node, 0))
            elif data.get('type') == 'allele_ref':
                # Dosage of reference is implied by absence of variant (simplification)
                # Assumes Diploidy (Max 2)
                alt_node_id = node.replace('ref_', 'var_')
                var_dosage = patient_genotype_map.get(alt_node_id, 0)
                data['x_dosage'] = float(max(0, 2 - var_dosage))
            else:
                data['x_dosage'] = 0.0  # Backbone has no dosage variance

            # 2. Feature: Node Type
            data['x_type'] = float(type_map.get(data.get('type', 'backbone'), 0))

            # 3. Feature: Variant Type
            data['x_vtype'] = float(var_map.get(data.get('variant_type', 'backbone'), 0))

            # 4. Feature: Activity Score (The Critical Metadata)
            data['x_score'] = float(data.get('activity_score', 1.0))

        # Convert to PyG
        # Select specific features to form the Node Feature Matrix (X)
        pyg_data = from_networkx(G, group_node_attrs=['x_dosage', 'x_type', 'x_vtype', 'x_score'])
        return pyg_data


# --- EXECUTION ENTRY POINT ---
if __name__ == "__main__":
    base_dir = Path("data")
    parquet_file = base_dir / "genome_library.parquet"

    if not parquet_file.exists():
        print(f"❌ Library file not found: {parquet_file}")
    else:
        # 1. Instantiate
        builder = GenomeGraphBuilder(parquet_file)

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
            n_variants = len([n for n, d in G.nodes(data=True) if d.get('type') == 'allele_alt'])
            print(f"Variant Bubbles: {n_variants}")

            # D. Data Inspection
            print("\n--- Sample Variant Node ---")
            for n, d in G.nodes(data=True):
                if d.get('type') == 'allele_alt':
                    print(f"ID: {n}")
                    print(f"Metadata: {d}")
                    break

