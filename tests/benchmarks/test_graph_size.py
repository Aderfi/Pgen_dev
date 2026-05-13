from pathlib import Path

import torch


def analyze_graph_directory(directory_path: Path, recursive: bool = False):
    """
    Parses a directory to calculate total and average graph statistics.
    :param recursive: Set to True for nested folder structures (e.g., Gene/Variant).
    """
    print(f"Analyzing: {directory_path}...")

    # Selection pattern: recursive (**/) or shallow
    pattern = "**/*.pt" if recursive else "*.pt"
    file_list = list(directory_path.glob(pattern))

    total_nodes = 0
    total_edges = 0
    node_feats = 0
    edge_feats = 0
    graph_count = len(file_list)

    if graph_count == 0:
        print(f"Warning: No .pt files found in {directory_path}")
        return None

    for f in file_list:
        # map_location='cpu' is vital to avoid filling up GPU during analysis
        data = torch.load(f, weights_only=False, map_location='cpu')

        total_nodes += data.num_nodes
        total_edges += data.num_edges

        # Update feature dimensions (only needs to be done once, but kept for robustness)
        if node_feats == 0 and hasattr(data, 'x') and data.x is not None:
            node_feats = data.x.shape[1]
        if edge_feats == 0 and hasattr(data, 'edge_attr') and data.edge_attr is not None:
            edge_feats = data.edge_attr.shape[1]

    return {
        "count": graph_count,
        "total_n": total_nodes,
        "total_e": total_edges,
        "avg_n": total_nodes / graph_count,
        "avg_e": total_edges / graph_count,
        "n_feat": node_feats,
        "e_feat": edge_feats
    }

# --- Estimation Logic (Same as previous, included for completeness) ---
def estimate_vram_usage(datasets, model_config, safety_margin=1.3):
    total_static_bytes = 0
    total_act_bytes = 0
    total_nodes = sum(d['total_n'] for d in datasets)
    total_edges = sum(d['total_e'] for d in datasets)

    for d in datasets:
        node_mem = d['total_n'] * d['n_feat'] * 4
        edge_feat_mem = d['total_e'] * d['e_feat'] * 4
        adj_mem = d['total_e'] * 2 * 8 # Int64 edge index
        total_static_bytes += (node_mem + edge_feat_mem + adj_mem)

    layers = [datasets[0]['n_feat']] + model_config['hidden_dims']
    heads = model_config['heads']

    for i in range(len(layers) - 1):
        # GATv2 materializes node projections and edge attention scores
        node_acts = total_nodes * layers[i+1] * heads * 4
        edge_acts = total_edges * heads * 4 # Attention coefficients per edge
        total_act_bytes += (node_acts + edge_acts)

    total_params = sum((layers[i] * layers[i+1] * heads) for i in range(len(layers)-1))
    training_overhead = total_params * 4 * 4 # Adam (3) + Gradients (1)

    grand_total_gb = (total_static_bytes + total_act_bytes + training_overhead) * safety_margin / 1e9

    return {
        "Static Data (GB)": total_static_bytes / 1e9,
        "Activations (GB)": total_act_bytes / 1e9,
        "Total Estimated VRAM (GB)": grand_total_gb
    }

# --- Execution ---
# Using absolute paths or ensuring relative paths match your project root
drug_path = Path("src/library/drugs/")
variant_path = Path("src/library/gene_graphs/")

# KEY FIX: Set recursive=True for the variant/gene structure
drug_stats = analyze_graph_directory(drug_path, recursive=False)
variant_stats = analyze_graph_directory(variant_path, recursive=True)

if drug_stats and variant_stats:
    model_cfg = {"hidden_dims": [64, 64], "heads": 8}
    report = estimate_vram_usage([drug_stats, variant_stats], model_cfg)

    print("\n" + "="*30)
    print("VRAM PREDICTION REPORT")
    print("="*30)
    for k, v in report.items():
        print(f"{k:<25} : {v:>7.2f}")
else:
    print("\nError: Could not find one or both datasets. Check paths and recursion.")
