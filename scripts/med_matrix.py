"""
inspect_drug_graph.py
=====================
Carga un archivo .pt de fármaco generado por library_creator_polars.py
y exporta su información como:
  - Matriz de adyacencia (CSV)
  - Tabla de nodos con features (CSV)
  - Tabla de aristas con features (CSV)
  - Resumen en consola

Uso:
    python inspect_drug_graph.py --pt ruta/al/farmaco.pt
    python inspect_drug_graph.py --pt ruta/al/farmaco.pt --output mi_carpeta
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# =============================================================================
#  NOMBRES DE FEATURES (según smiles_to_graph_complete en el script original)
# =============================================================================

NODE_FEATURE_NAMES = [
    "atomic_num_norm",  # Número atómico / 100
    "degree_raw",  # Grado (valor directo)
    # One-hot grado [0,1,2,3,4]
    "degree_0",
    "degree_1",
    "degree_2",
    "degree_3",
    "degree_4",
    # One-hot carga formal [-2,-1,0,1,2]
    "charge_-2",
    "charge_-1",
    "charge_0",
    "charge_+1",
    "charge_+2",
    # One-hot hibridación [SP, SP2, SP3]
    "hybrid_SP",
    "hybrid_SP2",
    "hybrid_SP3",
    # One-hot num Hs [0,1,2,3,4]
    "totalH_0",
    "totalH_1",
    "totalH_2",
    "totalH_3",
    "totalH_4",
    # One-hot quiralidad [UNSPECIFIED, CW, CCW]
    "chiral_unspec",
    "chiral_CW",
    "chiral_CCW",
    "is_aromatic",  # Aromaticidad (0/1)
    "mass_norm",  # Masa * 0.01
]

EDGE_FEATURE_NAMES = [
    "bond_SINGLE",
    "bond_DOUBLE",
    "bond_TRIPLE",
    "bond_AROMATIC",
    "is_conjugated",
    "is_in_ring",
    "has_stereo",
]


# =============================================================================
#  FUNCIONES DE EXPORTACIÓN
# =============================================================================


def load_graph(pt_path: Path):
    """Carga el objeto Data desde el .pt."""
    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    return data


def build_node_table(data) -> pd.DataFrame:
    """Devuelve DataFrame con un nodo por fila y sus features."""
    x = data.x.numpy()
    n_nodes, n_feats = x.shape

    # Ajustar nombres si el número de features no coincide exactamente
    feat_names = NODE_FEATURE_NAMES[:n_feats]
    if n_feats > len(feat_names):
        feat_names += [f"feat_{i}" for i in range(len(feat_names), n_feats)]

    df = pd.DataFrame(x, columns=feat_names)
    df.insert(0, "atom_idx", range(n_nodes))
    return df


def build_edge_table(data) -> pd.DataFrame:
    """Devuelve DataFrame con una arista por fila."""
    edge_index = data.edge_index.numpy()  # shape [2, num_edges]
    src = edge_index[0]
    dst = edge_index[1]

    rows = {"src": src, "dst": dst}

    if data.edge_attr is not None:
        ea = data.edge_attr.numpy()
        n_feats = ea.shape[1]
        feat_names = EDGE_FEATURE_NAMES[:n_feats]
        if n_feats > len(feat_names):
            feat_names += [f"edge_feat_{i}" for i in range(len(feat_names), n_feats)]
        for i, name in enumerate(feat_names):
            rows[name] = ea[:, i]

    return pd.DataFrame(rows)


def build_adjacency_matrix(data, weighted: bool = False) -> pd.DataFrame:
    """
    Construye la matriz de adyacencia N×N.
    Si weighted=True, el valor de cada celda es el tipo de enlace dominante
    (1=SINGLE, 2=DOUBLE, 3=TRIPLE, 4=AROMATIC), si no, binaria (0/1).
    """
    n = data.x.shape[0]
    adj = np.zeros((n, n), dtype=np.float32)

    edge_index = data.edge_index.numpy()
    src_arr = edge_index[0]
    dst_arr = edge_index[1]

    if weighted and data.edge_attr is not None:
        ea = data.edge_attr.numpy()
        for s, d, feat in zip(src_arr, dst_arr, ea):
            # bond_type one-hot → índice 1=SINGLE, 2=DOUBLE, 3=TRIPLE, 4=AROMATIC
            bond_val = np.argmax(feat[:4]) + 1 if feat[:4].sum() > 0 else 1
            adj[s, d] = bond_val
    else:
        for s, d in zip(src_arr, dst_arr):
            adj[s, d] = 1.0

    cols = [f"atom_{i}" for i in range(n)]
    return pd.DataFrame(adj, index=cols, columns=cols)


def print_summary(data, pt_path: Path):
    print("\n" + "=" * 60)
    print(f"  DRUG GRAPH SUMMARY: {pt_path.name}")
    print("=" * 60)
    print(f"  CID    : {getattr(data, 'cid', 'N/A')}")
    print(f"  Name   : {getattr(data, 'name', 'N/A')}")
    print(f"  SMILES : {getattr(data, 'smiles', 'N/A')}")
    print(f"  Nodes  : {data.x.shape[0]}  (atoms)")
    print(f"  Edges  : {data.edge_index.shape[1]}  (bonds, bidirectional)")
    print(f"  Node features : {data.x.shape[1]}")
    if data.edge_attr is not None:
        print(f"  Edge features : {data.edge_attr.shape[1]}")
    print("=" * 60 + "\n")


# =============================================================================
#  MAIN
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Inspect a drug .pt graph file")
    parser.add_argument("--pt", required=True, type=str, help="Path to the .pt file")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: same folder as .pt file)",
    )
    parser.add_argument(
        "--weighted-adj",
        action="store_true",
        help="Encode bond type in adjacency matrix (1=single,2=double,3=triple,4=aromatic)",
    )
    parser.add_argument(
        "--no-adj",
        action="store_true",
        help="Skip adjacency matrix export (useful for large molecules)",
    )
    args = parser.parse_args()

    pt_path = Path(args.pt)
    if not pt_path.exists():
        print(f"❌ File not found: {pt_path}")
        return

    out_dir = Path(args.output) if args.output else pt_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = pt_path.stem  # e.g. "2244_aspirin"

    # --- Load ---
    print(f"📂 Loading: {pt_path}")
    data = load_graph(pt_path)
    print_summary(data, pt_path)

    # --- Node table ---
    node_df = build_node_table(data)
    node_path = out_dir / f"{stem}_nodes.csv"
    node_df.to_csv(node_path, index=False)
    print(
        f"✅ Node table  → {node_path}  ({len(node_df)} atoms × {len(node_df.columns) - 1} features)"
    )

    # --- Edge table ---
    edge_df = build_edge_table(data)
    edge_path = out_dir / f"{stem}_edges.csv"
    edge_df.to_csv(edge_path, index=False)
    print(f"✅ Edge table  → {edge_path}  ({len(edge_df)} directed edges)")

    # --- Adjacency matrix ---
    if not args.no_adj:
        adj_df = build_adjacency_matrix(data, weighted=args.weighted_adj)
        adj_path = out_dir / f"{stem}_adjacency.csv"
        adj_df.to_csv(adj_path)
        mode = "weighted" if args.weighted_adj else "binary"
        print(
            f"✅ Adj matrix  → {adj_path}  ({adj_df.shape[0]}×{adj_df.shape[1]}, {mode})"
        )
    else:
        print("⏭️  Adjacency matrix skipped (--no-adj)")

    print(f"\n🎉 Done. Files saved in: {out_dir.resolve()}\n")


if __name__ == "__main__":
    main()
