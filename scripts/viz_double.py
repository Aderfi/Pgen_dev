"""
visualize_graphs.py
====================
Visualización de grafos PyTorch Geometric para TFM.
Genera imágenes de:
  1. Grafo genómico completo de una variante
  2. Grafo de molécula (fármaco)
  3. Subgrafo de región específica del gen (varios alelos del mismo gen)

USO:
    python visualize_graphs.py

CONFIGURACIÓN: ajusta las rutas en la sección CONFIG al final del archivo.
"""

import os
from pathlib import Path

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import torch
from matplotlib.lines import Line2D
from torch_geometric.data import Data

matplotlib.rcParams["font.family"] = "DejaVu Sans"


# =============================================================================
#  PALETAS Y ESTILOS
# =============================================================================

# Colores para nodos del grafo genómico
GENOMIC_NODE_COLORS = {
    "backbone": "#4A90D9",  # azul
    "backbone_end": "#4A90D9",
    "split": "#F5A623",  # naranja
    "merge": "#F5A623",
    "allele_ref": "#7ED321",  # verde
    "allele_alt": "#D0021B",  # rojo
    "unknown": "#9B9B9B",
}

GENOMIC_NODE_LABELS = {
    "backbone": "Backbone",
    "backbone_end": "Backbone End",
    "split": "Split",
    "merge": "Merge",
    "allele_ref": "Ref Allele",
    "allele_alt": "Alt Allele",
}

# Colores para aristas del grafo genómico según edge_attr [backbone, ref, alt]
EDGE_COLORS_GENOMIC = ["#4A90D9", "#7ED321", "#D0021B"]
EDGE_LABELS_GENOMIC = ["Backbone link", "Ref path", "Alt path"]

# Colores para átomos del grafo molecular (por número atómico)
ATOM_COLORS = {
    6: "#404040",  # C  - gris oscuro
    7: "#3050F8",  # N  - azul
    8: "#FF0D0D",  # O  - rojo
    9: "#90E050",  # F  - verde claro
    15: "#FF8000",  # P  - naranja
    16: "#FFFF30",  # S  - amarillo
    17: "#1FF01F",  # Cl - verde
    35: "#A62929",  # Br - marrón
    53: "#940094",  # I  - morado
    1: "#FFFFFF",  # H  - blanco
}
DEFAULT_ATOM_COLOR = "#AAAAAA"

# Colores para aristas moleculares según bond type [SINGLE, DOUBLE, TRIPLE, AROMATIC]
BOND_COLORS = ["#888888", "#FF6600", "#CC0000", "#9932CC"]
BOND_LABELS = ["Single", "Double", "Triple", "Aromatic"]


# =============================================================================
#  UTILIDADES
# =============================================================================


def infer_node_type_genomic(feat_vec: list) -> str:
    """Infiere tipo de nodo a partir del vector de features [Backbone, SplitMerge, Ref, Alt, ...]"""
    if feat_vec[0] == 1.0:
        return "backbone"
    elif feat_vec[1] == 1.0:
        return "split"  # split/merge comparten posición, los diferenciamos después si es necesario
    elif feat_vec[2] == 1.0:
        return "allele_ref"
    elif feat_vec[3] == 1.0:
        return "allele_alt"
    return "unknown"


def infer_edge_type_genomic(attr_vec: list) -> int:
    """Devuelve índice del tipo de arista [backbone=0, ref=1, alt=2]"""
    return int(max(range(len(attr_vec)), key=lambda i: attr_vec[i]))


def pyg_to_networkx_genomic(data: Data) -> nx.DiGraph:
    """Convierte un PyG Data genómico a NetworkX con atributos de visualización."""
    G = nx.DiGraph()
    x = data.x.tolist()
    edge_index = data.edge_index.t().tolist()
    edge_attr = (
        data.edge_attr.tolist()
        if data.edge_attr is not None
        else [[1, 0, 0]] * len(edge_index)
    )

    for i, feat in enumerate(x):
        ntype = infer_node_type_genomic(feat)
        G.add_node(
            i,
            node_type=ntype,
            color=GENOMIC_NODE_COLORS.get(ntype, GENOMIC_NODE_COLORS["unknown"]),
            activity_score=feat[4],
            is_coding=feat[5],
            is_regulatory=feat[6],
            is_splicing=feat[7],
            is_intergenic=feat[8],
        )

    for (src, dst), attr in zip(edge_index, edge_attr):
        etype = infer_edge_type_genomic(attr)
        G.add_edge(src, dst, edge_type=etype, color=EDGE_COLORS_GENOMIC[etype])

    return G


def pyg_to_networkx_molecular(data: Data) -> nx.Graph:
    """Convierte un PyG Data molecular a NetworkX con colores por tipo de átomo/enlace."""
    G = nx.Graph()
    x = data.x.tolist()
    edge_index = data.edge_index.t().tolist()
    edge_attr = data.edge_attr.tolist() if data.edge_attr is not None else []

    for i, feat in enumerate(x):
        atomic_num = round(feat[0] * 100)  # feat[0] = atomic_num / 100
        G.add_node(
            i,
            atomic_num=atomic_num,
            color=ATOM_COLORS.get(atomic_num, DEFAULT_ATOM_COLOR),
        )

    seen_edges = set()
    for idx, (src, dst) in enumerate(edge_index):
        key = tuple(sorted([src, dst]))
        if key in seen_edges:
            continue
        seen_edges.add(key)
        attr = edge_attr[idx] if idx < len(edge_attr) else [1, 0, 0, 0, 0, 0, 0]
        bond_type = int(max(range(4), key=lambda i: attr[i]))
        G.add_edge(src, dst, bond_type=bond_type, color=BOND_COLORS[bond_type])

    return G


# =============================================================================
#  VISUALIZACIÓN: GRAFO GENÓMICO
# =============================================================================


def plot_genomic_graph(
    data: Data, title: str = None, save_path: str = None, figsize=(14, 7)
):
    """Visualiza un grafo genómico con layout jerárquico."""
    G = pyg_to_networkx_genomic(data)

    # Layout: intentamos jerárquico, si no funciona usamos spring
    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    except Exception:
        # Fallback: layout manual por capas según tipo de nodo
        node_types = nx.get_node_attributes(G, "node_type")
        layer_order = [
            "backbone",
            "split",
            "allele_ref",
            "allele_alt",
            "merge",
            "backbone_end",
            "unknown",
        ]
        layer_map = {t: i for i, t in enumerate(layer_order)}

        layers = {}
        for node, ntype in node_types.items():
            layer = layer_map.get(ntype, 3)
            layers.setdefault(layer, []).append(node)

        pos = {}
        for layer, nodes in layers.items():
            for j, node in enumerate(nodes):
                x_offset = j - len(nodes) / 2
                pos[node] = (layer * 2.5, x_offset * 1.5)

    node_colors = [G.nodes[n].get("color", "#AAAAAA") for n in G.nodes()]
    edge_colors = [G.edges[e].get("color", "#888888") for e in G.edges()]
    node_sizes = [
        700 if G.nodes[n].get("node_type") in ("allele_ref", "allele_alt") else 500
        for n in G.nodes()
    ]

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("#F8F9FA")
    fig.patch.set_facecolor("#FFFFFF")

    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        edge_color=edge_colors,
        arrows=True,
        arrowsize=15,
        width=1.8,
        connectionstyle="arc3,rad=0.08",
        alpha=0.85,
    )

    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.95,
        linewidths=1.2,
        edgecolors="#333333",
    )

    # Labels simples: índice + tipo corto
    labels = {}
    type_abbrev = {
        "backbone": "BB",
        "backbone_end": "BB",
        "split": "SP",
        "merge": "MG",
        "allele_ref": "REF",
        "allele_alt": "ALT",
        "unknown": "?",
    }
    for n in G.nodes():
        ntype = G.nodes[n].get("node_type", "unknown")
        labels[n] = type_abbrev.get(ntype, "?")
    nx.draw_networkx_labels(
        G, pos, labels, ax=ax, font_size=7, font_color="white", font_weight="bold"
    )

    # Leyenda nodos
    node_legend = [
        mpatches.Patch(color=GENOMIC_NODE_COLORS[k], label=GENOMIC_NODE_LABELS[k])
        for k in ["backbone", "split", "allele_ref", "allele_alt"]
    ]
    # Leyenda aristas
    edge_legend = [
        Line2D(
            [0],
            [0],
            color=EDGE_COLORS_GENOMIC[i],
            linewidth=2,
            label=EDGE_LABELS_GENOMIC[i],
        )
        for i in range(3)
    ]

    ax.legend(
        handles=node_legend + edge_legend,
        loc="upper right",
        fontsize=8,
        framealpha=0.9,
        title="Legend",
        title_fontsize=9,
    )

    variant_name = getattr(data, "variant_name", "")
    ax.set_title(
        title or f"Genomic Graph — {variant_name}",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_axis_off()

    # Info box
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    n_alts = sum(1 for n in G.nodes() if G.nodes[n].get("node_type") == "allele_alt")
    info = f"Nodes: {n_nodes}  |  Edges: {n_edges}  |  Alt alleles: {n_alts}"
    fig.text(0.5, 0.01, info, ha="center", fontsize=9, color="#555555")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"   ✅ Saved: {save_path}")
    else:
        plt.show()
    plt.close()


# =============================================================================
#  VISUALIZACIÓN: GRAFO MOLECULAR
# =============================================================================

# Mapa número atómico → símbolo (solo los más comunes)
ATOMIC_SYMBOLS = {
    1: "H",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    15: "P",
    16: "S",
    17: "Cl",
    35: "Br",
    53: "I",
}


def plot_molecular_graph(
    data: Data, title: str = None, save_path: str = None, figsize=(12, 9)
):
    """Visualiza un grafo molecular con layout spring."""
    G = pyg_to_networkx_molecular(data)

    pos = nx.spring_layout(G, seed=42, k=1.5)

    node_colors = [G.nodes[n].get("color", DEFAULT_ATOM_COLOR) for n in G.nodes()]
    [G.edges[e].get("color", "#888888") for e in G.edges()]

    # Tamaño según atomic num (átomos más pesados = más grandes)
    node_sizes = []
    for n in G.nodes():
        anum = G.nodes[n].get("atomic_num", 6)
        node_sizes.append(300 + anum * 8)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("#F8F9FA")
    fig.patch.set_facecolor("#FFFFFF")

    # Aristas con grosor según tipo de enlace
    bond_widths = {0: 1.5, 1: 3.0, 2: 4.5, 3: 2.0}
    for src, dst, attrs in G.edges(data=True):
        btype = attrs.get("bond_type", 0)
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=[(src, dst)],
            ax=ax,
            edge_color=[attrs.get("color", "#888888")],
            width=bond_widths.get(btype, 1.5),
            alpha=0.8,
        )

    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.95,
        linewidths=1.0,
        edgecolors="#333333",
    )

    # Labels: símbolo del átomo
    labels = {
        n: ATOMIC_SYMBOLS.get(G.nodes[n].get("atomic_num", 6), "?") for n in G.nodes()
    }
    nx.draw_networkx_labels(
        G, pos, labels, ax=ax, font_size=8, font_color="white", font_weight="bold"
    )

    # Leyenda
    atom_types_present = set(G.nodes[n].get("atomic_num", 6) for n in G.nodes())
    atom_legend = [
        mpatches.Patch(
            color=ATOM_COLORS.get(a, DEFAULT_ATOM_COLOR),
            label=ATOMIC_SYMBOLS.get(a, f"Z={a}"),
        )
        for a in sorted(atom_types_present)
        if a in ATOM_COLORS
    ]

    bond_types_present = set(G.edges[e].get("bond_type", 0) for e in G.edges())
    bond_legend = [
        Line2D([0], [0], color=BOND_COLORS[i], linewidth=2.5, label=BOND_LABELS[i])
        for i in sorted(bond_types_present)
    ]

    ax.legend(
        handles=atom_legend + bond_legend,
        loc="upper right",
        fontsize=8,
        framealpha=0.9,
        title="Atoms / Bonds",
        title_fontsize=9,
    )

    drug_name = getattr(data, "name", "") or ""
    smiles = getattr(data, "smiles", "") or ""
    ax.set_title(
        title or f"Molecular Graph — {drug_name}",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_axis_off()

    n_atoms = G.number_of_nodes()
    n_bonds = G.number_of_edges()
    info = f"Atoms: {n_atoms}  |  Bonds: {n_bonds}"
    if smiles:
        smiles_short = smiles[:80] + "…" if len(smiles) > 80 else smiles
        info += f"\nSMILES: {smiles_short}"
    fig.text(0.5, 0.01, info, ha="center", fontsize=8, color="#555555")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"   ✅ Saved: {save_path}")
    else:
        plt.show()
    plt.close()


# =============================================================================
#  VISUALIZACIÓN: RECONSTRUCCIÓN MOLECULAR (RDKit 2D + grafo PyG)
# =============================================================================


def _rdkit_2d_to_ax(mol, ax, data: Data, highlight_features: bool = True):
    """
    Dibuja la molécula 2D de RDKit sobre un Axes de Matplotlib.
    Colorea cada átomo según sus features (convención CPK).
    Usa la API oficial de rdMolDraw2D (compatible con RDKit >= 2022).
    """
    from io import BytesIO

    import numpy as np
    from PIL import Image as PILImage
    from rdkit.Chem.Draw import rdMolDraw2D

    # Construir mapa átomo → color CPK desde las features del PyG
    atom_col_map = {}
    atom_radii = {}
    if highlight_features and data.x is not None:
        for i, feat in enumerate(data.x.tolist()):
            atomic_num = round(feat[0] * 100)
            hex_col = ATOM_COLORS.get(atomic_num, DEFAULT_ATOM_COLOR).lstrip("#")
            r, g, b = (int(hex_col[j : j + 2], 16) / 255.0 for j in (0, 2, 4))
            atom_col_map[i] = [r, g, b]
            atom_radii[i] = 0.3

    drawer = rdMolDraw2D.MolDraw2DCairo(900, 700)

    # Opciones compatibles con la API actual (sin atomLabelFontSize)
    opts = drawer.drawOptions()
    opts.addStereoAnnotation = True
    opts.addAtomIndices = False
    opts.bondLineWidth = 2.5

    if atom_col_map:
        # DrawMoleculeWithHighlights(mol, legend, atom_col_map, bond_col_map,
        #                            atom_radii, linewidth_multipliers [, confId])
        drawer.DrawMoleculeWithHighlights(
            mol,
            "",
            {i: list(c) for i, c in atom_col_map.items()},
            {},  # bond colours vacío → RDKit usa colores por tipo de enlace
            atom_radii,
            {},  # linewidth multipliers
        )
    else:
        drawer.DrawMolecule(mol)

    drawer.FinishDrawing()
    png_bytes = drawer.GetDrawingText()

    img = PILImage.open(BytesIO(png_bytes))
    ax.imshow(np.array(img))
    ax.set_axis_off()


def plot_molecule_reconstructed(
    data: Data, title: str = None, save_path: str = None, figsize=(20, 9)
):
    """
    Figura con dos paneles:
      Izquierda — reconstrucción 2D de la molécula con RDKit (a partir del SMILES)
      Derecha   — grafo PyG (átomos como nodos, enlaces como aristas)

    Requiere: rdkit, Pillow
    """
    from rdkit import Chem
    from rdkit.Chem import rdDepictor

    smiles = getattr(data, "smiles", None)
    if not smiles:
        print("   ⚠️  No SMILES found in data object — skipping reconstruction.")
        return

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"   ⚠️  RDKit could not parse SMILES: {smiles}")
        return

    rdDepictor.Compute2DCoords(mol)

    drug_name = getattr(data, "name", "") or ""
    cid = getattr(data, "cid", "") or ""

    fig, (ax_mol, ax_pyg) = plt.subplots(
        1, 2, figsize=figsize, gridspec_kw={"width_ratios": [1.1, 1]}
    )
    fig.patch.set_facecolor("#FFFFFF")

    # ── Panel izquierdo: RDKit 2D ──────────────────────────────────────────────
    ax_mol.set_facecolor("#FAFAFA")
    _rdkit_2d_to_ax(mol, ax_mol, data, highlight_features=True)
    ax_mol.set_title(
        "2D Structure  (RDKit)", fontsize=11, fontweight="bold", pad=8, color="#333"
    )

    # Leyenda de átomos presentes
    if data.x is not None:
        present_atoms = {round(f[0] * 100) for f in data.x.tolist()}
        atom_legend = [
            mpatches.Patch(
                color=ATOM_COLORS.get(a, DEFAULT_ATOM_COLOR),
                label=ATOMIC_SYMBOLS.get(a, f"Z={a}"),
            )
            for a in sorted(present_atoms)
            if a in ATOM_COLORS
        ]
        ax_mol.legend(
            handles=atom_legend,
            loc="lower left",
            fontsize=8,
            framealpha=0.85,
            title="Atoms",
            title_fontsize=8,
        )

    # ── Panel derecho: grafo PyG ───────────────────────────────────────────────
    ax_pyg.set_facecolor("#F8F9FA")
    G = pyg_to_networkx_molecular(data)
    pos = nx.spring_layout(G, seed=42, k=1.8)

    node_colors = [G.nodes[n].get("color", DEFAULT_ATOM_COLOR) for n in G.nodes()]
    node_sizes = [300 + G.nodes[n].get("atomic_num", 6) * 8 for n in G.nodes()]

    bond_widths = {0: 1.5, 1: 3.2, 2: 5.0, 3: 2.2}
    for src, dst, attrs in G.edges(data=True):
        btype = attrs.get("bond_type", 0)
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=[(src, dst)],
            ax=ax_pyg,
            edge_color=[attrs.get("color", "#888888")],
            width=bond_widths.get(btype, 1.5),
            alpha=0.82,
        )

    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax_pyg,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.95,
        linewidths=1.0,
        edgecolors="#333333",
    )

    labels = {
        n: ATOMIC_SYMBOLS.get(G.nodes[n].get("atomic_num", 6), "?") for n in G.nodes()
    }
    nx.draw_networkx_labels(
        G, pos, labels, ax=ax_pyg, font_size=8, font_color="white", font_weight="bold"
    )

    # Leyenda enlaces
    bond_types_present = sorted({G.edges[e].get("bond_type", 0) for e in G.edges()})
    bond_legend = [
        Line2D([0], [0], color=BOND_COLORS[i], linewidth=2.5, label=BOND_LABELS[i])
        for i in bond_types_present
    ]
    ax_pyg.legend(
        handles=bond_legend,
        loc="lower left",
        fontsize=8,
        framealpha=0.85,
        title="Bond type",
        title_fontsize=8,
    )

    ax_pyg.set_title(
        "PyG Graph  (atoms as nodes)",
        fontsize=11,
        fontweight="bold",
        pad=8,
        color="#333",
    )
    ax_pyg.set_axis_off()

    # ── Título global y footer ─────────────────────────────────────────────────
    header = title or f"Molecular Reconstruction — {drug_name}"
    if cid:
        header += f"  (CID: {cid})"
    fig.suptitle(header, fontsize=14, fontweight="bold", y=1.01)

    n_atoms = G.number_of_nodes()
    n_bonds = G.number_of_edges()
    smiles_short = smiles[:100] + "…" if len(smiles) > 100 else smiles
    fig.text(
        0.5,
        -0.01,
        f"Atoms: {n_atoms}  |  Bonds: {n_bonds}  |  SMILES: {smiles_short}",
        ha="center",
        fontsize=8,
        color="#666666",
    )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"   ✅ Saved: {save_path}")
    else:
        plt.show()
    plt.close()


# =============================================================================
#  VISUALIZACIÓN: SUBGRAFO REGIONAL (varios alelos del mismo gen)
# =============================================================================


def plot_gene_subgraph(
    pt_files: list, gene_name: str, save_path: str = None, figsize=(18, 10)
):
    """
    Crea un subgrafo combinando múltiples variantes de un mismo gen.
    Cada variante es un subgrafo con sus propios nodos; los backbones
    se fusionan visualmente en una 'columna vertebral' compartida.
    """
    fig, axes = plt.subplots(1, len(pt_files), figsize=figsize, squeeze=False)
    fig.suptitle(
        f"Gene Subgraph — {gene_name}\n({len(pt_files)} variants)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig.patch.set_facecolor("#FFFFFF")

    for idx, (pt_path, ax) in enumerate(zip(pt_files, axes[0])):
        data = torch.load(pt_path, map_location="cpu", weights_only=False)
        G = pyg_to_networkx_genomic(data)

        try:
            pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
        except Exception:
            pos = nx.spring_layout(G, seed=idx * 7)

        node_colors = [G.nodes[n].get("color", "#AAAAAA") for n in G.nodes()]
        edge_colors = [G.edges[e].get("color", "#888888") for e in G.edges()]

        ax.set_facecolor("#F8F9FA")
        nx.draw_networkx_edges(
            G,
            pos,
            ax=ax,
            edge_color=edge_colors,
            arrows=True,
            arrowsize=12,
            width=1.5,
            connectionstyle="arc3,rad=0.08",
            alpha=0.8,
        )
        nx.draw_networkx_nodes(
            G,
            pos,
            ax=ax,
            node_color=node_colors,
            node_size=400,
            alpha=0.95,
            linewidths=1.0,
            edgecolors="#333333",
        )

        variant = getattr(data, "variant_name", Path(pt_path).stem)
        n_alts = sum(
            1 for n in G.nodes() if G.nodes[n].get("node_type") == "allele_alt"
        )
        ax.set_title(
            f"{variant}\n({n_alts} alt allele{'s' if n_alts != 1 else ''})",
            fontsize=8,
            pad=4,
        )
        ax.set_axis_off()

    # Leyenda global (solo en la última subgráfica)
    node_legend = [
        mpatches.Patch(color=GENOMIC_NODE_COLORS[k], label=GENOMIC_NODE_LABELS[k])
        for k in ["backbone", "split", "allele_ref", "allele_alt"]
    ]
    axes[0][-1].legend(
        handles=node_legend, loc="lower right", fontsize=7, framealpha=0.9
    )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"   ✅ Saved: {save_path}")
    else:
        plt.show()
    plt.close()


# =============================================================================
#  GRAFO DE REFERENCIA (genoma base sin polimorfismos)
# =============================================================================


def build_reference_graph(gene_name: str, pos: int = 1000) -> Data:
    """
    Construye un grafo de referencia mínimo que representa el genoma base:
    start → backbone → split → ref_allele → merge → end
    Sin nodos alt, sirve como contraste visual frente a los polimorfismos.
    """
    # x: [Backbone, SplitMerge, Ref, Alt, Score, Coding, Regulatory, Splicing, Intergenic]
    x = torch.tensor(
        [
            [1, 0, 0, 0, 0, 0, 0, 0, 0],  # 0: start (backbone)
            [1, 0, 0, 0, 0, 0, 0, 0, 0],  # 1: bb_pos (backbone)
            [0, 1, 0, 0, 0, 0, 0, 0, 0],  # 2: split
            [0, 0, 1, 0, 0, 0, 0, 0, 0],  # 3: ref_allele
            [0, 1, 0, 0, 0, 0, 0, 0, 0],  # 4: merge
            [1, 0, 0, 0, 0, 0, 0, 0, 0],  # 5: end (backbone_end)
        ],
        dtype=torch.float32,
    )

    # edge_index: [backbone, backbone, link, ref, join, backbone_link]
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 4],
            [1, 2, 3, 4, 5],
        ],
        dtype=torch.long,
    )

    # edge_attr: [backbone=1,0,0] para backbone links, [0,1,0] para ref path
    edge_attr = torch.tensor(
        [
            [1, 0, 0],  # backbone
            [1, 0, 0],  # link
            [0, 1, 0],  # ref path
            [0, 1, 0],  # join (ref)
            [1, 0, 0],  # backbone
        ],
        dtype=torch.float32,
    )

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.variant_name = f"{gene_name} — Reference (no variants)"
    return data


# =============================================================================
#  VISUALIZACIÓN: TODOS LOS POLIMORFISMOS DE UN GEN (panel completo)
# =============================================================================


def plot_all_variants(
    pt_files: list,
    gene_name: str,
    with_reference: bool,
    save_path: str = None,
    max_cols: int = 4,
):
    """
    Panel con un subplot por cada variante del gen.
    Si with_reference=True, el primer subplot es el grafo de referencia.
    """
    all_data = []
    if with_reference:
        all_data.append(("Reference", build_reference_graph(gene_name)))
    for pt_path in pt_files:
        data = torch.load(pt_path, map_location="cpu", weights_only=False)
        label = getattr(data, "variant_name", Path(pt_path).stem)
        all_data.append((label, data))

    n_total = len(all_data)
    n_cols = min(n_total, max_cols)
    n_rows = (n_total + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5.5 * n_cols, 5 * n_rows), squeeze=False
    )
    fig.suptitle(
        f"Gene: {gene_name}  —  All variants ({n_total - int(with_reference)} polymorphisms"
        + ("  +  reference)" if with_reference else ")"),
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    fig.patch.set_facecolor("#FFFFFF")

    for idx, (label, data) in enumerate(all_data):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]
        G = pyg_to_networkx_genomic(data)

        try:
            pos_layout = nx.nx_agraph.graphviz_layout(G, prog="dot")
        except Exception:
            pos_layout = nx.spring_layout(G, seed=idx * 7)

        is_ref = with_reference and idx == 0
        node_colors = [
            "#B0C4DE" if is_ref else G.nodes[n].get("color", "#AAAAAA")
            for n in G.nodes()
        ]
        edge_colors = [G.edges[e].get("color", "#888888") for e in G.edges()]

        ax.set_facecolor("#F8F9FA" if not is_ref else "#EEF4FF")
        nx.draw_networkx_edges(
            G,
            pos_layout,
            ax=ax,
            edge_color=edge_colors,
            arrows=True,
            arrowsize=10,
            width=1.4,
            connectionstyle="arc3,rad=0.08",
            alpha=0.8,
        )
        nx.draw_networkx_nodes(
            G,
            pos_layout,
            ax=ax,
            node_color=node_colors,
            node_size=350,
            alpha=0.95,
            linewidths=1.0,
            edgecolors="#333333",
        )

        n_alts = sum(
            1 for n in G.nodes() if G.nodes[n].get("node_type") == "allele_alt"
        )
        subtitle = f"{label}" + (
            f"\n({n_alts} alt allele{'s' if n_alts != 1 else ''})"
            if not is_ref
            else "\n(no alt alleles)"
        )
        ax.set_title(subtitle, fontsize=7.5, pad=4)
        ax.set_axis_off()

    # Ocultar subplots sobrantes
    for idx in range(n_total, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].set_visible(False)

    # Leyenda global
    node_legend = [
        mpatches.Patch(color=GENOMIC_NODE_COLORS[k], label=GENOMIC_NODE_LABELS[k])
        for k in ["backbone", "split", "allele_ref", "allele_alt"]
    ]
    edge_legend = [
        Line2D(
            [0],
            [0],
            color=EDGE_COLORS_GENOMIC[i],
            linewidth=2,
            label=EDGE_LABELS_GENOMIC[i],
        )
        for i in range(3)
    ]
    fig.legend(
        handles=node_legend + edge_legend,
        loc="lower center",
        ncol=7,
        fontsize=8,
        framealpha=0.9,
        bbox_to_anchor=(0.5, -0.02),
    )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"   ✅ Saved: {save_path}")
    else:
        plt.show()
    plt.close()


# =============================================================================
#  MAIN — ARGUMENTOS CLI
# =============================================================================


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualize PyG genomic and/or molecular graphs for TFM presentation.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  # Only drug
  python visualize_graphs.py --drug src/library/drugs/1775_phenytoin.pt

  # Only gene (with reference baseline)
  python visualize_graphs.py --gene src/library/gene_graphs/CYP17A1 --reference

  # Both
  python visualize_graphs.py --drug src/library/drugs/1775_phenytoin.pt \\
                              --gene src/library/gene_graphs/CYP17A1 --reference

  # Custom output dir and column layout
  python visualize_graphs.py --gene src/library/gene_graphs/CYP2D6 \\
                              --output figures/cyp2d6 --cols 5
        """,
    )
    parser.add_argument(
        "--drug",
        type=Path,
        default=None,
        metavar="FILE.pt",
        help="Path to a drug .pt file (molecular graph).",
    )
    parser.add_argument(
        "--gene",
        type=Path,
        default=None,
        metavar="GENE_DIR",
        help="Path to a gene folder containing one .pt per polymorphism.\n"
        "The folder name is used as the gene name.",
    )
    parser.add_argument(
        "--reference",
        action="store_true",
        help="Include a reference (no-variant) graph as the first panel\n"
        "when plotting a gene. Useful as a visual baseline.",
    )
    parser.add_argument(
        "--single",
        type=Path,
        default=None,
        metavar="VARIANT.pt",
        help="Plot a single genomic variant .pt file in detail.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures"),
        metavar="OUTPUT_DIR",
        help="Directory where figures will be saved (default: ./figures).",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=4,
        metavar="N",
        help="Max columns in the gene panel grid (default: 4).",
    )
    parser.add_argument(
        "--reconstruct",
        action="store_true",
        help="For --drug: also generate a dual figure with the RDKit 2D\n"
        "structure side-by-side with the PyG graph.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively instead of saving to disk.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not any([args.drug, args.gene, args.single]):
        import sys

        print(
            "❌  Please provide at least one of: --drug, --gene, --single\n"
            "    Run with --help for usage examples."
        )
        sys.exit(1)

    args.output.mkdir(parents=True, exist_ok=True)
    save = not args.show  # si --show, no guardamos

    # ── 1. Molécula ──────────────────────────────────────────────────────────
    if args.drug:
        if not args.drug.exists():
            print(f"❌  Drug file not found: {args.drug}")
        else:
            print(f"\n💊 Plotting molecular graph: {args.drug.name}")
            data_drug = torch.load(args.drug, map_location="cpu", weights_only=False)
            drug_name = getattr(data_drug, "name", args.drug.stem) or args.drug.stem
            out = str(args.output / f"mol_{args.drug.stem}.png") if save else None
            plot_molecular_graph(
                data_drug, title=f"Molecular Graph — {drug_name}", save_path=out
            )

            if args.reconstruct:
                print("   🔬 Reconstructing molecule with RDKit...")
                out_rec = (
                    str(args.output / f"mol_{args.drug.stem}_reconstructed.png")
                    if save
                    else None
                )
                plot_molecule_reconstructed(
                    data_drug,
                    title=f"Molecular Reconstruction — {drug_name}",
                    save_path=out_rec,
                )

    # ── 2. Variante individual ────────────────────────────────────────────────
    if args.single:
        if not args.single.exists():
            print(f"❌  Variant file not found: {args.single}")
        else:
            print(f"\n🧬 Plotting single genomic variant: {args.single.name}")
            data_gen = torch.load(args.single, map_location="cpu", weights_only=False)
            out = str(args.output / f"variant_{args.single.stem}.png") if save else None
            plot_genomic_graph(
                data_gen, title=f"Genomic Graph — {args.single.stem}", save_path=out
            )

    # ── 3. Gen completo (panel todos los polimorfismos) ───────────────────────
    if args.gene:
        gene_dir = args.gene
        if not gene_dir.is_dir():
            print(f"❌  Gene directory not found: {gene_dir}")
        else:
            gene_name = gene_dir.name
            pt_files = sorted(gene_dir.glob("*.pt"))
            if not pt_files:
                print(f"⚠️  No .pt files found in {gene_dir}")
            else:
                print(
                    f"\n🧬 Plotting all {len(pt_files)} variants of {gene_name}"
                    + (" + reference baseline" if args.reference else "")
                )
                out = str(args.output / f"gene_{gene_name}_all.png") if save else None
                plot_all_variants(
                    pt_files=[str(p) for p in pt_files],
                    gene_name=gene_name,
                    with_reference=args.reference,
                    save_path=out,
                    max_cols=args.cols,
                )

    if save:
        print(f"\n✅ Done. Figures saved in: {args.output.resolve()}")
