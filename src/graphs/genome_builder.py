import logging
from pathlib import Path
from typing import Optional, Union, cast

import networkx as nx
import pandas as pd
import torch
from torch_geometric.data import Data

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class GenomeGraphBuilder:
    """
    Construye un Grafo Dirigido Múltiple (MultiDiGraph).
    Implementa lógica de agrupación por posición (groupby) para manejar sitios multi-alélicos.
    """

    def __init__(self, parquet_path: Union[str, Path]):
        print(f"📂 Cargando librería genómica: {parquet_path}")
        self.library = pd.read_parquet(parquet_path)

    def build_gene_graph(
        self, gene_name: str, output_json_path: Optional[Union[str, Path]] = None
    ) -> Optional[nx.MultiDiGraph]:
        # 1. Filtrar variantes del gen
        df_gene = self.library[self.library["gene_context"] == gene_name].copy()
        if df_gene.empty:
            print(f"⚠️ No hay variantes para: {gene_name}")
            return None

        # Ordenar por posición
        df_gene = df_gene.sort_values("POS").reset_index(drop=True)

        # Inicializar MultiDiGraph
        G = nx.MultiDiGraph(name=gene_name)

        # Ancla 5' inicial (Backbone start)
        current_pos = max(1, df_gene["POS"].min() - 50)
        last_node_id = f"start_{current_pos}"
        G.add_node(last_node_id, type="backbone", pos=current_pos, seq="N/A")

        print(f"🧬 Construyendo grafo para {gene_name}...")

        # 2. OPTIMIZED: Group by position without sorting for better performance
        grouped_variants = df_gene.groupby("POS", sort=False)

        for pos, group in grouped_variants:
            pos = cast(int, pos)

            # A. Conectar Backbone (del último nodo al inicio de este variante)
            if pos > current_pos:
                dist = pos - current_pos
                pre_split_id = f"backbone_{pos}"
                G.add_node(pre_split_id, type="backbone", pos=pos, length=dist)
                G.add_edge(last_node_id, pre_split_id, type="backbone_link")
                last_node_id = pre_split_id

            # B. Abrir Burbuja (Split Node único por posición)
            split_node = f"split_{pos}"
            if split_node not in G:
                G.add_node(split_node, type="split", pos=pos)

            G.add_edge(last_node_id, split_node, type="link")

            # C. Determinar punto de Cierre (Merge)
            # Simplificación: Merge ocurre tras la variante.
            # Tomamos la longitud de la Referencia del primer elemento del grupo.
            ref_seq = str(group.iloc[0]["REF"])
            merge_pos = pos + len(ref_seq)
            merge_node = f"merge_{merge_pos}"

            if merge_node not in G:
                G.add_node(merge_node, type="merge", pos=merge_pos)

            # D. Crear camino de REFERENCIA (Solo uno por sitio)
            ref_node = f"ref_{pos}"
            if ref_node not in G:
                G.add_node(ref_node, type="allele_ref", seq=ref_seq, pos=pos + 0.5)
                # Key explícito 'ref' para diferenciar en MultiGraph
                G.add_edge(
                    split_node, ref_node, key="ref", label="reference", weight=1.0
                )
                G.add_edge(ref_node, merge_node, key="ref_join", label="join")

            # E. Crear caminos de VARIANTES (Múltiples por sitio)
            for idx, row in group.iterrows():
                alt_seq = str(row["ALT"])
                var_id = f"var_{pos}_{alt_seq}"  # ID único incluyendo la secuencia

                # Metadatos farmacéuticos
                act_score = (
                    float(row["activity_score"])
                    if pd.notna(row.get("activity_score"))
                    else -1.0
                )

                # Nodo Variante
                G.add_node(
                    var_id,
                    type="allele_alt",
                    seq=alt_seq,
                    pos=pos + 0.5,
                    variant_type=row["variant_type"],
                    haplotype=str(row.get("haplotype_label", "")),
                    activity_score=act_score,
                )

                # Aristas (Split -> Var -> Merge)
                # Usamos IDs únicos en 'key' para permitir aristas paralelas
                edge_key = f"alt_{idx}"
                G.add_edge(
                    split_node, var_id, key=edge_key, label="variant", weight=1.0
                )
                G.add_edge(var_id, merge_node, key=f"{edge_key}_join", label="join")

            # Actualizar cursor
            last_node_id = merge_node
            current_pos = merge_pos

        # 3. Finalización (Ancla 3')
        end_node = f"end_{current_pos}"
        G.add_node(end_node, type="backbone_end", pos=current_pos)
        G.add_edge(last_node_id, end_node)

        print(
            f"✅ Grafo completado: {G.number_of_nodes()} nodos, {G.number_of_edges()} aristas."
        )

        return G

    def build_denoised_gene_graph(
        self,
        gene_name: str,
        output_json_path: Optional[Union[str, Path]] = None,
        only_functional: bool = True,
    ) -> Optional[nx.MultiDiGraph]:
        """
        only_functional: Si es True, filtra variantes con Activity Score == 1.0 (Benignas)
                         o que no sean exónicas/importantes.
        """
        # 1. Filtrar variantes del gen
        df_gene = self.library[self.library["gene_context"] == gene_name].copy()
        if df_gene.empty:
            print(f"⚠️ No hay variantes para: {gene_name}")
            return None

        original_count = len(df_gene)
        if only_functional:
            print("🧹 Filtrando variantes no funcionales (Ruido)...")
            # Mantenemos variantes si:
            # 1. El score NO es 1.0 (es decir, <1 disminuida/nula, o >1 aumentada)
            # 2. O es un Frameshift
            # 3. O es un Alelo Estrella (explicitamente etiquetado)
            # 4. O el score es desconocido (NaN/-1) por seguridad

            mask_functional = (
                (df_gene["activity_score"] != 1.0)
                | (df_gene["is_frameshift"] == True)
                | (df_gene["variant_type"] == "STAR_ALLELE")
                | (df_gene["activity_score"].isna())
            )
            df_gene = df_gene[mask_functional]

            print(
                f"📉 Reducción: {original_count} -> {len(df_gene)} variantes clínicamente relevantes."
            )

        if df_gene.empty:
            print("⚠️ Tras el filtrado no quedaron variantes relevantes para graficar.")
            return None

        # Ordenar por posición
        df_gene = df_gene.sort_values("POS").reset_index(drop=True)

        # Inicializar MultiDiGraph
        G = nx.MultiDiGraph(name=gene_name)

        # Ancla 5' inicial (Backbone start)
        current_pos = max(1, df_gene["POS"].min() - 50)
        last_node_id = f"start_{current_pos}"
        G.add_node(last_node_id, type="backbone", pos=current_pos, seq="N/A")

        print(f"🧬 Construyendo grafo para {gene_name}...")

        # 2. OPTIMIZED: Group by position without sorting for better performance
        grouped_variants = df_gene.groupby("POS", sort=False)

        for pos, group in grouped_variants:
            pos = cast(int, pos)

            # A. Conectar Backbone (del último nodo al inicio de este variante)
            if pos > current_pos:
                dist = pos - current_pos
                pre_split_id = f"backbone_{pos}"
                G.add_node(pre_split_id, type="backbone", pos=pos, length=dist)
                G.add_edge(last_node_id, pre_split_id, type="backbone_link")
                last_node_id = pre_split_id

            # B. Abrir Burbuja (Split Node único por posición)
            split_node = f"split_{pos}"
            if split_node not in G:
                G.add_node(split_node, type="split", pos=pos)

            G.add_edge(last_node_id, split_node, type="link")

            # C. Determinar punto de Cierre (Merge)
            # Simplificación: Merge ocurre tras la variante.
            # Tomamos la longitud de la Referencia del primer elemento del grupo.
            ref_seq = str(group.iloc[0]["REF"])
            merge_pos = pos + len(ref_seq)
            merge_node = f"merge_{merge_pos}"

            if merge_node not in G:
                G.add_node(merge_node, type="merge", pos=merge_pos)

            # D. Crear camino de REFERENCIA (Solo uno por sitio)
            ref_node = f"ref_{pos}"
            if ref_node not in G:
                G.add_node(ref_node, type="allele_ref", seq=ref_seq, pos=pos + 0.5)
                # Key explícito 'ref' para diferenciar en MultiGraph
                G.add_edge(
                    split_node, ref_node, key="ref", label="reference", weight=1.0
                )
                G.add_edge(ref_node, merge_node, key="ref_join", label="join")

            # E. Crear caminos de VARIANTES (Múltiples por sitio)
            for idx, row in group.iterrows():
                alt_seq = str(row["ALT"])
                var_id = f"var_{pos}_{alt_seq}"  # ID único incluyendo la secuencia

                # Metadatos farmacéuticos
                act_score = (
                    float(row["activity_score"])
                    if pd.notna(row.get("activity_score"))
                    else -1.0
                )

                # Nodo Variante
                G.add_node(
                    var_id,
                    type="allele_alt",
                    seq=alt_seq,
                    pos=pos + 0.5,
                    variant_type=row["variant_type"],
                    haplotype=str(row.get("haplotype_label", "")),
                    activity_score=act_score,
                )

                # Aristas (Split -> Var -> Merge)
                # Usamos IDs únicos en 'key' para permitir aristas paralelas
                edge_key = f"alt_{idx}"
                G.add_edge(
                    split_node, var_id, key=edge_key, label="variant", weight=1.0
                )
                G.add_edge(var_id, merge_node, key=f"{edge_key}_join", label="join")

            # Actualizar cursor
            last_node_id = merge_node
            current_pos = merge_pos

        # 3. Finalización (Ancla 3')
        end_node = f"end_{current_pos}"
        G.add_node(end_node, type="backbone_end", pos=current_pos)
        G.add_edge(last_node_id, end_node)

        print(
            f"✅ Grafo completado: {G.number_of_nodes()} nodos, {G.number_of_edges()} aristas."
        )

        return G

    def to_pytorch_geometric(self, G: nx.MultiDiGraph) -> Data:
        """
        Convierte el grafo NetworkX a un objeto Data de PyTorch Geometric.

        Feature Engineering (Diseño de características por nodo):
        - One-Hot Encoding del tipo de nodo (Backbone, Split, Merge, Ref, Alt).
        - Activity Score (Normalizado).
        - Longitud de la secuencia.

        Returns:
            Data: Objeto con x (features), edge_index (conectividad) y edge_attr (tipo de arista).
        """
        if G is None or G.number_of_nodes() == 0:
            print("⚠️ Grafo vacío. Retornando objeto Data vacío.")
            return Data()

        # 1. Mapeo de Nodos (String ID -> Índice Entero)
        # PyG requiere índices contiguos de 0 a N-1
        node_mapping = {node_id: i for i, node_id in enumerate(G.nodes())}

        # 2. Construcción de Features de Nodos (Matriz X)
        # Definimos dimensiones:
        # [0-5]: One-hot Type, [6]: Activity Score, [7]: Seq Length, [8]: Is Variant
        node_features = []

        # Diccionario para one-hot encoding de tipos de nodos biológicos
        type_to_idx = {
            "backbone": 0,
            "backbone_end": 0,  # Tratamos inicio/fin como backbone
            "split": 1,
            "merge": 2,
            "allele_ref": 3,
            "allele_alt": 4,
        }

        for node_id in G.nodes():
            node_data = G.nodes[node_id]
            attr_vec = [0, 0, 0, 0, 0, 0.0, 0, 0, 0]  # [0] * 9 # Vector base

            # A. One-Hot Type
            node_type = node_data.get("type", "backbone")
            type_idx = type_to_idx.get(node_type, 0)
            attr_vec[type_idx] = 1

            # B. Activity Score (Farmacología)
            # Si es backbone/ref, asumimos score 1.0 (funcionalidad normal).
            # Si es variante y tiene score, lo usamos. Si es -1 (desconocido), ponemos 1.0 o 0.5 según criterio.
            raw_score = node_data.get("activity_score", 1.0)
            # Sanitización: si es -1.0 (placeholder de tu builder), lo pasamos a 1.0 (neutral) para no romper gradientes
            attr_vec[6] = max(0.0, float(raw_score)) if raw_score != -1.0 else 1.0

            # C. Sequence Length (Estructural)
            seq = str(node_data.get("seq", ""))
            attr_vec[7] = len(seq) if seq != "N/A" else 0

            # D. Is Variant Flag (Binario explícito)
            attr_vec[8] = 1 if node_type == "allele_alt" else 0

            node_features.append(attr_vec)

        x = torch.tensor(node_features, dtype=torch.float)

        # 3. Construcción de Aristas (Edge Index & Edge Attr)
        src_nodes = []
        dst_nodes = []
        edge_attributes = []

        # Mapping para tipos de aristas
        edge_type_map = {
            "backbone_link": [1, 0, 0],
            "link": [1, 0, 0],  # Estructural
            "variant": [0, 1, 0],  # Camino alternativo (variante)
            "reference": [0, 0, 1],  # Camino de referencia
            "join": [1, 0, 0],  # Cierre de burbuja
        }

        # Iteramos sobre aristas (MultiDiGraph puede tener múltiples aristas entre u y v)
        for u, v, data in G.edges(data=True):
            src_nodes.append(node_mapping[u])
            dst_nodes.append(node_mapping[v])

            # Edge Features (Tipo de conexión)
            etype = data.get("label", "link")
            # Fallback simple si la etiqueta no está en el mapa
            feat = edge_type_map.get(etype, [1, 0, 0])
            edge_attributes.append(feat)

        edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
        edge_attr = torch.tensor(edge_attributes, dtype=torch.float)

        # 4. Creación del Objeto Data
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        # Metadatos útiles para depuración
        data.gene_name = G.name
        data.num_nodes = G.number_of_nodes()

        return data


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
