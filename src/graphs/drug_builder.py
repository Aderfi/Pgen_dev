import io
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import rdkit.Chem.Descriptors as Dcrp
import rdkit.Chem.rdDepictor
import torch
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Atom, rdchem
from rdkit.Chem.Draw import rdMolDraw2D
from torch_geometric.data import Data

# Definición de valores permitidos para las características atómicas y de enlace


def one_hot_encoding(value, options, allow_unknown=True):
    if value not in options:
        if allow_unknown:
            # Todo ceros y un 1 al final para 'Desconocido'
            return [0] * len(options) + [1]
        else:
            # Comportamiento fallback: Asumir el último o el primero (según preferencia)
            # Aquí optamos por devolver todo ceros si no es estricto
            return [0] * len(options)

    encoded = [0] * len(options)
    if allow_unknown:
        encoded.append(0)  # Espacio para el bit de 'unknown'

    encoded[options.index(value)] = 1
    return encoded


def smiles_to_graph_complete(smiles) -> None | Data:
    # Definición de valores permitidos
    ALLOWED_DEGREES = [0, 1, 2, 3, 4]
    ALLOWED_CHARGES = [-2, -1, 0, 1, 2]
    ALLOWED_HYBRIDIZATIONS = [
        rdchem.HybridizationType.SP,
        rdchem.HybridizationType.SP2,
        rdchem.HybridizationType.SP3,
        rdchem.HybridizationType.SP3D,
        rdchem.HybridizationType.SP3D2,
    ]
    ALLOWED_TOTAL_HS = [0, 1, 2, 3, 4]
    ALLOWED_CHIRAL_TAGS = [
        rdchem.ChiralType.CHI_UNSPECIFIED,
        rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    ]
    ALLOWED_BOND_TYPES = [
        rdchem.BondType.SINGLE,
        rdchem.BondType.DOUBLE,
        rdchem.BondType.TRIPLE,
        rdchem.BondType.AROMATIC,
    ]

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # -----------------------------------------------------
    # 1. NODOS (Átomos)
    # -----------------------------------------------------
    """
    Features del átomo

        1- Número atómico (Z) normalizado (dividido por 100)
        2- Grado del átomo (one-hot) [0,1,2,3,4]
        3- Carga formal (one-hot) [-2,-1,0,1,2]
        4- Hibridación (one-hot) [SP, SP2, SP3]
        5- Número total de Hs (one-hot) [0,1,2,3,4]
        6- Etiqueta quiral (one-hot) [No especificado, CW (R), CCW (S)]
        7- Aromaticidad (1 si es aromático, 0 si no)
        8- Masa atómica (normalizada multiplicando por 0.01)

    """
    atom_features = []

    assert type(mol) is rdchem.Mol

    for atom in mol.GetAtoms():
        atom = cast(Atom, atom)
        features = []

        # 1. Z Normalizado
        features.append(atom.GetAtomicNum() / 100.0)

        # 2. Features categóricas (One-Hot)
        # Usamos allow_unknown=False si estamos seguros de la lista,
        # o True si esperamos cosas raras.
        features += one_hot_encoding(
            atom.GetDegree(), ALLOWED_DEGREES, allow_unknown=True
        )
        features += one_hot_encoding(
            atom.GetFormalCharge(), ALLOWED_CHARGES, allow_unknown=True
        )
        features += one_hot_encoding(
            atom.GetHybridization(), ALLOWED_HYBRIDIZATIONS, allow_unknown=True
        )
        features += one_hot_encoding(
            atom.GetTotalNumHs(), ALLOWED_TOTAL_HS, allow_unknown=True
        )
        features += one_hot_encoding(
            atom.GetChiralTag(), ALLOWED_CHIRAL_TAGS, allow_unknown=False
        )

        # 3. Booleanos y escalares
        features.append(1.0 if atom.GetIsAromatic() else 0.0)
        features.append(atom.GetMass() * 0.01)

        atom_features.append(features)

    x = torch.tensor(atom_features, dtype=torch.float32)

    # -----------------------------------------------------
    # 2. ARISTAS (Enlaces) - Aquí está la novedad
    # -----------------------------------------------------
    edge_indices = []
    edge_attrs = []

    for bond in mol.GetBonds():
        bond = cast(rdchem.Bond, bond)
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()

        # Features del enlace
        b_type = bond.GetBondType()
        bond_feats = one_hot_encoding(b_type, ALLOWED_BOND_TYPES, allow_unknown=True)

        bond_feats.append(1 if bond.GetIsConjugated() else 0)
        bond_feats.append(1 if bond.IsInRing() else 0)

        # Stereo: Simplificado a binario como tenías, o podrías hacer one-hot si la estereoquímica E/Z es vital.
        bond_feats.append(1 if bond.GetStereo() != rdchem.BondStereo.STEREONONE else 0)
        # Grafo no dirigido: agregamos ambas direcciones
        edge_indices.append([start, end])
        edge_attrs.append(bond_feats)

        edge_indices.append([end, start])
        edge_attrs.append(bond_feats)

    # Convertir a Tensores
    if len(edge_indices) == 0:
        # Manejo de moléculas de un solo átomo (sin enlaces)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, len(bond_feats)), dtype=torch.float32)  # type: ignore | Ajuste dinámico de la dimensión
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)

    # DESCRIPTORES MOLECULARES ADICIONALES (Opcional)
    global_features = Dcrp.CalcMolDescriptors(mol).values().to_list()
    u = torch.tensor([global_features], dtype=torch.float32)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def visualizar_grafo_interpretado(
    grafo_pyg: Data, smiles_title: str = "Molecula"
) -> None:
    """
    Visualiza un objeto Data de PyG utilizando el motor de dibujo de RDKit (estilo display_one_graph),
    reconstruyendo la molécula directamente desde los tensores de PyG.
    """
    grafo_pyg = cast(Data, grafo_pyg)
    grafo_pyg.x = cast(torch.Tensor, grafo_pyg.x)
    grafo_pyg.edge_index = cast(torch.Tensor, grafo_pyg.edge_index)
    if hasattr(grafo_pyg, "edge_attr"):
        grafo_pyg.edge_attr = cast(torch.Tensor, grafo_pyg.edge_attr)

    # 1. Reconstrucción de la Molécula desde PyG (Sin diccionarios intermedios externos)
    mol = Chem.RWMol()
    node_colors = {}

    # Obtener tensores como numpy para iteración rápida
    x_features = grafo_pyg.x.detach().cpu().numpy()

    # Añadir átomos
    num_nodes = cast(int, grafo_pyg.num_nodes)
    for i in range(num_nodes):
        # Asumiendo que la pos 0 es Z normalizado (x100)
        z_val = int(round(x_features[i, 0] * 100))
        atom = Chem.Atom(z_val)

        # idx = mol.AddAtom(atom) # AddAtom devuelve el índice del átomo añadido
        # atom.SetAtomMapNum(atom.GetIdx())
        idx = mol.AddAtom(atom)
        atom.SetAtomMapNum(atom.GetIdx())

        # Lógica de color basada en features (ej: aromático en la última posición)
        # Replicamos el estilo visual: Rojo si es aromático, Azul si no
        is_aromatic = x_features[i, -1] == 1.0
        if is_aromatic:
            node_colors[idx] = (1.0, 0.6, 0.6)  # Rojo pastel
        else:
            node_colors[idx] = (0.6, 0.8, 1.0)  # Azul pastel

    # Añadir enlaces desde edge_index
    # edge_index tiene forma [2, num_edges]
    edge_index = grafo_pyg.edge_index.detach().cpu().numpy()

    if hasattr(grafo_pyg, "edge_attr") and grafo_pyg.edge_attr is not None:
        edge_attrs = grafo_pyg.edge_attr.detach().cpu().numpy()
        has_edge_attr = True
    else:
        has_edge_attr = False

    # Usamos un set para evitar duplicados, ya que PyG suele tener enlaces dirigidos (i->j y j->i)
    added_bonds = set()

    for k in range(edge_index.shape[1]):
        i, j = int(edge_index[0, k]), int(edge_index[1, k])

        # Evitar duplicados (PyG suele tener i->j y j->i)
        bond_tuple = tuple(sorted((i, j)))
        if bond_tuple in added_bonds:
            continue

        added_bonds.add(bond_tuple)

        rdkit_bond_type = Chem.BondType.SINGLE  # Default

        if has_edge_attr:
            # Lógica de Decodificación de Enlace
            # Asumimos One-Hot Encoding de 4 dimensiones: [Single, Double, Triple, Aromatic]
            # Si tu modelo usa un solo entero (ordinal), cambia argmax por el valor directo.
            attr_vec = edge_attrs[k]  # type: ignore
            bond_idx = np.argmax(attr_vec)

            if bond_idx == 0:
                rdkit_bond_type = Chem.BondType.SINGLE
            elif bond_idx == 1:
                rdkit_bond_type = Chem.BondType.DOUBLE
            elif bond_idx == 2:
                rdkit_bond_type = Chem.BondType.TRIPLE
            elif bond_idx == 3:
                rdkit_bond_type = Chem.BondType.AROMATIC

        mol.AddBond(i, j, rdkit_bond_type)

    # Congelar la molécula para permitir cálculos de RDKit
    mol = mol.GetMol()

    # Calcular coordenadas 2D para el dibujo
    try:
        mol.UpdatePropertyCache()
        rdkit.Chem.rdDepictor.Compute2DCoords(mol)
        print("Coordenadas 2D calculadas exitosamente.")
    except:
        # Fallback si la valencia química no es perfecta al reconstruir desde un grafo simplificado
        print("Advertencia: No se pudieron calcular las coordenadas 2D correctamente.")
        pass

    # 2. Configuración del Dibujo (Estilo display_molecule_graph)
    # Usamos MolDraw2DCairo para generar PNG directamente en memoria (más rápido que SVG->PNG->Disco)
    width, height = 1050, 1000
    try:
        d = rdMolDraw2D.MolDraw2DCairo(width, height)
    except AttributeError:
        # Fallback a SVG si Cairo no está instalado en el entorno
        d = rdMolDraw2D.MolDraw2DSVG(width, height)

    d = rdMolDraw2D.MolDraw2DCairo(width, height)

    d.drawOptions().addAtomIndices = True  # Útil para debuggear grafos
    d.drawOptions().annotationFontScale = 0.6
    d.SetFontSize(40)  # Ajuste de tamaño relativo

    # Prepara la molécula
    rdMolDraw2D.PrepareMolForDrawing(mol, kekulize=True, wedgeBonds=True)

    # Dibujar usando los colores definidos directamente desde el tensor
    d.DrawMolecule(
        mol, highlightAtoms=list(node_colors.keys()), highlightAtomColors=node_colors
    )

    d.FinishDrawing()

    # 3. Conversión a Imagen para Matplotlib
    # Extraemos los datos binarios de la imagen
    bin_data = d.GetDrawingText()

    if isinstance(d, rdMolDraw2D.MolDraw2DSVG):
        # Si tuvimos que usar SVG (fallback), necesitamos convertirlo a raster
        # Nota: Esto requiere cairosvg instalado. Si no, devuelve el SVG crudo.
        try:
            import cairosvg

            png_data: bytes = cast(bytes, cairosvg.svg2png(bytestring=bin_data))
            img = Image.open(io.BytesIO(png_data))
        except ImportError:
            print(
                "Advertencia: Instala 'cairosvg' para renderizar SVG en matplotlib. Mostrando placeholder."
            )
            return
    else:
        # Flujo normal con Cairo (PNG directo)
        img = Image.open(io.BytesIO(bin_data))

    # 4. Renderizado en Matplotlib (Estilo display_one_graph)
    plt.figure(figsize=(10, 8), dpi=150)  # Ajustado para buena visibilidad
    ax = plt.subplot(1, 1, 1)

    # Eliminar ejes y grids como en la función de referencia
    ax.grid(False)
    ax.axis("off")

    ax.imshow(img)
    plt.title(f"{smiles_title}", fontsize=14, pad=20)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # smiles_test = "CC(=O)Nc1ccc(O)cc1"
    smiles_test = "C1=CC=C(C(=C1)C2=NC(C(=O)NC3=C2C=C(C=C3)Cl)O)Cl"
    grafo = smiles_to_graph_complete(smiles_test)

    if grafo:
        visualizar_grafo_interpretado(grafo, smiles_test)
