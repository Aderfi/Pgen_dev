try:
    import os
    import re
    import json
    import subprocess
    import torch
    import numpy as np
    import pandas as pd
    import networkx as nx
    from pathlib import Path
    from typing import Dict, List, Any, Optional, cast
    from tqdm import tqdm

    # Importaciones de Bioinformática
    from pyfaidx import Fasta
    from intervaltree import IntervalTree
    from torch_geometric.data import Data
except ImportError as e:
    missing_module = str(e).split()[-1]
    
    raise ImportError(f"Falta el módulo requerido: {missing_module}. Por favor, instálalo para continuar.")


# Importaciones del proyecto (Asumiendo estructura src/)

PGX_METADATA = {
    # --- ENZIMAS MAYORES (METABOLISMO DE FÁRMACOS) ---

    "CYP2D6": {
        "*1": {"metabolic_function": "Normal Function", "activity_score": 1.0},
        "*2": {"metabolic_function": "Normal Function", "activity_score": 1.0},
        "*3": {"metabolic_function": "No Function", "activity_score": 0.0},
        "*4": {"metabolic_function": "No Function", "activity_score": 0.0},
        "*5": {"metabolic_function": "No Function (Deletion)", "activity_score": 0.0},
        "*6": {"metabolic_function": "No Function", "activity_score": 0.0},
        "*9": {"metabolic_function": "Decreased Function", "activity_score": 0.5},
        "*10": {"metabolic_function": "Decreased Function", "activity_score": 0.25},  # Ajuste CPIC reciente
        "*17": {"metabolic_function": "Decreased Function", "activity_score": 0.5},
        "*29": {"metabolic_function": "Decreased Function", "activity_score": 0.5},
        "*41": {"metabolic_function": "Decreased Function", "activity_score": 0.5},
        "*1xN": {"metabolic_function": "Increased Function", "activity_score": 2.0},  # Duplicación
        "*2xN": {"metabolic_function": "Increased Function", "activity_score": 2.0},  # Duplicación
        "default": {"metabolic_function": "Unknown_DEF", "activity_score": None}
    },

    "CYP2C19": {
        "*1": {"metabolic_function": "Normal Function", "activity_score": 1.0},
        "*2": {"metabolic_function": "No Function", "activity_score": 0.0},
        "*3": {"metabolic_function": "No Function", "activity_score": 0.0},
        "*4": {"metabolic_function": "No Function", "activity_score": 0.0},
        "*17": {"metabolic_function": "Increased Function", "activity_score": 1.0},  # Nota: *17/*17 es UM
        "default": {"metabolic_function": "Unknown", "activity_score": None}
    },

    "CYP2C9": {
        "*1": {"metabolic_function": "Normal Function", "activity_score": 1.0},
        "*2": {"metabolic_function": "Decreased Function", "activity_score": 0.5},
        "*3": {"metabolic_function": "Decreased Function (Severe)", "activity_score": 0.0},  # Actividad muy baja (<10%)
        "*5": {"metabolic_function": "Decreased Function", "activity_score": None},
        "*6": {"metabolic_function": "No Function", "activity_score": 0.0},
        "*8": {"metabolic_function": "Decreased Function", "activity_score": None},
        "*11": {"metabolic_function": "Decreased Function", "activity_score": None},
        "default": {"metabolic_function": "Unknown", "activity_score": None}
    },

    "CYP2B6": {
        "*1": {"metabolic_function": "Normal Function", "activity_score": 1.0},
        "*4": {"metabolic_function": "Increased Function", "activity_score": None},
        "*6": {"metabolic_function": "Decreased Function", "activity_score": 0.0},  # Principal alelo PM (ej. Efavirenz)
        "*9": {"metabolic_function": "Decreased Function", "activity_score": None},
        "*18": {"metabolic_function": "No Function", "activity_score": 0.0},
        "default": {"metabolic_function": "Unknown", "activity_score": None}
    },

    "CYP3A4": {
        "*1": {"metabolic_function": "Normal Function", "activity_score": 1.0},
        "*20": {"metabolic_function": "No Function", "activity_score": 0.0},
        "*22": {"metabolic_function": "Decreased Function", "activity_score": 0.5},  # Intrónico, afecta expresión
        "*26": {"metabolic_function": "No Function", "activity_score": 0.0},
        "default": {"metabolic_function": "Unknown", "activity_score": None}
    },

    "CYP3A5": {
        "*1": {"metabolic_function": "Normal Function (Expresser)", "activity_score": 1.0},
        "*3": {"metabolic_function": "No Function (Non-expresser)", "activity_score": 0.0},  # El más común en caucásicos
        "*6": {"metabolic_function": "No Function", "activity_score": 0.0},
        "*7": {"metabolic_function": "No Function", "activity_score": 0.0},
        "default": {"metabolic_function": "Unknown", "activity_score": None}
    },

    "CYP1A2": {
        "*1F": {"metabolic_function": "Inducible/Normal", "activity_score": 1.0},  # Alto riesgo con tabaco/café
        "*1C": {"metabolic_function": "Decreased Function", "activity_score": 0.5},
        "*1K": {"metabolic_function": "Decreased Function", "activity_score": 0.5},
        "default": {"metabolic_function": "Normal Function", "activity_score": 1.0}  # *1A asumed
    },

    "CYP2C8": {
        "*1": {"metabolic_function": "Normal Function", "activity_score": 1.0},
        "*2": {"metabolic_function": "Decreased Function", "activity_score": None},  # Común en ancestría africana
        "*3": {"metabolic_function": "Decreased/Altered", "activity_score": None},  # Afecta eliminación de paclitaxel
        "*4": {"metabolic_function": "Decreased Function", "activity_score": None},
        "default": {"metabolic_function": "Unknown", "activity_score": None}
    },

    # --- TOXICIDAD / SEGURIDAD (CRÍTICOS) ---

    "DPYD": {
        "*1": {"metabolic_function": "Normal Function", "activity_score": 1.0},
        "*2A": {"metabolic_function": "No Function", "activity_score": 0.0},  # Riesgo mortal con 5-FU/Capecitabina
        "*13": {"metabolic_function": "No Function", "activity_score": 0.0},
        "HapB3": {"metabolic_function": "Decreased Function", "activity_score": 0.5},
        "c.2846A>T": {"metabolic_function": "Decreased Function", "activity_score": 0.5},
        "c.1236G>A": {"metabolic_function": "Decreased Function", "activity_score": 0.5},
        "*9A": {"metabolic_function": "Normal Function", "activity_score": 1.0},  # Históricamente debatido, ahora Normal
        "default": {"metabolic_function": "Unknown", "activity_score": None}
    },

    "NUDT15": {
        "*1": {"metabolic_function": "Normal Function", "activity_score": 1.0},
        "*2": {"metabolic_function": "No Function", "activity_score": 0.0},  # Riesgo mielosupresión con Tiopurinas
        "*3": {"metabolic_function": "No Function", "activity_score": 0.0},
        "*4": {"metabolic_function": "Uncertain", "activity_score": None},
        "default": {"metabolic_function": "Unknown", "activity_score": None}
    },

    "SLCO1B1": {
        "*1": {"metabolic_function": "Normal Function", "activity_score": 1.0},
        "*5": {"metabolic_function": "No Function/Decreased", "activity_score": 0.0},  # Riesgo miopatía estatinas
        "*15": {"metabolic_function": "No Function/Decreased", "activity_score": 0.0},
        "*37": {"metabolic_function": "Normal Function", "activity_score": 1.0},
        "default": {"metabolic_function": "Unknown", "activity_score": None}
    },

    "NAT2": {
        # NAT2 se clasifica por fenotipo Acetilador (Rapid, Intermediate, Slow)
        "*4": {"metabolic_function": "Rapid Acetylator", "activity_score": 1.0},
        "*5": {"metabolic_function": "Slow Acetylator", "activity_score": 0.0},
        "*6": {"metabolic_function": "Slow Acetylator", "activity_score": 0.0},
        "*7": {"metabolic_function": "Slow Acetylator", "activity_score": 0.0},
        "*12": {"metabolic_function": "Rapid Acetylator", "activity_score": 1.0},
        "*14": {"metabolic_function": "Slow Acetylator", "activity_score": 0.0},
        "default": {"metabolic_function": "Unknown", "activity_score": None}
    },

    # --- OTROS ENZIMAS METABÓLICOS ---

    "CYP4F2": {
        "*1": {"metabolic_function": "Normal Function", "activity_score": 1.0},
        "*3": {"metabolic_function": "Decreased Function", "activity_score": 0.5},
        # (V433M) Reduce metabolismo de Vit K -> Requiere MAYOR dosis de Warfarina
        "default": {"metabolic_function": "Unknown", "activity_score": None}
    },

    "CYP2A6": {
        "*1": {"metabolic_function": "Normal Function", "activity_score": 1.0},
        "*2": {"metabolic_function": "No Function", "activity_score": 0.0},
        "*4": {"metabolic_function": "No Function (Deletion)", "activity_score": 0.0},  # Común en asiáticos
        "*9": {"metabolic_function": "Decreased Function", "activity_score": 0.5},
        "*12": {"metabolic_function": "Decreased Function", "activity_score": 0.5},
        "default": {"metabolic_function": "Unknown", "activity_score": None}
    },

    "CYP2A13": {
        "*1": {"metabolic_function": "Normal Function", "activity_score": 1.0},
        "*2": {"metabolic_function": "Decreased Function", "activity_score": 0.5},  # Menor activación de carcinógenos tabáquicos
        "*3": {"metabolic_function": "Decreased Function", "activity_score": 0.5},
        "*4": {"metabolic_function": "Decreased Function", "activity_score": 0.5},
        "*7": {"metabolic_function": "No Function", "activity_score": 0.0},
        "default": {"metabolic_function": "Unknown", "activity_score": None}
    }
}

# Intentar importar pandarallel
try:
    from pandarallel import pandarallel
    pandarallel.initialize(progress_bar=True, verbose=1)
except ImportError:
    raise ImportError("pandarallel no está instalado. Por favor, instálalo para ejecutar este script.")

# =============================================================================
#  CONFIGURACIÓN GLOBAL Y MAPEOS
# =============================================================================

DPYD_RS_MAP = {
    "rs3918290": "*2A", 
    "rs55886062": "*13",
    "rs67376798": "c.2846A>T",
    "rs75017182": "HapB3",
    "rs1801265": "*4"
}

# Variables Globales para Workers (Optimización de memoria en Linux/Fork)
GLOBAL_GENOME = None
GLOBAL_GENE_TREES = None
GLOBAL_CHROM_MAPPING = {
    '1': 'NC_000001.11', '2': 'NC_000002.12', '3': 'NC_000003.12',
    '4': 'NC_000004.12', '5': 'NC_000005.10', '6': 'NC_000006.12',
    '7': 'NC_000007.14', '8': 'NC_000008.11', '9': 'NC_000009.12',
    '10': 'NC_000010.11', '11': 'NC_000011.10', '12': 'NC_000012.12',
    '13': 'NC_000013.11', '14': 'NC_000014.9', '15': 'NC_000015.10',
    '16': 'NC_000016.10', '17': 'NC_000017.11', '18': 'NC_000018.10',
    '19': 'NC_000019.10', '20': 'NC_000020.11', '21': 'NC_000021.9',
    '22': 'NC_000022.11',
    'X': 'NC_000023.11', 'Y': 'NC_000024.10',
    'M': 'NC_012920.1', 'MT': 'NC_012920.1'
}

# =============================================================================
#  HELPER FUNCTIONS (Global scope for Pickling)
# =============================================================================

def smiles_to_graph_complete(smiles):
    from rdkit import Chem
    from rdkit.Chem import rdchem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # -----------------------------------------------------
    # 1. NODOS (Átomos)
    # -----------------------------------------------------
    atom_features = []

    for atom in mol.GetAtoms():
        z_scalar = atom.GetAtomicNum() / 100.0
        features = [z_scalar]

        # Características adicionales del atomo (nodo)
        features += one_hot_encoding(atom.GetDegree(),
                                     [0, 1, 2, 3, 4])
        features += one_hot_encoding(atom.GetFormalCharge(),
                                     [-2, -1, 0, 1, 2])
        features += one_hot_encoding(atom.GetHybridization(), [
            rdchem.HybridizationType.SP,
            rdchem.HybridizationType.SP2,
            rdchem.HybridizationType.SP3
        ])
        features += one_hot_encoding(atom.GetTotalNumHs(),
                                     [0, 1, 2, 3, 4])

        chiral_tag = atom.GetChiralTag()
        features += one_hot_encoding(chiral_tag, [
            rdchem.ChiralType.CHI_UNSPECIFIED,
            rdchem.ChiralType.CHI_TETRAHEDRAL_CW,  # Clockwise (R)
            rdchem.ChiralType.CHI_TETRAHEDRAL_CCW  # Counter-Clockwise (S)
        ])

        charge = atom.GetFormalCharge()
        features.append(charge)

        features.append(1 if atom.GetIsAromatic() else 0)
        features.append(atom.GetMass() * 0.01)
        atom_features.append(features)

    x = torch.tensor(atom_features, dtype=torch.float)

    # -----------------------------------------------------
    # 2. ARISTAS (Enlaces) - Aquí está la novedad
    # -----------------------------------------------------
    edge_indices = []
    edge_attrs = []

    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()

        # A. Extraer Tipo de Enlace (One-Hot)
        # Vector de 4 posiciones: [Simple, Doble, Triple, Aromático]
        bt = bond.GetBondType()
        bond_feats = one_hot_encoding(bt, [
            rdchem.BondType.SINGLE,
            rdchem.BondType.DOUBLE,
            rdchem.BondType.TRIPLE,
            rdchem.BondType.AROMATIC
        ])

        bond_feats.append(1 if bond.GetIsConjugated() else 0)
        bond_feats.append(1 if bond.IsInRing() else 0)

        # Estereoquímica del enlace
        bond_feats.append(1 if bond.GetStereo() != rdchem.BondStereo.STEREONONE else 0)

        # Dirección 1: Start -> End
        edge_indices.append([start, end])
        edge_attrs.append(bond_feats)

        # Dirección 2: End -> Start
        edge_indices.append([end, start])
        edge_attrs.append(bond_feats)

    # Convertir a Tensores
    if len(edge_indices) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 6), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

def one_hot_encoding(value, possible_values):
    encoding = [0] * len(possible_values)
    if value in possible_values:
        encoding[possible_values.index(value)] = 1
    return encoding


def worker_validate_row(row):
    """
    Función pura ejecutada en paralelo. Valida variantes, extrae secuencias
    y asigna nombres clínicos (variant_name).
    """
    def _map_chrom(c):
        c_str = str(c).strip()
        if GLOBAL_CHROM_MAPPING and c_str in GLOBAL_CHROM_MAPPING: 
            return GLOBAL_CHROM_MAPPING[c_str]
        clean = c_str.replace('chr', '').replace('Chr', '')
        if GLOBAL_CHROM_MAPPING:
            return GLOBAL_CHROM_MAPPING.get(clean, c_str)
        return c_str

    def _get_gene(chrom, pos):
        if GLOBAL_GENE_TREES and chrom in GLOBAL_GENE_TREES:
            matches = GLOBAL_GENE_TREES[chrom].at(pos)
            if matches:
                names = sorted(list(set([m.data['gene'] for m in matches])))
                return names[0]
        return "Intergenic"

    def _parse_variant_name_logic(gene, alt, label):
        raw = str(label) if pd.notna(label) and str(label).strip() else str(alt)
        if '*' in raw: return raw
        if raw.startswith("rs"): return raw
        return None

    ref_raw = str(row['REF']).upper().strip()
    alt_raw = str(row['ALT']).upper().strip()
    is_star_allele = '*' in alt_raw

    ref_clean = "" if ref_raw == '-' else ref_raw
    alt_clean = alt_raw if is_star_allele else ("" if alt_raw == '-' else alt_raw)
    
    chrom_mapped = _map_chrom(row['CHROM'])
    
    row['validated'] = False
    row['validation_error'] = None

    try:
        pos_0 = int(row['POS']) - 1
        if pos_0 < 0: raise ValueError
    except ValueError:
        row['validation_error'] = "Bad POS"
        return row

    if GLOBAL_GENOME:
        if chrom_mapped not in GLOBAL_GENOME:
            row['validation_error'] = f"Chr missing: {chrom_mapped}"
            return row
        try:
            if not is_star_allele and len(ref_clean) > 0:
                ref_fasta = GLOBAL_GENOME[chrom_mapped][pos_0 : pos_0 + len(ref_clean)].seq.upper()
                if ref_fasta != ref_clean and 'N' not in ref_fasta:
                    row['validation_error'] = f"Ref Mismatch: {ref_clean} vs {ref_fasta}"
                    return row
        except IndexError:
            row['validation_error'] = "Out of bounds"
            return row

    existing_gene = str(row.get('gene_provided', ''))
    if not existing_gene or existing_gene.lower() in ['nan', 'none', '.', '']:
        row['gene_context'] = _get_gene(chrom_mapped, pos_0)
    else:
        row['gene_context'] = existing_gene

    row['variant_name'] = _parse_variant_name_logic(row['gene_context'], alt_clean, row.get('haplotype_label'))
    
    if is_star_allele: row['variant_type'] = 'STAR_ALLELE'
    elif len(ref_clean) == len(alt_clean): row['variant_type'] = 'SNP' if len(ref_clean)==1 else 'MNP'
    elif len(ref_clean) > len(alt_clean): row['variant_type'] = 'DEL'
    else: row['variant_type'] = 'INS'

    score = -1.0
    if pd.notna(row.get('activity_score')):
        try: score = float(row['activity_score'])
        except: pass
    else:
        if row['variant_type'] == 'STAR_ALLELE': score = 0.5 
    
    row['activity_score'] = float(score)
    net = len(alt_clean) - len(ref_clean)
    row['is_frameshift'] = (net != 0 and abs(net) % 3 != 0) if not is_star_allele else False

    row['validated'] = True
    row['REF'] = ref_clean
    row['ALT'] = alt_clean
    row['CHROM'] = chrom_mapped
    return row

# =============================================================================
#  CLASE 1: CONSTRUCTOR DE GRAFOS GENÓMICOS
# =============================================================================

class GenomicGraphBuilder:
    def __init__(self, fasta_path: Path, gff_path: Path, pgx_dir: Path):
        self.fasta_path = fasta_path
        self.gff_path = gff_path
        self.pgx_dir = pgx_dir

    def run_pipeline(self, tsv_input: Path, output_parquet: Path, output_graph_dir: Path):
        print(f"\n🧬 [GENOMICS] Iniciando pipeline genómico...")
        
        # 1. Construir Librería Parquet
        if self.fasta_path.exists() and self.gff_path.exists():
            clean_df = self._build_library(tsv_input)
            if clean_df is not None and not clean_df.empty:
                clean_df.to_parquet(output_parquet, index=False)
                print(f"✅ Librería guardada en: {output_parquet} ({len(clean_df)} variantes)")
                
                # 2. Generar Grafos
                self._generate_graphs(clean_df, output_graph_dir)
                
                # 3. Organizar Archivos (OS Specific)
                self._organize_files_os_specific(output_graph_dir)
            else:
                print("⚠️ No se generó dataframe limpio de variantes.")
        else:
            print(f"❌ Faltan archivos de referencia (FASTA/GFF) en {self.fasta_path.parent}")

    def _build_library(self, tsv_path: Path) -> pd.DataFrame:
        print("   🔹 Indexando genoma y anotaciones...")
        genome = Fasta(str(self.fasta_path), key_function=lambda x: x.split()[0])
        gene_trees = self._build_gene_index()

        dfs = []
        # Cargar TSV Principal
        if tsv_path.exists():
            print("   🔹 Cargando TSV de variantes genéticas...")
            t_df = pd.read_csv(tsv_path, sep='\t')
            rename_map = {'chr': 'CHROM', 'pos': 'POS', 'Ref_Allele': 'REF', 
                          'Alt_Allele': 'ALT', 'gene': 'gene_provided', 'snp': 'haplotype_label'}
            t_df.rename(columns=lambda x: rename_map.get(x.lower(), rename_map.get(x, x.upper())), inplace=True)
            dfs.append(t_df)
        
        # Cargar Datos PGx
        pgx_df = self._load_pgx_folder()
        if not pgx_df.empty:
            dfs.append(pgx_df)

        if not dfs: return None

        master_df = pd.concat(dfs, ignore_index=True)
        master_df['POS'] = pd.to_numeric(master_df['POS'], errors='coerce').dropna().astype(int)

        # Configurar Globales para Workers
        global GLOBAL_GENOME, GLOBAL_GENE_TREES
        GLOBAL_GENOME = genome
        GLOBAL_GENE_TREES = gene_trees

        print(f"   ⚡ Validando {len(master_df)} variantes...")
        if pandarallel:
            processed = master_df.parallel_apply(worker_validate_row, axis=1)
        else:
            tqdm.pandas(desc="Validando filas")
            processed = master_df.progress_apply(worker_validate_row, axis=1)

        # Limpiar referencias
        GLOBAL_GENOME = None; GLOBAL_GENE_TREES = None

        clean = processed[processed['validated'] == True].copy()
        clean['has_haplo'] = clean['haplotype_label'].notna()
        clean.sort_values(by=['CHROM', 'POS', 'has_haplo'], ascending=[True, True, False], inplace=True)
        clean.drop_duplicates(subset=['CHROM', 'POS', 'REF', 'ALT'], keep='first', inplace=True)
        
        return clean

    def _build_gene_index(self) -> Dict:
        comp = 'gzip' if str(self.gff_path).endswith('.gz') else None
        try:
            df = pd.read_csv(self.gff_path, sep='\t', comment='#', compression=comp,
                             names=['seq', 'src', 'type', 'start', 'end', 'sc', 'str', 'ph', 'attr'],
                             usecols=['seq', 'type', 'start', 'end', 'attr'])
            df = df[df['attr'].str.contains('gene=', na=False) | (df['type'] == 'gene')]
            tree_dict = {}
            for chrom, grp in df.groupby('seq'):
                t = IntervalTree()
                for row in grp.itertuples():
                    match = re.search(r'Name=([^;]+)', str(row.attr))
                    if not match: match = re.search(r'gene=([^;]+)', str(row.attr))
                    name = match.group(1) if match else "Unknown"
                    t.addi(row.start -1, row.end, data={'gene': name})
                tree_dict[str(chrom)] = t
            return tree_dict
        except Exception as e:
            print(f"Error GFF: {e}")
            return {}

    def _load_pgx_folder(self) -> pd.DataFrame:
        all_variants = []
        if not self.pgx_dir.exists(): return pd.DataFrame()
        
        for gene_folder in self.pgx_dir.iterdir():
            if gene_folder.is_dir():
                gene_name = gene_folder.name
                for vcf_file in gene_folder.glob("*.vcf"):
                    haplo_label = self._parse_haplo_name(gene_name, vcf_file.stem)
                    try:
                        vcf_df = pd.read_csv(vcf_file, sep='\t', comment='#', header=None,
                            names=['CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO'],
                            dtype={'CHROM': str, 'REF': str, 'ALT': str})
                        if vcf_df.empty: continue
                        vcf_df['gene_provided'] = gene_name
                        vcf_df['haplotype_label'] = haplo_label
                        all_variants.append(vcf_df[['CHROM', 'POS', 'REF', 'ALT', 'gene_provided', 'haplotype_label']])
                    except: continue
        return pd.concat(all_variants, ignore_index=True) if all_variants else pd.DataFrame()

    def _parse_haplo_name(self, gene, fname):
        if fname.startswith("rs"): return DPYD_RS_MAP.get(fname, fname)
        clean = fname.replace(f"{gene}_", "").replace(gene, "")
        base = clean.split('.')[0] if "." in clean else clean
        return f"*{base}" if base.isdigit() or not base.startswith("*") else base

    def _generate_graphs(self, library_df: pd.DataFrame, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        genes = library_df['gene_context'].dropna().unique()
        print(f"   🚀 Generando grafos PyG para {len(genes)} genes...")
        
        count = 0
        for gene in tqdm(genes, desc="Procesando Genes"):
            df_gene = library_df[library_df['gene_context'] == gene]
            for var_name in df_gene['variant_name'].dropna().unique():
                if str(var_name).strip() == "": continue
                df_variant = df_gene[df_gene['variant_name'] == var_name]
                G = self._build_nx_graph(df_variant, gene, var_name)
                if G:
                    pyg_data = self._to_pyg(G)
                    safe_var = str(var_name).replace("*", "star").replace(":", "_").replace("/", "_")
                    torch.save(pyg_data, output_dir / f"{gene}_{safe_var}.pt")
                    count += 1

    def _build_nx_graph(self, df, gene_name, var_name):
        G = nx.MultiDiGraph(name=f"{gene_name}_{var_name}")
        df = df.sort_values('POS')
        prev_node = "start"
        start_pos = df['POS'].min() - 100
        G.add_node(prev_node, type="backbone", pos=start_pos)
        
        for idx, row in df.iterrows():
            pos = row['POS']
            bb_node = f"bb_{pos}"
            G.add_node(bb_node, type="backbone", pos=pos)
            G.add_edge(prev_node, bb_node, type="backbone_link")
            split = f"split_{pos}"
            G.add_node(split, type="split", pos=pos)
            G.add_edge(bb_node, split, type="link")
            merge = f"merge_{pos+1}"
            G.add_node(merge, type="merge", pos=pos+1)
            
            # Camino Ref
            ref_n = f"ref_{pos}"
            G.add_node(ref_n, type="allele_ref", seq=row['REF'])
            G.add_edge(split, ref_n, attr="ref")
            G.add_edge(ref_n, merge, attr="join")
            
            # Camino Alt
            alt_n = f"alt_{pos}"
            G.add_node(alt_n, type="allele_alt", seq=row['ALT'], score=row['activity_score'], variant_name=var_name)
            G.add_edge(split, alt_n, attr="alt")
            G.add_edge(alt_n, merge, attr="join")
            prev_node = merge

        G.add_node("end", type="backbone_end", pos=df['POS'].max() + 100)
        G.add_edge(prev_node, "end", type="backbone_link")
        return G

    def _to_pyg(self, G: nx.MultiDiGraph) -> Data:
        nodes = list(G.nodes(data=True))
        node_idx = {n: i for i, (n, _) in enumerate(nodes)}
        x_list = []
        variant_name_str = "Unknown"
        
        for _, d in nodes:
            t = d.get('type', '')
            score = d.get('score', 0.5) if d.get('score') != -1.0 else 0.5
            if 'variant_name' in d: variant_name_str = d['variant_name']
            vec = [0.0] * 5
            if 'backbone' in t: vec[0] = 1.0
            elif 'split' in t or 'merge' in t: vec[1] = 1.0
            elif 'ref' in t: vec[2] = 1.0
            elif 'alt' in t: 
                vec[3] = 1.0; vec[4] = float(score)
            x_list.append(vec)
            
        edge_index = [[node_idx[u], node_idx[v]] for u, v, _ in G.edges(data=True)]
        edge_attr = []
        for _, _, data in G.edges(data=True):
            vec = [0.0, 0.0, 0.0]
            if 'ref' in data.get('attr', ''): vec[1] = 1.0
            elif 'alt' in data.get('attr', ''): vec[2] = 1.0
            else: vec[0] = 1.0
            edge_attr.append(vec)
            
        data = Data(x=torch.tensor(x_list, dtype=torch.float32), 
                    edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
                    edge_attr=torch.tensor(edge_attr, dtype=torch.float32))
        data.variant_name = variant_name_str
        return data

    def _organize_files_os_specific(self, graph_dir: Path):
        """Genera y ejecuta scripts de organización según el SO."""
        print(f"   📂 Organizando estructura de carpetas en {graph_dir}...")
        
        if os.name == 'posix':  # Linux / macOS
            script_content = """#!/bin/bash
echo "Organizando..."
count=0
for filename in *.pt; do
    [ -e "$filename" ] || continue
    raw_gene="${filename%%_*}"
    gene_name="${raw_gene%%;*}"
    if [[ "$gene_name" =~ ^UGT1A([1-9]|10)$ ]]; then target_dir="UGT1A"; else target_dir="$gene_name"; fi
    mkdir -p "$target_dir"
    mv "$filename" "$target_dir/"
    ((count++))
done
echo "Archivos movidos: $count"
"""
            script_path = graph_dir / "organize_genes.sh"
            with open(script_path, "w") as f: f.write(script_content)
            os.chmod(script_path, 0o755)
            subprocess.run(["bash", script_path.name], cwd=graph_dir, check=True)

        elif os.name == 'nt':  # Windows
            script_content = """
$ErrorActionPreference = "Continue"
$files = Get-ChildItem -Filter *.pt -File
foreach ($file in $files) {
    $rawGene = $file.BaseName.Split('_')[0]
    $geneName = $rawGene.Split(';')[0]
    if ($geneName -match '^UGT1A([1-9]|10)$') { $targetDirName = "UGT1A" } else { $targetDirName = $geneName }
    $targetPath = Join-Path -Path $PWD -ChildPath $targetDirName
    if (-not (Test-Path -Path $targetPath)) { New-Item -Path $targetPath -ItemType Directory -Force | Out-Null }
    $file | Move-Item -Destination $targetPath -Force
}
"""
            script_path = graph_dir / "organize_genes.ps1"
            with open(script_path, "w") as f: f.write(script_content)
            subprocess.run(["powershell", "-ExecutionPolicy", "Bypass", "-File", script_path.name], cwd=graph_dir, check=True)

# =============================================================================
#  CLASE 2: CONSTRUCTOR DE GRAFOS DE FÁRMACOS
# =============================================================================

class DrugGraphBuilder:
    def __init__(self):
        self.illegal_chars = re.compile(r'[<>:"/\\|?*]')

    def run_pipeline(self, tsv_input: Path, output_dir: Path):
        print(f"\n💊 [DRUGS] Iniciando pipeline de fármacos...")
        if not tsv_input.exists():
            print(f"❌ Error: No se encuentra {tsv_input}")
            return

        if smiles_to_graph_complete is None:
            print("❌ No se puede ejecutar: falta función `smiles_to_graph_complete`.")
            return

        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Usamos 'cmpd_name_cleaned' según el contexto guardado, o fallback a 'cmpd_name'
            cols = ["cid", "smiles", "cmpd_name_cleaned"]
            df = pd.read_csv(tsv_input, sep="\t", usecols=cols)
        except Exception as e:
            print(f"❌ Error leyendo columnas del TSV de fármacos: {e}")
            return

        failed = []
        count = 0
        
        for row in tqdm(df.itertuples(index=False), total=len(df), desc="Creando grafos Drugs"):
            raw_name = str(row.cmpd_name_cleaned).strip()
            safe_name = self.illegal_chars.sub("_", raw_name).replace(" ", "_")[:100]
            
            out_f = output_dir / f"{row.cid}_{safe_name}.pt"
            if out_f.exists(): continue
            
            try:
                graph = smiles_to_graph_complete(cast(row.smiles, str).strip()) 
                if graph is not None:
                    graph.cid = row.cid
                    graph.name = row.cmpd_name_cleaned
                    graph.smiles = row.smiles.strip()
                    torch.save(graph, out_f)
                    count += 1
                else:
                    failed.append((row.cid, raw_name, "Invalid SMILES"))
            except Exception as e:
                failed.append((row.cid, raw_name, str(e)))

        print(f"✨ Fármacos procesados: {count}")
        if failed:
            self._log_errors(failed)

    def _log_errors(self, failed_list):
        with open("drug_generation_errors.log", "a", encoding="utf-8") as log:
            log.write("CID\tName\tReason\n")
            for cid, name, reason in failed_list:
                log.write(f"{cid}\t{name}\t{reason}\n")
        print(f"⚠️ {len(failed_list)} errores registrados en 'drug_generation_errors.log'")


# =============================================================================
#  EJECUCIÓN PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    # ---------------- Configuraciones de Rutas ----------------
    BASE_DIR = Path("data")
    REF_DIR = BASE_DIR / "ref_genome"
    LIB_DIR = Path("library")
    
    # Archivos de Entrada definidos por usuario
    GENE_VAR_TSV = BASE_DIR / "gene_var.tsv"
    DRUGS_TSV = BASE_DIR / "drugs.tsv"
    
    # Archivos de Referencia
    FASTA_FILE = REF_DIR / "genome.fna"
    GFF_FILE = REF_DIR / "gen_annotations.gff"
    PGX_FOLDER = BASE_DIR / "haplotype_variants"
    
    # Salidas
    GENE_OUT_DIR = LIB_DIR / "gene_graphs"
    DRUG_OUT_DIR = LIB_DIR / "drugs"
    PARQUET_FILE = LIB_DIR / "genome_library.parquet"

    # Crear directorios base
    LIB_DIR.mkdir(exist_ok=True)

    print("="*60)
    print("   🧬💊 UNIFIED GRAPH LIBRARY GENERATOR 💊🧬")
    print("="*60)

    # ---------------- 1. Pipeline de Genes ----------------
    gene_builder = GenomicGraphBuilder(FASTA_FILE, GFF_FILE, PGX_FOLDER)
    gene_builder.run_pipeline(GENE_VAR_TSV, PARQUET_FILE, GENE_OUT_DIR)

    # ---------------- 2. Pipeline de Fármacos -------------
    drug_builder = DrugGraphBuilder()
    drug_builder.run_pipeline(DRUGS_TSV, DRUG_OUT_DIR)

    print("\n✅ PROCESO FINALIZADO EXITOSAMENTE.")
    print(f"📂 Librería generada en: {LIB_DIR.resolve()}")