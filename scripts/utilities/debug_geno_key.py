# quick_debug_geno_key.py
"""
Script de verificación para la construcción de geno_key y matching con el índice de grafos.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter

# Imports del proyecto
from src.config. manager import LIBRARY
from src.data. graph_indexing import GraphIndexBuilder
from src.utils.io import DataLoaderUtils

# Paths
TRAIN_DATA_PATH = Path("train_data/train_data.tsv")
VARIANT_LIB = LIBRARY / "gene_graphs"

print("=" * 70)
print("🔍 VERIFICACIÓN DE GENO_KEY Y MATCHING CON ÍNDICE")
print("=" * 70)

# 1. Cargar DataFrame raw
print("\n📂 1. Cargando DataFrame...")
df_raw = pd.read_csv(TRAIN_DATA_PATH, sep="\t")
print(f"   Filas originales: {len(df_raw):,}")

# 2. Aplicar clean_and_prepare_data
print("\n🧹 2. Aplicando clean_and_prepare_data...")
df_clean = DataLoaderUtils.clean_and_prepare_data(df_raw)
print(f"   Filas después de limpieza: {len(df_clean):,}")

# 3. Construir índice de grafos
print("\n📚 3. Construyendo índice de grafos...")
index = GraphIndexBuilder.build_gene_variant_index(VARIANT_LIB)
total_variants = sum(len(v) for v in index.values())
print(f"   Genes indexados: {len(index)}")
print(f"   Variantes totales: {total_variants:,}")

# 4. Verificar matching
print("\n🎯 4. Verificando matching de geno_keys...")

unique_geno_keys = df_clean["geno_key"].unique()
print(f"   geno_keys únicos en DataFrame: {len(unique_geno_keys):,}")

found = []
not_found = []

for geno_key in unique_geno_keys:
    geno_str = str(geno_key)
    
    if "_" not in geno_str:
        not_found.append((geno_key, "NO_UNDERSCORE", None))
        continue
    
    gene, variant = geno_str.split("_", 1)
    
    if gene in index and variant in index[gene]:
        found.append(geno_key)
    else:
        not_found.append((geno_key, gene, variant))

match_rate = len(found) / len(unique_geno_keys) * 100

print(f"\n   ✅ Encontrados: {len(found):,} ({match_rate:.1f}%)")
print(f"   ❌ No encontrados: {len(not_found):,} ({100-match_rate:.1f}%)")

# 5. Análisis de no encontrados
if not_found: 
    print("\n" + "=" * 70)
    print("📊 5. ANÁLISIS DE NO ENCONTRADOS")
    print("=" * 70)
    
    # Agrupar por razón
    genes_not_in_index = [x for x in not_found if x[1] not in index and x[1] != "NO_UNDERSCORE"]
    variants_not_in_gene = [x for x in not_found if x[1] in index]
    no_underscore = [x for x in not_found if x[1] == "NO_UNDERSCORE"]
    
    print(f"\n   Sin underscore: {len(no_underscore)}")
    print(f"   Gen no existe en índice: {len(genes_not_in_index)}")
    print(f"   Variante no existe para el gen: {len(variants_not_in_gene)}")
    
    # Mostrar ejemplos de variantes no encontradas
    if variants_not_in_gene: 
        print("\n   📝 Ejemplos de variantes no encontradas (primeros 15):")
        for geno_key, gene, variant in variants_not_in_gene[:15]:
            available = list(index. get(gene, {}).keys())[:5]
            print(f"\n      geno_key: '{geno_key}'")
            print(f"      gene: '{gene}', variant buscado: '{variant}'")
            print(f"      variantes disponibles: {available}...")

# 6. Mostrar ejemplos exitosos
print("\n" + "=" * 70)
print("✅ 6. EJEMPLOS DE MATCHES EXITOSOS (primeros 10)")
print("=" * 70)

for geno_key in found[:10]:
    gene, variant = str(geno_key).split("_", 1)
    path = index[gene][variant]
    print(f"   '{geno_key}' -> {path. name}")

# 7. Verificar distribución por tipo de variante
print("\n" + "=" * 70)
print("📈 7. DISTRIBUCIÓN DE GENO_KEYS")
print("=" * 70)

# Contar star alleles vs rsIDs
star_alleles = [gk for gk in unique_geno_keys if "*" in str(gk)]
rs_ids = [gk for gk in unique_geno_keys if str(gk).split("_")[-1]. startswith("rs")]
otros = [gk for gk in unique_geno_keys if gk not in star_alleles and gk not in rs_ids]

print(f"\n   Star alleles (*): {len(star_alleles):,}")
print(f"   rsIDs (rs... ): {len(rs_ids):,}")
print(f"   Otros:  {len(otros):,}")

# Match rate por tipo
star_found = len([gk for gk in star_alleles if gk in found])
rs_found = len([gk for gk in rs_ids if gk in found])

print(f"\n   Match rate star alleles: {star_found}/{len(star_alleles)} ({star_found/max(len(star_alleles),1)*100:.1f}%)")
print(f"   Match rate rsIDs: {rs_found}/{len(rs_ids)} ({rs_found/max(len(rs_ids),1)*100:.1f}%)")

# 8. Resumen final
print("\n" + "=" * 70)
print("📋 8. RESUMEN FINAL")
print("=" * 70)

print(f"""
   DataFrame: 
   - Filas totales: {len(df_clean):,}
   - geno_keys únicos: {len(unique_geno_keys):,}
   
   Índice de grafos: 
   - Genes:  {len(index)}
   - Variantes: {total_variants:,}
   
   Matching: 
   - Encontrados: {len(found):,} ({match_rate:.1f}%)
   - No encontrados:  {len(not_found):,}
   
   Estado:  {'✅ OK' if match_rate > 80 else '⚠️ REVISAR' if match_rate > 50 else '❌ PROBLEMA'}
""")

# 9. Verificar una muestra de carga real
print("=" * 70)
print("🧪 9. TEST DE CARGA REAL DE GRAFOS")
print("=" * 70)

if found:
    import torch
    
    test_keys = found[:5]
    print(f"\n   Probando carga de {len(test_keys)} grafos...")
    
    for geno_key in test_keys: 
        gene, variant = str(geno_key).split("_", 1)
        path = index[gene][variant]
        
        try:
            data = torch.load(path, weights_only=False)
            n_nodes = data.x.shape[0] if hasattr(data, 'x') else 0
            n_edges = data.edge_index.shape[1] if hasattr(data, 'edge_index') else 0
            print(f"   ✅ '{geno_key}':  {n_nodes} nodes, {n_edges} edges")
        except Exception as e: 
            print(f"   ❌ '{geno_key}': Error - {e}")

print("\n" + "=" * 70)
print("🏁 VERIFICACIÓN COMPLETADA")
print("=" * 70)