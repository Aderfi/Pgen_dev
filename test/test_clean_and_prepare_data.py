# test_clean_and_prepare_data.py
"""
Test suite para clean_and_prepare_data con STAR_ALLELE_MAP. 

Ejecutar:
    python test_clean_and_prepare_data.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ==============================================================================
# CONFIGURACIÓN - Copiar las constantes necesarias
# ==============================================================================

STAR_ALLELE_MAP = {
    "CYP2D6*3": "rs35742686",
    "CYP2D6*4": "rs3892097",
    "CYP2D6*6": "rs5030656",
    "CYP2D6*9": "rs5030655",
    "CYP2D6*10": "rs1065852",
    "CYP2D6*17": "rs28371706",
    "CYP2D6*29": "rs55811643",
    "CYP2D6*41": "rs28371725",
    "CYP2C19*2": "rs4244285",
    "CYP2C19*3": "rs4986893",
    "CYP2C19*4": "rs28399504",
    "CYP2C19*17": "rs12248560",
    "CYP2C9*2": "rs1799853",
    "CYP2C9*3": "rs1057910",
    "CYP2C9*5": "rs28371686",
    "CYP2C9*6": "rs9332131",
    "CYP2C9*8": "rs7900194",
    "CYP2C9*11": "rs28371685",
    "DPYD*2A": "rs3918290",
    "DPYD*13": "rs55886062",
    "DPYD*9A": "rs1801159",
    "c.2846A>T": "rs67376798",
    "c.1236G>A": "rs75017182",
    "SLCO1B1*5": "rs4149056",
    "SLCO1B1*15": "rs4149056|rs2306283",
    "SLCO1B1*37": "rs2306283",
    "CYP3A5*3": "rs776746",
    "CYP3A5*6": "rs10276036",
    "CYP3A5*7": "rs41303343",
    "NUDT15*2": "rs116855232|rs147390019",
    "NUDT15*3": "rs116855232",
    "NAT2*5": "rs1801280",
    "NAT2*6": "rs1799930",
    "NAT2*7": "rs1799931",
    "NAT2*12": "rs1208",
    "NAT2*14": "rs1801279",
    "CYP4F2*3": "rs2108622",
    "CYP2B6*6": "rs2279343|rs3211371",
    "CYP2B6*18": "rs28399499",
    "CYP1A2*1F": "rs762551",
}

# Crear mapa inverso
RSID_TO_STAR_ALLELES:  dict[str, list[str]] = {}
for star_allele, rsids in STAR_ALLELE_MAP.items():
    for rsid in rsids. split("|"):
        rsid = rsid.strip()
        if rsid not in RSID_TO_STAR_ALLELES:
            RSID_TO_STAR_ALLELES[rsid] = []
        RSID_TO_STAR_ALLELES[rsid].append(star_allele)

MULTI_LABEL_COLS = {"phenotype_category"}  # Ajustar según tu proyecto


# ==============================================================================
# FUNCIÓN A TESTEAR (copiar tu implementación aquí)
# ==============================================================================

def clean_and_prepare_data(
    df:  pd.DataFrame, stratify_col: list[str] | str | None = None
) -> pd.DataFrame:
    """Versión optimizada con explosión de filas."""
    work_df = df.copy()

    # 1-2. Limpieza inicial
    count_pre = len(work_df)
    work_df = work_df.dropna(subset=["gene", "genotype"])
    
    mask_valid = (
        (work_df["gene"].str.strip() != "") & 
        (work_df["genotype"].str.strip() != "")
    )
    work_df = work_df[mask_valid].copy()
    
    print(f"ℹ️  Eliminadas {count_pre - len(work_df)} filas inválidas.")
    count_before = len(work_df)

    # 3. CONSTRUCCIÓN DE GENO_KEY
    genes = work_df["gene"].astype(str).str.strip()
    genotypes = (
        work_df["genotype"].astype(str)
        .str.strip()
        .str.replace(r"^REF_SEQ\|", "", regex=True)
    )
    
    has_alleles_col = "alleles" in work_df.columns
    if has_alleles_col:
        alleles = work_df["alleles"]. fillna("").astype(str).str.strip()
    else:
        alleles = pd. Series("", index=work_df.index)

    results = []
    
    for idx, (gene, genotype, allele_str) in enumerate(zip(genes, genotypes, alleles)):
        row_data = work_df.iloc[idx]. to_dict()
        geno_keys = set()
        
        # Prioridad 1: Star alleles directos
        if "*" in allele_str:
            for allele in allele_str.split("/"):
                allele = allele.strip()
                if "*" in allele: 
                    geno_keys. add(f"{gene}_{allele}")
        
        # Prioridad 2: rsID -> star allele
        for rsid in genotype.split("|"):
            rsid = rsid.strip()
            if rsid in RSID_TO_STAR_ALLELES:
                for star_allele in RSID_TO_STAR_ALLELES[rsid]:
                    if "*" in star_allele:
                        star_suffix = "*" + star_allele.split("*")[-1]
                        geno_keys. add(f"{gene}_{star_suffix}")
                    else: 
                        geno_keys. add(f"{gene}_{star_allele}")
            elif not geno_keys:
                geno_keys.add(f"{gene}_{rsid}")
        
        if not geno_keys:
            geno_keys.add(f"{gene}_{genotype.split('|')[0]}")
        
        for gk in geno_keys: 
            row_copy = row_data.copy()
            row_copy["geno_key"] = gk
            results.append(row_copy)
    
    work_df = pd.DataFrame(results)
    work_df = work_df.drop_duplicates()
    
    print(f"ℹ️  Expansión: {count_before} -> {len(work_df)} filas ({len(work_df) - count_before:+d})")

    return work_df


# ==============================================================================
# TESTS
# ==============================================================================

def print_header(title: str):
    """Imprime encabezado de sección."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_result(test_name: str, passed: bool, details: str = ""):
    """Imprime resultado de test."""
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{status} | {test_name}")
    if details:
        print(f"       {details}")


def test_star_allele_direct():
    """Test 1: Star allele directo desde columna 'alleles'."""
    print_header("TEST 1: Star Allele Directo")
    
    df = pd.DataFrame({
        "gene": ["CYP2C9", "CYP2D6"],
        "genotype": ["rs9999999", "rs8888888"],  # rsIDs que NO están en el mapa
        "alleles":  ["*3", "*4/*4"],
    })
    
    print("Input:")
    print(df.to_string(index=False))
    print()
    
    result = clean_and_prepare_data(df)
    
    print("\nOutput:")
    print(result[["gene", "genotype", "alleles", "geno_key"]].to_string(index=False))
    
    # Verificaciones
    expected_keys = {"CYP2C9_*3", "CYP2D6_*4"}
    actual_keys = set(result["geno_key"].unique())
    
    passed = expected_keys == actual_keys
    print_result(
        "Star alleles extraídos correctamente",
        passed,
        f"Esperado: {expected_keys}, Obtenido: {actual_keys}"
    )
    
    return passed


def test_rsid_to_star_conversion():
    """Test 2: Conversión de rsID a star allele via STAR_ALLELE_MAP."""
    print_header("TEST 2: Conversión rsID -> Star Allele")
    
    df = pd.DataFrame({
        "gene": ["CYP2D6", "CYP2C9", "CYP2C19"],
        "genotype": ["rs3892097", "rs1057910", "rs4244285"],
        "alleles": [None, None, None],
    })
    
    print("Input:")
    print(df.to_string(index=False))
    print()
    
    result = clean_and_prepare_data(df)
    
    print("\nOutput:")
    print(result[["gene", "genotype", "alleles", "geno_key"]].to_string(index=False))
    
    # Verificaciones
    expected_keys = {"CYP2D6_*4", "CYP2C9_*3", "CYP2C19_*2"}
    actual_keys = set(result["geno_key"]. unique())
    
    passed = expected_keys == actual_keys
    print_result(
        "rsIDs convertidos a star alleles",
        passed,
        f"Esperado:  {expected_keys}, Obtenido: {actual_keys}"
    )
    
    return passed


def test_multiple_rsids_expansion():
    """Test 3: Múltiples rsIDs generan múltiples filas."""
    print_header("TEST 3: Explosión de Múltiples rsIDs")
    
    df = pd.DataFrame({
        "gene": ["CYP2D6"],
        "genotype": ["rs3892097|rs35742686|rs5030655"],  # *4, *3, *9
        "alleles": [None],
    })
    
    print("Input:")
    print(df.to_string(index=False))
    print()
    
    result = clean_and_prepare_data(df)
    
    print("\nOutput:")
    print(result[["gene", "genotype", "geno_key"]].to_string(index=False))
    
    # Verificaciones
    expected_keys = {"CYP2D6_*4", "CYP2D6_*3", "CYP2D6_*9"}
    actual_keys = set(result["geno_key"].unique())
    
    passed = expected_keys == actual_keys
    print_result(
        "Múltiples rsIDs expandidos correctamente",
        passed,
        f"Esperado: {expected_keys}, Obtenido:  {actual_keys}"
    )
    
    # Verificar número de filas
    passed_count = len(result) == 3
    print_result(
        "Número correcto de filas generadas",
        passed_count,
        f"Esperado: 3, Obtenido: {len(result)}"
    )
    
    return passed and passed_count


def test_rsid_fallback():
    """Test 4: rsID desconocido se mantiene como fallback."""
    print_header("TEST 4: Fallback para rsID Desconocido")
    
    df = pd.DataFrame({
        "gene": ["UNKNOWN_GENE"],
        "genotype": ["rs9999999"],  # rsID que NO está en el mapa
        "alleles": ["CC"],  # Sin star allele
    })
    
    print("Input:")
    print(df.to_string(index=False))
    print()
    
    result = clean_and_prepare_data(df)
    
    print("\nOutput:")
    print(result[["gene", "genotype", "alleles", "geno_key"]].to_string(index=False))
    
    # Verificaciones
    expected_key = "UNKNOWN_GENE_rs9999999"
    actual_key = result["geno_key"].iloc[0]
    
    passed = expected_key == actual_key
    print_result(
        "rsID desconocido usado como fallback",
        passed,
        f"Esperado:  '{expected_key}', Obtenido: '{actual_key}'"
    )
    
    return passed


def test_ref_seq_prefix_removal():
    """Test 5: Prefijo REF_SEQ| se elimina correctamente."""
    print_header("TEST 5: Eliminación de Prefijo REF_SEQ|")
    
    df = pd.DataFrame({
        "gene": ["CYP2C9"],
        "genotype": ["REF_SEQ|rs1057910"],  # Con prefijo REF_SEQ|
        "alleles": [None],
    })
    
    print("Input:")
    print(df.to_string(index=False))
    print()
    
    result = clean_and_prepare_data(df)
    
    print("\nOutput:")
    print(result[["gene", "genotype", "geno_key"]].to_string(index=False))
    
    # Verificaciones
    expected_key = "CYP2C9_*3"
    actual_key = result["geno_key"].iloc[0]
    
    passed = expected_key == actual_key
    print_result(
        "Prefijo REF_SEQ| eliminado y rsID convertido",
        passed,
        f"Esperado:  '{expected_key}', Obtenido: '{actual_key}'"
    )
    
    return passed


def test_duplicate_alleles():
    """Test 6: Alelos duplicados se dedupliccan."""
    print_header("TEST 6: Deduplicación de Alelos")
    
    df = pd.DataFrame({
        "gene": ["CYP2C9"],
        "genotype": ["rs9999999"],
        "alleles": ["*3/*3"],  # Mismo alelo dos veces
    })
    
    print("Input:")
    print(df.to_string(index=False))
    print()
    
    result = clean_and_prepare_data(df)
    
    print("\nOutput:")
    print(result[["gene", "genotype", "alleles", "geno_key"]]. to_string(index=False))
    
    # Verificaciones
    passed = len(result) == 1
    print_result(
        "Solo una fila generada para *3/*3",
        passed,
        f"Esperado:  1 fila, Obtenido:  {len(result)} filas"
    )
    
    return passed


def test_nan_handling():
    """Test 7: Filas con NaN se eliminan."""
    print_header("TEST 7: Manejo de NaN")
    
    df = pd.DataFrame({
        "gene": ["CYP2D6", None, "CYP2C9", ""],
        "genotype": ["rs3892097", "rs1234", None, "rs5678"],
        "alleles": [None, None, None, None],
    })
    
    print("Input:")
    print(df.to_string(index=False))
    print()
    
    result = clean_and_prepare_data(df)
    
    print("\nOutput:")
    print(result[["gene", "genotype", "geno_key"]].to_string(index=False))
    
    # Verificaciones
    passed = len(result) == 1  # Solo la primera fila es válida
    print_result(
        "Filas con NaN/vacío eliminadas",
        passed,
        f"Esperado: 1 fila, Obtenido: {len(result)} filas"
    )
    
    return passed


def test_priority_alleles_over_rsid():
    """Test 8: Columna 'alleles' tiene prioridad sobre rsID."""
    print_header("TEST 8: Prioridad de 'alleles' sobre rsID")
    
    df = pd. DataFrame({
        "gene": ["CYP2D6"],
        "genotype": ["rs3892097"],  # Esto sería *4
        "alleles": ["*10"],  # Pero alleles dice *10
    })
    
    print("Input:")
    print(df.to_string(index=False))
    print()
    
    result = clean_and_prepare_data(df)
    
    print("\nOutput:")
    print(result[["gene", "genotype", "alleles", "geno_key"]].to_string(index=False))
    
    # Verificaciones - debería tener AMBOS porque ambos son válidos
    expected_keys = {"CYP2D6_*10", "CYP2D6_*4"}
    actual_keys = set(result["geno_key"].unique())
    
    passed = expected_keys == actual_keys
    print_result(
        "Ambos alleles y rsID convertido incluidos",
        passed,
        f"Esperado: {expected_keys}, Obtenido: {actual_keys}"
    )
    
    return passed


def test_mixed_scenario():
    """Test 9: Escenario mixto realista."""
    print_header("TEST 9: Escenario Mixto Realista")
    
    df = pd.DataFrame({
        "gene": ["CYP2D6", "CYP2C9", "CYP2C19", "NAT2", "SLCO1B1"],
        "genotype": [
            "rs3892097|rs35742686",  # *4 y *3
            "REF_SEQ|rs1057910",      # *3 con prefijo
            "rs4244285",              # *2
            "rs1801280|rs1799930",    # *5 y *6
            "rs4149056",              # *5 y *15
        ],
        "alleles": [
            None,
            "*3/*3",
            None,
            None,
            None,
        ],
    })
    
    print("Input:")
    print(df.to_string(index=False))
    print()
    
    result = clean_and_prepare_data(df)
    
    print("\nOutput:")
    print(result[["gene", "genotype", "alleles", "geno_key"]].to_string(index=False))
    
    # Verificaciones
    print(f"\nTotal filas generadas: {len(result)}")
    print(f"geno_keys únicos: {result['geno_key'].unique().tolist()}")
    
    # Verificar que se generaron múltiples filas
    passed = len(result) > 5  # Debería haber más de 5 filas por la expansión
    print_result(
        "Expansión correcta en escenario mixto",
        passed,
        f"Filas generadas: {len(result)} (esperado: > 5)"
    )
    
    return passed


def test_with_real_data():
    """Test 10: Prueba con datos reales (si existe el archivo)."""
    print_header("TEST 10: Datos Reales")
    
    data_path = Path("train_data/train_data.tsv")
    
    if not data_path.exists():
        print("⚠️  Archivo de datos reales no encontrado.  Saltando test.")
        return True
    
    df = pd. read_csv(data_path, sep="\t")
    print(f"Cargadas {len(df)} filas del archivo real.")
    
    # Tomar muestra para test rápido
    df_sample = df.head(100)
    
    result = clean_and_prepare_data(df_sample)
    
    print(f"\nMuestra de geno_keys generados:")
    for gk in result["geno_key"]. unique()[:15]: 
        print(f"  - {gk}")
    
    # Verificar que hay star alleles
    star_allele_count = result["geno_key"].str. contains(r"\*", regex=True).sum()
    rsid_count = result["geno_key"].str.contains(r"rs\d+", regex=True).sum()
    
    print(f"\nEstadísticas:")
    print(f"  - Filas con star allele: {star_allele_count}")
    print(f"  - Filas con rsID fallback: {rsid_count}")
    print(f"  - Total filas: {len(result)}")
    
    passed = star_allele_count > 0
    print_result(
        "Se generaron geno_keys con star alleles",
        passed,
        f"Star alleles encontrados: {star_allele_count}"
    )
    
    return passed


# ==============================================================================
# MAIN
# ==============================================================================

def run_all_tests():
    """Ejecuta todos los tests."""
    print("\n" + "=" * 70)
    print("  SUITE DE TESTS:  clean_and_prepare_data")
    print("=" * 70)
    
    tests = [
        ("Star Allele Directo", test_star_allele_direct),
        ("Conversión rsID -> Star", test_rsid_to_star_conversion),
        ("Explosión Múltiples rsIDs", test_multiple_rsids_expansion),
        ("Fallback rsID Desconocido", test_rsid_fallback),
        ("Eliminación REF_SEQ|", test_ref_seq_prefix_removal),
        ("Deduplicación Alelos", test_duplicate_alleles),
        ("Manejo de NaN", test_nan_handling),
        ("Prioridad alleles vs rsID", test_priority_alleles_over_rsid),
        ("Escenario Mixto", test_mixed_scenario),
        ("Datos Reales", test_with_real_data),
    ]
    
    results = []
    for name, test_func in tests: 
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n❌ ERROR en '{name}': {e}")
            results.append((name, False))
    
    # Resumen final
    print("\n" + "=" * 70)
    print("  RESUMEN DE RESULTADOS")
    print("=" * 70)
    
    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)
    
    for name, passed in results:
        status = "✅" if passed else "❌"
        print(f"  {status} {name}")
    
    print("\n" + "-" * 70)
    print(f"  Total: {passed_count}/{total_count} tests pasados")
    
    if passed_count == total_count: 
        print("  🎉 ¡Todos los tests pasaron!")
    else:
        print(f"  ⚠️  {total_count - passed_count} test(s) fallaron")
    
    print("=" * 70 + "\n")
    
    return passed_count == total_count


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)