import pandas as pd
import os
import numpy as np

def process_genomic_data():
    # --- CONFIGURACIÓN ---
    files = {
        'drug': 'var_drug_ann.tsv',
        'fa': 'var_fa_ann.tsv',
        'pheno': 'var_pheno_ann.tsv'
    }
    
    # IMPORTANTE: Define aquí las columnas que son ÚNICAMENTE para identificar la fila.
    # Si lo dejas vacío [], el script intentará adivinar usando la intersección de los 3 archivos.
    # Ejemplos típicos: ['variant_id'] o ['chr', 'pos', 'ref', 'alt'] o ['rsid']
    ID_COLS = [] 

    dfs = {}
    print("--- 1. Cargando archivos TSV ---")
    for key, filepath in files.items():
        if os.path.exists(filepath):
            try:
                # Cargamos todo como string al principio para facilitar la concatenación posterior sin errores de tipo
                # Luego pandas inferirá tipos numéricos si es necesario al guardar, o puedes forzarlo.
                df = pd.read_csv(filepath, sep='\t', low_memory=False, dtype=str)
                dfs[key] = df
                print(f"✅ {key.upper()} cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
            except Exception as e:
                print(f"❌ Error cargando {filepath}: {e}")
                return
        else:
            print(f"❌ Archivo no encontrado: {filepath}")
            return

    # --- 2. DEFINIR LLAVES DE CRUCE (MERGE KEYS) ---
    if not ID_COLS:
        # Si no se definieron manualmente, usamos las columnas presentes en TODOS los archivos
        common_cols = set(dfs['drug'].columns)
        for key in ['fa', 'pheno']:
            common_cols = common_cols.intersection(dfs[key].columns)
        ID_COLS = list(common_cols)
        print(f"ℹ️ Llaves de cruce detectadas automáticamente: {ID_COLS}")
    else:
        print(f"ℹ️ Usando llaves de cruce manuales: {ID_COLS}")

    if not ID_COLS:
        print("❌ Error: No hay columnas comunes entre los 3 archivos para usar como llave (ID).")
        return

    # --- 3. FUNCIÓN DE FUSIÓN INTELIGENTE ---
    def merge_and_consolidate(df_left, df_right, keys, suffix_left='_old', suffix_right='_new'):
        """
        Realiza un outer merge y concatena el contenido de las columnas que chocan (no son keys).
        """
        # Realizar el merge (Outer Join)
        merged = pd.merge(df_left, df_right, on=keys, how='outer', suffixes=(suffix_left, suffix_right))
        
        # Detectar columnas duplicadas (las que pandas renombró con sufijos)
        # Buscamos columnas que terminen en suffix_left y vemos si existe su par suffix_right
        cols_to_fix = [c for c in merged.columns if c.endswith(suffix_left)]
        
        count_fused = 0
        for col_old in cols_to_fix:
            base_name = col_old[:-len(suffix_left)] # Nombre original (ej: 'Description')
            col_new = base_name + suffix_right
            
            if col_new in merged.columns:
                # Tenemos una colisión. Vamos a fusionarlas.
                # Lógica: Si ambos tienen dato: "A ; B". Si solo uno: "A" o "B". Si ninguno: ""
                
                # Reemplazar NaN con cadena vacía para poder sumar strings
                s1 = merged[col_old].fillna('')
                s2 = merged[col_new].fillna('')
                
                # Concatenación vectorizada (rápida)
                # Añadimos un separador temporal solo si ambos tienen datos
                separator = np.where((s1 != '') & (s2 != ''), '; ', '')
                
                # Crear la columna final combinada
                merged[base_name] = s1 + separator + s2
                
                # Limpiar posibles espacios extra si alguno estaba vacío
                merged[base_name] = merged[base_name].str.strip('; ')
                
                # Eliminar las columnas temporales con sufijos
                merged.drop([col_old, col_new], axis=1, inplace=True)
                count_fused += 1
        
        if count_fused > 0:
            print(f"   ↳ Se fusionaron (concatenaron) {count_fused} columnas con nombres repetidos.")
            
        return merged

    # --- 4. EJECUCIÓN DEL MERGE EN CADENA ---
    print("\n--- Iniciando Fusión (Outer Merge + Concatenación) ---")
    
    # Paso 1: Drug + Fa
    print("1. Fusionando 'Drug' con 'Fa'...")
    final_df = merge_and_consolidate(dfs['drug'], dfs['fa'], ID_COLS)
    
    # Paso 2: Resultado + Pheno
    print("2. Fusionando resultado con 'Pheno'...")
    final_df = merge_and_consolidate(final_df, dfs['pheno'], ID_COLS)

    # --- 5. LIMPIEZA FINAL Y GUARDADO ---
    # Reemplazar cadenas vacías con NaN para consistencia (opcional, depende de tu preferencia)
    final_df.replace('', pd.NA, inplace=True)

    output_file = 'final_genomic_data_merged.tsv'
    final_df.to_csv(output_file, sep='\t', index=False)
    
    print(f"\n✅ Archivo guardado exitosamente: {output_file}")
    print(f"Dimensión final: {final_df.shape}")
    print("Nota: Las columnas duplicadas se han concatenado separadas por '; '")

if __name__ == "__main__":
    process_genomic_data()