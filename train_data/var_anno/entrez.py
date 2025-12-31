import pandas as pd
from Bio import Entrez
import time
import numpy as np

# ================= CONFIGURACIÓN =================
# NCBI requiere un email válido para usar E-utils
Entrez.email = "zeooloo@gmail.com"  # <--- POR FAVOR, CAMBIA ESTO
# Clave API opcional pero recomendada (aumenta el límite de 3 a 10 req/seg)
# Entrez.api_key = "tu_api_key" 

INPUT_FILE = "summary_data_frame.tsv"
OUTPUT_FILE = "summary_data_frame_relleno.tsv"
BATCH_SIZE = 50  # Cantidad de SNPs a consultar por llamada (respetuoso con el servidor)

def fetch_snp_details(rs_ids):
    """
    Consulta dbSNP vía E-utilities (esummary) para una lista de rsIDs.
    Retorna un diccionario mapeado: {rs_id: {chr, pos, gene, clin_sig, ...}}
    """
    if not rs_ids:
        return {}
    
    mapping_results = {}
    
    try:
        # Convertir lista a string separado por comas
        ids_str = ",".join([x.replace('rs', '') for x in rs_ids]) # dbSNP usa números, a veces 'rs' sobra
        
        # Llamada a la API
        handle = Entrez.esummary(db="snp", id=ids_str)
        record = Entrez.read(handle)
        handle.close()
        
        # Parseo de la respuesta (Estructura de DocumentSummary de dbSNP)
        for doc in record:
            # El ID devuelto por NCBI es numérico
            rs_key = f"rs{doc['Id']}"
            
            # Extracción segura de datos
            # Nota: La estructura del JSON de dbSNP puede variar, usamos .get()
            chrom = doc.get('Chromosome', '')
            pos = doc.get('BP', '')  # Base Position
            
            # Genes: suelen venir en una lista de diccionarios 'Genes'
            genes = []
            if 'Genes' in doc:
                for g in doc['Genes']:
                    if 'name' in g:
                        genes.append(g['name'])
            gene_str = "|".join(genes) if genes else ''
            
            # Significado clínico: suele estar en 'ClinicalSignificance'
            clin_sig = doc.get('ClinicalSignificance', '')
            
            mapping_results[rs_key] = {
                'chr': chrom,
                'pos': pos,
                'gene': gene_str,
                'clin_sig': clin_sig
            }
            
    except Exception as e:
        print(f"Error consultando lote: {e}")
        # En producción, podrías querer reintentar o loguear errores específicos
        
    return mapping_results

def main():
    print("Cargando archivo...")
    # Cargar el TSV. 'snp' parece ser la columna clave según tu archivo.
    df = pd.read_csv(INPUT_FILE, sep='\t')
    
    # Identificar filas que tienen un rsID válido en la columna 'snp'
    # Asumimos que la columna se llama 'snp' o 'snp_id' (ajustar según header real)
    target_col = 'snp' if 'snp' in df.columns else 'snp_id'
    
    # Crear máscara para rsIDs (empiezan por 'rs' y seguido de números)
    # Se usa 'na=False' para ignorar valores vacíos
    mask = df[target_col].str.startswith('rs', na=False)
    
    # Obtener lista única de rsIDs a consultar
    unique_rs_ids = df.loc[mask, target_col].unique().tolist()
    print(f"Se encontraron {len(unique_rs_ids)} rsIDs únicos para consultar.")
    
    # Procesar en lotes
    mapped_data = {}
    total = len(unique_rs_ids)
    
    for i in range(0, total, BATCH_SIZE):
        batch = unique_rs_ids[i : i + BATCH_SIZE]
        print(f"Procesando lote {i} a {min(i+BATCH_SIZE, total)}...")
        
        results = fetch_snp_details(batch)
        mapped_data.update(results)
        
        # Respetar Rate Limiting de NCBI (max 3 req/s sin API key)
        time.sleep(0.5) 

    print("Mapeo finalizado. Rellenando DataFrame...")

    # Rellenar el DataFrame original
    # Iteramos sobre las columnas que queremos rellenar
    for idx, row in df.iterrows():
        rs_id = row[target_col]
        
        # Solo intentamos rellenar si es un rsID y tenemos datos
        if rs_id in mapped_data:
            data = mapped_data[rs_id]
            
            # Actualizamos si el valor original está vacío o es NaN
            if pd.isna(row['chr']) or row['chr'] == '':
                df.at[idx, 'chr'] = data['chr']
            
            if pd.isna(row['pos']) or row['pos'] == '':
                df.at[idx, 'pos'] = data['pos']
                
            # A veces ya tienes el gen, decides si sobrescribir o no.
            # Aquí rellenamos solo si falta.
            if pd.isna(row['gene']) or row['gene'] == '':
                df.at[idx, 'gene'] = data['gene']
                
            if pd.isna(row['clin_sig']) or row['clin_sig'] == '':
                df.at[idx, 'clin_sig'] = data['clin_sig']

    # Guardar
    df.to_csv(OUTPUT_FILE, sep='\t', index=False)
    print(f"Archivo guardado exitosamente en: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()