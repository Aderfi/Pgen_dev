# Graph Builder.  
- #### root_dir/src/utils/library_creator.py

Este documento detalla el uso, configuración y estructura de datos necesaria para ejecutar el script `library_creator_polars.py`. Esta herramienta automatiza la construcción de grafos moleculares (fármacos) y grafos genómicos (variantes) utilizando **PyTorch Geometric**, **Polars** y **RDKit**.

El script tiene dos propósitos principales:

1.  **Módulo Genómico:** Procesa variantes (SNPs, Indels, Star Alleles) desde archivos TSV y VCF, las valida contra un genoma de referencia ``(GRCh38p14)`` y genera grafos dirigidos que representan la topología de la variante.

2.  **Módulo Farmacológico:** Convierte representaciones SMILES de fármacos en grafos moleculares con características atómicas y de enlace detalladas.

```text  
PROJECT/  
├── library_creator_polars.py  
└── data/                       <-- DIR BASE (Obligatorio)  
    ├── snp_data_output.tsv     <-- Entrada: Variantes generales  
    ├── drugs_cid.tsv           <-- Entrada: Fármacos  
    ├── ref_genome/             <-- Entrada: Referencia  
    │   ├── genome.fna          (FASTA del genoma, e.g., GRCh38)  
    │   └── gen_annotations.gff (Anotaciones GFF - opcional según lógica)  
    └── haplotype_variants/     <-- Entrada: Variantes PGx por gen  
        ├── CYP2D6/  
        │   ├── variante1.vcf  
        │   └── ...  
        ├── DPYD/  
        └── ...
```

Este script tiene como objetivo construir grafos para fármacos y genes a partir de archivos de variantes **(.vcf)** y tablas de compuestos.

## **REQUERIMIENTOS**

### **1. Librerías**

* polars: Procesamiento de datos de alto rendimiento y validación vectorizada.  
* torch & torch_geometric: Construcción de tensores y objetos Data (grafos).  
* rdkit: Química computacional (necesario para DrugGraphBuilder).  
* networkx: Construcción intermedia de la topología del grafo (nodos y aristas).  
* pyfaidx: Acceso indexado rápido a secuencias FASTA masivas.  
* tqdm: Visualización de barras de progreso.

### **2. Archivo de Variantes (TSV)**

El archivo snp_data_output.tsv debe cumplir estrictamente con el siguiente esquema:

| Columna | Tipo | Descripción |
| :---- | :---- | :---- |
| chr | str | Identificador del cromosoma (ej. "1", "X", "chr1"). Debe coincidir con el FASTA. |
| start_pos | int | Posición genómica absoluta (formato **1-based**). |
| Ref_Allele | str | Alelo de referencia (debe coincidir exactamente con la secuencia FASTA). |
| Alt_Allele | str | Alelo alternativo. |
| gene | str | (Opcional) Nombre del gen asociado para agrupar los grafos de salida. |
| variant_type | str | (Opcional) Tipo explícito: SNP, MNP, INS, DEL, STAR_ALLELE. |
| FXN_CLASS | str | (Opcional) Clasificación funcional (ej. "missense_variant", "intron_variant"). |

### **3. Archivo de Fármacos (TSV)**

El archivo drugs_cid.tsv es la fuente para los grafos moleculares:

| Columna | Tipo | Descripción |
| :---- | :---- | :---- |
| cid | str/int | Identificador único del compuesto (ej. PubChem CID). |
| smiles | str | Cadena SMILES válida del compuesto (ej. CC(=O)Oc1ccccc1C(=O)O). |
| cmpd_name_cleaned | str | Nombre limpio del fármaco (se usará para nombrar el archivo .pt). |

## ---

**INSTRUCCIONES DE USO**

### **Ejecución Básica**

El script está diseñado para correr desde la raíz del proyecto. Detectará automáticamente el sistema operativo (Debian/Linux o Windows) para gestionar la organización de carpetas.


***Any OS***
```bash
python -m src.utils.library_creator.py

```

### **Argumentos de Línea de Comandos**

El script soporta argumentos para documentación y pruebas:

* **Ayuda interactiva:** Muestra menús de documentación sobre inputs, outputs y uso.    
  ```bash
  python library_creator_polars.py --help
  ```

* **Modo Verificación:** Ejecuta el pipeline pero imprime logs específicos de prueba para un gen dado. Útil para validar si una carpeta en haplotype_variants se está leyendo bien.  
  ```bash  
  python library_creator_polars.py --verify "CYP2D6"
  ```

### **Salidas (Outputs)**

Los archivos se generarán en src/library/:

1. **genome_library.parquet**: DataFrame binario con todas las variantes validadas.  
2. **gene_graphs/**: Carpetas organizadas por gen conteniendo los archivos .pt (grafos).  
3. **drugs/**: Archivos .pt correspondientes a los fármacos.

## ---

**SOLUCIÓN A PROBLEMAS COMUNES**

| Error / Síntoma | Causa Probable | Solución |
| :---- | :---- | :---- |
| **Chr missing: X** | El nombre del cromosoma en el TSV no coincide con el encabezado del archivo FASTA. | Verifique si su FASTA usa nomenclatura chr1 o NC_000001. El script tiene una variable GLOBAL_CHROM_MAPPING para ajustar esto. |
| **Ref Mismatch** | La base de referencia en el TSV no es igual a la del genoma en esa posición. | Asegúrese de que start_pos sea **1-based**. El script resta 1 internamente. Verifique que usa la versión correcta del genoma (GRCh38 vs hg19). |
| **Invalid POS format** | Valores no numéricos en start_pos. | Limpie el TSV de valores "N/A", "-", o puntos en las columnas de posición. |
| **Grafos de fármacos faltantes** | RDKit no pudo parsear el SMILES. | Revise el archivo drug_generation_errors.log generado en la raíz para ver qué CIDs fallaron. |
| **Permisos (Linux/Debian)** | Error al ejecutar organize_genes.sh. | Asegúrese de tener permisos de escritura en la carpeta de salida o ejecute chmod +x manualmente si el script de Python falla al asignar permisos. |

<br>