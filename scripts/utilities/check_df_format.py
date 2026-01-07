import pandas as pd

# Carga tu csv
TRAIN_TSV_PATH = "train_data/train_data.tsv"
df = pd.read_csv(TRAIN_TSV_PATH, sep="\t")  # Pon el path real

# Busca filas que NO contienen "_"
columna_genes = "gene"  # Ajusta esto según tu CSV
columna_haplo = "genotype"  # Ajusta esto según tu CSV

columna_fusionada = df[[columna_genes, columna_haplo]].agg("_".join, axis=1)

errores = df[~columna_fusionada.astype(str).str.contains("_")]

if not errores.empty:
    print(f"Se encontraron {len(errores)} filas con formato incorrecto:")
    print(errores[columna_haplo].head())
else:
    print(
        "Todos los formatos parecen correctos. Revisa si hay espacios extra o caracteres invisibles."
    )
