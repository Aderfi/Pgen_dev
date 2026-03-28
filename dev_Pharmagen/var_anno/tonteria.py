import numpy as np
import pandas as pd

# Configuración de rutas
DICT_FILE = "all_haplotypes_combined.tsv"
FILE = "summary_data_frame.tsv"
FILE_OUT = "summary_data_frame_relleno.tsv"


def main():
    # 1. Cargar datos
    # Usamos usecols para cargar solo lo necesario y ahorrar memoria
    df = pd.read_csv(FILE, sep="\t")
    dict_df = pd.read_csv(DICT_FILE, sep="\t")

    # 2. Mapeo de tipos de variantes (Pre-procesamiento del diccionario)
    # Es mucho más rápido mapear una vez en el diccionario que hacerlo fila por fila
    variant_mapping = {
        "substitution": "snp",
        "deletion": "del",
        "insertion": "ins",
        "indel": "indel",
    }
    dict_df["Type_mapped"] = (
        dict_df["Type"].map(variant_mapping).fillna(dict_df["Type"])
    )

    # 3. Renombrar columnas del diccionario para que coincidan con el DF original
    # Esto facilita el proceso de "rellenado" (coalesce)
    dict_df = dict_df.rename(
        columns={
            "rsID": "snp_dict",
            "ReferenceSequence": "chr_dict",
            "Variant Start": "pos_dict",
            "gene": "gene_dict",
            "Type_mapped": "variant_type_dict",
            "Reference Allele": "Ref_Allele_dict",
            "Variant Allele": "Alt_Allele_dict",
        }
    )

    # 4. Realizar el MERGE (Left Join)
    # Unimos por la columna de la variante (rsID/snp)
    merged_df = pd.merge(df, dict_df, on="snp", how="left")

    # 5. Rellenar valores nulos (Vectorización)
    # Si la columna original es nula o vacía, toma el valor del diccionario
    cols_to_fix = {
        "chr": "chr_dict",
        "pos": "pos_dict",
        "gene": "gene_dict",
        "variant_type": "variant_type_dict",
        "Ref_Allele": "Ref_Allele_dict",
        "Alt_Allele": "Alt_Allele_dict",
    }

    for target_col, dict_col in cols_to_fix.items():
        # Reemplazamos strings vacíos por NaN para que combine() funcione
        merged_df[target_col] = merged_df[target_col].replace("", np.nan)
        # combine_first toma el valor de la columna original y, si es NaN, usa la del diccionario
        merged_df[target_col] = merged_df[target_col].combine_first(merged_df[dict_col])

    # 6. Limpieza: Eliminar las columnas auxiliares del diccionario y guardar
    cols_to_drop = [c for c in merged_df.columns if c.endswith("_dict")]
    df_final = merged_df.drop(columns=cols_to_drop)

    df_final.to_csv(FILE_OUT, sep="\t", index=False)
    print(f"Proceso completado. Archivo guardado en {FILE_OUT}")


if __name__ == "__main__":
    main()
