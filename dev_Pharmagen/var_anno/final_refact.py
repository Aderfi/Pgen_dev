import numpy as np
import pandas as pd

FILE = "final_data_clean.tsv"
FILE2 = "ready_data.tsv"
COLS_SAMPLE = [
    "drugs_cid",
    "drugs",
    "genotype",
    "gene",
    "alleles",
    "phenotype category",
    "direction of effect",
    "effect_function",
    "phenotype_product",
]

df = pd.read_csv(FILE, sep="\t")

counts = df.groupby("effect_function")["effect_function"].transform("count")

df_filtered = df[counts >= 14].copy()

tarkov = df_filtered[
    [
        "phenotype category",
        "direction of effect",
        "effect_function",
        "phenotype_product",
    ]
].copy()


for col in tarkov:
    print(f"\n {col}:")
    if tarkov[col].unique().size < 35:
        print(
            "\t -- Val Count \n\t", tarkov[col].value_counts()
        )  # Número de valores únicos
    else:
        print("\t -- Número de valores únicos:\n\t", tarkov[col].nunique())

    print("\t -- NaN count:\n\t", tarkov[col].isna().sum())  # Número de NaN

    print(
        "PORCENTAJE DE CELDAS VACIAS EN LA COLUMNA:  ",
        tarkov[col].isna().sum() / len(tarkov),
    )


# print(df[mask][COLS_SAMPLE].sample(10))

# df_filtered = df[~mask].copy()

df_filtered["phenotype category"].fillna("__UNK__", inplace=True)
df_filtered["direction of effect"].fillna("__UNDETERMINED__", inplace=True)
df_filtered["effect_function"].fillna("__NOTSTATED__", inplace=True)

df_filtered["drugs_cid"].fillna(np.nan, inplace=True)

df_filtered.to_csv(FILE2, sep="\t", index=False)
