import re

import pandas as pd

FILE = "ready_data.tsv"
FILE3 = "super_duper_ready_data.tsv"

NAN_TOKENS = set(
    ["__UNK__", "__UNDETERMINED__", "__NOTSTATED__", "nan", "NaN", "", " ", "NA", "N/A"]
)


prefixes = ["Disease", "Side Effect", "Efficacy", "Other", "PK"]
pattern = r",\s+(?=(?:" + "|".join(prefixes) + r"):)"


def clean_effect_function(val: pd.Series):
    if pd.isna(val[1]) or val[1] == "nan":
        return val[1]

    function_val = val[0]
    type_val = val[1]

    if function_val == "activity":
        return "enzyme"
    if function_val == "expression":
        return "gene"
    if function_val == "transcription":
        return "gene"

    return type_val


def clean_phenotypes_product(val):
    val = str(val)

    # 1. DETECCIÓN DE GEN: Si no tiene dos puntos ":", asumimos que es un Gen
    # y lo devolvemos tal cual (equivalente a tu "continue")
    if ":" not in val:
        return val

    # 2. TRANSFORMACIÓN DE LISTA:
    # Reemplazamos SOLAMENTE las comas que separan categorías
    return re.sub(pattern, "|", val)


def map_value(x):
    if pd.isna(x) or x == "nan":
        return x
    if ":" not in x:
        return x.upper()

    x = str(x).lower()

    if "|" in x:
        values_list = x.split("|")
    else:
        values_list = [x]

    category_list = []
    for val in values_list:
        cat = val.split(":")[0].strip()
        category_list.append(cat)

    category_list_return = [
        cat.strip().replace(r"\s+", "_") for cat in category_list if cat
    ]

    return "|".join(set(category_list_return))


df = pd.read_csv(FILE, sep="\t")

for col in df.columns:
    if col == "drugs_cid":
        continue
    df[col] = df[col].astype(str).str.strip()

df["population types"] = df["population types"].str.replace(
    r"\bin\s+(\w+)\s+[a-z]+\b",  # Patrón de búsqueda
    r"\1",  # Reemplazo (el grupo capturado)
    regex=True,
    flags=re.IGNORECASE,  # Opcional: para que detecte "In" mayúscula también
)

df["effect_function"] = (
    df["effect_function"]
    .str.replace(r"\b\s*(?:to|of)\b", "", regex=True, flags=re.IGNORECASE)
    .str.strip()
)

df["drugs"] = df["drugs"].str.replace(r"\s*,\s*", "|", regex=True)
df["phenotype category"] = df["phenotype category"].str.replace(
    r"\s*,\s*", "|", regex=True
)
df["gene"] = df["gene"].str.replace(r"\s*,\s*", "|", regex=True)
# df["phenotype_product"] = df["phenotype_product"].str.replace(r'\s*,\s*', '|', regex=True)
df["genotype"] = df["genotype"].str.replace(r"\s*,\s*", "|", regex=True)
# df["population phenotypes or diseases"] = df["population phenotypes or diseases"].str.replace(r'\s*,\s*', '|', regex=True)

df.rename(lambda x: re.sub(r"\s+", "_", x.strip().lower()), axis=1, inplace=True)

df.insert(8, "effect_type", "")


df["phenotype_product"] = df["phenotype_product"].apply(clean_phenotypes_product)
df["population_phenotypes_or_diseases"] = df["population_phenotypes_or_diseases"].apply(
    clean_phenotypes_product
)

df["effect_type"] = df["phenotype_product"].apply(map_value)

df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)

# df["effect_type"] = df[["effect_function", "effect_type"]].apply(clean_effect_function, raw=True, axis=0)

# df_shuffled = df.sample(frac=1, random_state=6203573).reset_index(drop=True)

# df_shuffled.to_csv('shuffled_ready_data.tsv', sep='\t', index=False)
with pd.option_context("display.max_rows", None, "display.max_columns", None):
    print(df["population_phenotypes_or_diseases"].sample(20))
