import pandas as pd

FILE = 'final_data_with_cid.tsv'
FILE2= 'final_data_clean.tsv'

df = pd.read_csv(FILE, sep='\t')

# Remove duplicate rows


df_cleaned = df.drop_duplicates()

df_cleaned["effect_function"] = df_cleaned['functional terms'].str.cat(df_cleaned[['pd/pk terms', 'side effect/efficacy/other']], sep='', na_rep='')

df_cleaned["phenotype_product"] = df_cleaned['gene/gene product'].str.cat(df_cleaned['phenotype'], sep='', na_rep='')


strip_df = df_cleaned[
    [
"drugs_cid",
"drugs",
"genotype",
"gene",
"alleles",
"phenotype category",
"direction of effect",
"effect_function",
"phenotype_product",
"metabolizer types",
"population types",
"population phenotypes or diseases",
"comparison allele(s) or genotype(s)",
"comparison metabolizer types",
"significance",
"is/is not associated",
"variant annotation id",
"pmid",
"sentence",
"notes",
]].copy()

strip_df.drop_duplicates(inplace=True)



# Reset index after dropping duplicates
df_final = strip_df.reset_index(drop=True)
print(df_final["drugs"].isna().sum() / len(df_final))

df_final.to_csv(FILE2, sep='\t', index=False)

"""

specialty population
metabolizer types
isplural
is/is not associated
direction of effect
pd/pk terms
multiple drugs and/or
comparison allele(s) or genotype(s)
comparison metabolizer types
assay type
functional terms
gene/gene product
cell type
side effect/efficacy/other
phenotype
multiple phenotypes and/or
population types
"""