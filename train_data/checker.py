import json

import pandas as pd

df = pd.read_csv("final_enriched_data.tsv", sep="\t", dtype=str)

df2 = pd.read_csv("snp_genes_table.tsv", sep="\t", dtype=str)

dict_id = {f"rs{str(row['SNP_Id'])}": row["Genalle"] for idx, row in df2.iterrows()}


# df[['Entity1_name', 'Entity2_name']] = df[['Entity1_name', 'Entity2_name']].map(lambda x: dict_id.get(x) if x in dict_id else x)

# df.to_csv('relationships_associated_corrected_mapped.tsv', sep='\t', index=False)
# 35320474
with open("test.json", "w") as f:
    json.dump(dict_id, f, indent=1)

print(df.sample(10))
