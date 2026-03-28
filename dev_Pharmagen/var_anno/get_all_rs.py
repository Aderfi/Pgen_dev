import pandas as pd

FILE = "ultra_ready_data.tsv"

df = pd.read_csv(FILE, sep="\t")

genotypes = df["genotype"].unique().tolist()
list_rs = []

for gt in genotypes:
    if "|" in gt:
        list_rs.extend(gt.split("|"))
    else:
        list_rs.append(gt)

lista_cleaned = [x.strip() for x in list_rs if x.startswith("rs")]
list_rs = list(set(sorted(lista_cleaned)))


with open("all_data_rs.txt", "w") as f:
    for rs in list_rs:
        f.write(rs + "\n")
