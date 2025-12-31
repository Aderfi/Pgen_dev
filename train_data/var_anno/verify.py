from pathlib import Path
import pandas as pd
import numpy as np

ruta_genes=Path('../dev_Pharmagen/library/gene_graphs')

lista_genes_dir = [d.name for d in ruta_genes.iterdir() if d.is_dir()]


df = pd.read_csv('shuffled_ready_data.tsv', sep='\t')

new_df = pd.DataFrame(columns=["snp", "snp_id", "chr", "pos", "variant", "variant_type", "gene", "clin_sig", "Ref_Allele", "Alt_Allele", "Star_Allele"])


new_df["gene"] = df["gene"].copy()
new_df["snp"] = df["genotype"].copy()

OUT_F="summary_data_frame.tsv"

new_df.to_csv(OUT_F, sep='\t', index=False)


