import pandas as pd

df = pd.read_csv('ultra_ready_data.tsv', sep='\t')

shuffled_df = df.sample(frac=1, random_state=50889735).reset_index(drop=True)

shuffled_df.to_csv('train_data.tsv', sep='\t', index=False)