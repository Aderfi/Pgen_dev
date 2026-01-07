# check_dataframe.py
import pandas as pd

df = pd.read_csv("train_data/train_data.tsv", sep="\t")

print("=" * 60)
print("COLUMNAS DISPONIBLES:")
print(df.columns.tolist())

print("\n" + "=" * 60)
print("VERIFICANDO geno_key:")

if "geno_key" in df.columns:
    print("\n✅ Columna 'geno_key' existe")
    print(f"Ejemplos de geno_key:")
    for gk in df["geno_key"].dropna().unique()[:15]:
        print(f"  '{gk}'")
else:
    print("\n❌ Columna 'geno_key' NO existe")
    
    if "gene" in df.columns and "genotype" in df.columns:
        print("\nColumnas 'gene' y 'genotype' encontradas:")
        print(f"\nEjemplos de 'gene': {df['gene']. dropna().unique()[:10].tolist()}")
        print(f"Ejemplos de 'genotype': {df['genotype'].dropna().unique()[:15].tolist()}")
        
        # Simular geno_key
        print("\n📝 Simulando geno_key = gene + '_' + genotype:")
        for _, row in df[["gene", "genotype"]]. drop_duplicates().head(15).iterrows():
            simulated = f"{row['gene']}_{row['genotype']}"
            print(f"  '{simulated}'")

# Buscar columnas que podrían contener el alelo
print("\n" + "=" * 60)
print("BUSCANDO COLUMNA DE ALELO/VARIANTE:")
for col in df.columns:
    if any(x in col. lower() for x in ['allele', 'allelo', 'variant', 'haplo', 'geno', 'star']):
        print(f"\n  Columna:  '{col}'")
        print(f"  Ejemplos: {df[col].dropna().unique()[:10].tolist()}")