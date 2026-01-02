import torch
from pathlib import Path
from tqdm import tqdm

# Ajusta la ruta a tu carpeta real de drogas
DRUG_DIR = Path("src/library/drugs")
EXPECTED_EDGE_DIM = 7

print(f"🔍 Escaneando {DRUG_DIR} buscando grafos con dimensión != {EXPECTED_EDGE_DIM}...")

bad_files = []

for pt_file in tqdm(list(DRUG_DIR.glob("*.pt"))):
    try:
        data = torch.load(pt_file, weights_only=False)
        
        # Verificar si tiene atributos de arista
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            if data.edge_attr.numel() > 0:
                curr_dim = data.edge_attr.shape[1]
                if curr_dim != EXPECTED_EDGE_DIM:
                    print(f"⚠️  Detectado archivo antiguo/corrupto: {pt_file.name} (Dim: {curr_dim})")
                    bad_files.append(pt_file)
    except Exception as e:
        print(f"❌ Error leyendo {pt_file.name}: {e}")

if bad_files:
    print(f"\n🗑️ Se encontraron {len(bad_files)} archivos incompatibles.")
    # Descomenta la siguiente línea si quieres borrarlos automáticamente
    # for f in bad_files: f.unlink()
    # print("Archivos eliminados.")
else:
    print("\n✅ Todos los archivos de drogas tienen la dimensión correcta (7).")