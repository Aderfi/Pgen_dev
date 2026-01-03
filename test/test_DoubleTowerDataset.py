import logging
import shutil
from pathlib import Path

import pandas as pd

from src.data.loaders import DoubleTowerDataset

if __name__ == "__main__":
    # Configuración de Logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    print("\n🔬 --- INICIANDO DIAGNÓSTICO DE DOUBLE TOWER DATASET ---\n")

    # 1. CONFIGURACIÓN DEL ENTORNO MOCK (Simulación de disco)
    # Definimos una ruta temporal local para no afectar tu sistema real
    TEST_DIR = Path("./test_env_temp")
    LIBRARY_MOCK = TEST_DIR / "library"

    # Sobreescribimos la variable global LIBRARY para que apunte al entorno de prueba
    # NOTA: En tu código real, asegúrate de que LIBRARY esté definida antes de la clase
    # LIBRARY = LIBRARY_MOCK

    """
    try:
        # Limpieza previa por si acaso
        if TEST_DIR.exists(): shutil.rmtree(TEST_DIR)
        
        # Crear estructura de directorios
        (LIBRARY_MOCK / "drugs").mkdir(parents=True)
        (LIBRARY_MOCK / "gene_graphs").mkdir(parents=True)
        
        logger.info(f"📁 Entorno temporal creado en: {TEST_DIR}")

        # 2. GENERACIÓN DE DATOS DUMMY (Archivos .pt y DataFrame)
        
        # A. Crear grafos dummy para Fármacos (ID_Nombre.pt)
        drug_ids = ["1001", "1002"]
        for d_id in drug_ids:
            dummy_graph = Data(x=torch.randn(5, 10), edge_index=torch.tensor([[0, 1], [1, 0]]))
            torch.save(dummy_graph, LIBRARY_MOCK / "drugs" / f"{d_id}_testdrug.pt")
            
        # B. Crear grafos dummy para Genes (Gen_Variante.pt)
        # Nota: Usamos el formato que tu _build_genes_index espera (GEN_VARIANTE)
        gene_files = ["CYP2D6_star4.pt", "CYP2D6_star1.pt", "HLA-B_star5701.pt"]
        for g_file in gene_files:
            dummy_graph = Data(x=torch.randn(8, 10), edge_index=torch.tensor([[0, 1], [1, 0]]))
            torch.save(dummy_graph, LIBRARY_MOCK / "gene_graphs" / g_file)
        """
    try:
        # C. Crear DataFrame de prueba
        df_data = {
            "compound_id": [68624, 1004, 99999999],  # 9999 no existe (prueba robustez)
            "gene_id": ["CYP2D6", "HLA-B", "CYP2D6"],
            "variant": ["*4", "*5701", "*X"],  # starX no existe (prueba robustez)
            "metabolizer": ["Poor", "Normal", "Ultra"],  # Target Single-label
            "side_effects": ["Headache|Nausea", "Nausea", None],  # Target Multi-label
        }
        df_test = pd.DataFrame(df_data)
        df_test = pd.concat(
            [
                df_test,
                pd.Series(
                    df_test["gene_id"].str.strip()
                    + "_"
                    + df_test["variant"].str.strip(),
                    name="gene_variant",
                ),
            ],
            axis=1,
        )
        logger.info("📊 DataFrame de prueba generado.")
        print(df_test.sample(3))

        # 3. INSTANCIACIÓN DEL DATASET
        logger.info("⚙️ Instanciando DoubleTowerDataset...")

        dataset = DoubleTowerDataset(
            df=df_test,
            drug_col="compound_id",
            haplo_col="gene_variant",
            target_cols=["metabolizer", "side_effects"],
            multilabel_cols=["side_effects"],
        )

        # 4. EVALUACIÓN DE COMPONENTES

        # A. Evaluar Encoding de Targets
        print("\n--- 1. Evaluación de Target Encoding ---")
        for col, encoder in dataset.encoders.items():
            print(f"✅ Columna '{col}': Encoder tipo {type(encoder).__name__}")
            if hasattr(encoder, "classes_"):
                print(f"   Clases detectadas: {encoder.classes_}")

        # B. Evaluar Carga de Datos (__getitem__)
        print("\n--- 2. Evaluación de Carga de Grafos (Item 0) ---")
        try:
            item = dataset[0]

            # Chequeo Fármaco
            drug_data = item["drug_data"]
            print(f"💊 Drug Data: {drug_data}")
            if drug_data.x.shape[0] == 1 and drug_data.x.shape[1] == 5:
                print(
                    "   ⚠️ (Aviso) Se cargó el grafo vacío por defecto (checkear IDs)."
                )
            else:
                print(
                    f"   ✅ Grafo de fármaco cargado correctamente. Nodos: {drug_data.num_nodes}"
                )

            # Chequeo Targets
            print(f"🎯 Targets: {item['targets']}")

            # Chequeo Variante (Aquí es donde tu código actual podría fallar)
            haplo_data = item["haplo_data"]
            print(f"🧬 Haplo Data: {haplo_data}")

        except AttributeError as e:
            print("\n❌ ERROR CRÍTICO DETECTADO EN LÓGICA DE VARIANTES:")
            print(f"   El error fue: {e}")
            print(
                "   👉 DIAGNÓSTICO: Tu método '_build_genes_index' crea un diccionario anidado:"
            )
            print("      Ej: {'CYP2D6': {'star4': Path(...)} }")
            print(
                "      Pero en '__getitem__', intentas acceder como si fuera una ruta directa:"
            )
            print(
                "      'haplo_path.exists()' falla porque 'haplo_path' es un diccionario, no un Path."
            )
            print(
                "   👉 SOLUCIÓN: Debes decidir si 'haplo_col' en el DF tiene 'CYP2D6' o 'CYP2D6_star4'."
            )

        except Exception as e:
            print(f"❌ Error inesperado: {e}")

        # 5. PRUEBA DE DATALOADER (Simulación de Batch)
        print("\n--- 3. Prueba de Batching (DataLoader) ---")
        try:
            from torch_geometric.loader import DataLoader

            loader = DataLoader(dataset, batch_size=2, shuffle=False)
            batch = next(iter(loader))
            print("📦 Batch generado exitosamente.")
            print(f"   Batch Size: {batch['drug_data'].batch.max() + 1}")
            print(
                f"   Targets Batch Shape (side_effects): {batch['targets']['side_effects'].shape}"
            )
        except Exception as e:
            print(
                f"⚠️ No se pudo crear el batch (probablemente por el error anterior): {e}"
            )

    finally:
        # LIMPIEZA
        # Comenta la siguiente línea si quieres inspeccionar los archivos generados
        if TEST_DIR.exists():
            shutil.rmtree(TEST_DIR)
        print("\n🧹 Entorno temporal eliminado. Test finalizado.")
