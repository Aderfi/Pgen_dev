import json
from typing import Dict

import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer


class DynamicTargetEncoder:
    """
    Codificador de objetivos para modelos Multi-Tarea en PyTorch.
    Maneja columnas mixtas (Single-Label y Multi-Label) y valores nulos.
    """

    def __init__(self, task_config: dict[str, str]):
        """
        :param task_config: Diccionario {nombre_columna: tipo_tarea}
                            Tipos: 'multiclass' (Single-Label) o 'multilabel'.
        """
        self.task_config = task_config
        self.encoders = {}
        self.output_dims = {}

    def fit(self, df: pd.DataFrame):
        """Aprende los mapeos (String -> Número) de las columnas."""
        for col, task_type in self.task_config.items():
            if col not in df.columns:
                continue

            # Filtramos NaNs para el fit (aprendemos solo de etiquetas reales)
            valid_data = df[col].dropna()

            if task_type == "multiclass":
                le = LabelEncoder()
                le.fit(valid_data.astype(str))
                self.encoders[col] = le
                self.output_dims[col] = len(le.classes_)

            elif task_type == "multilabel":
                # Asumimos que los strings multilabel vienen separados por algo (ej. coma)
                # Ojo: Si ya son listas en el DF, omitir el .str.split
                if valid_data.dtype == "object" and isinstance(valid_data.iloc[0], str):
                    parsed_data = valid_data.str.split(",")  # Ajustar separador
                else:
                    parsed_data = valid_data

                mlb = MultiLabelBinarizer()
                mlb.fit(parsed_data)
                self.encoders[col] = mlb
                self.output_dims[col] = len(mlb.classes_)

    def transform(self, df: pd.DataFrame) -> dict[str, torch.Tensor]:
        """
        Convierte el DF en un diccionario de tensores listos para el modelo.
        Genera máscaras para ignorar datos faltantes durante el entrenamiento.
        """
        encoded_targets = {}
        masks = {}  # Para saber qué filas tienen dato real y cuáles son NaN

        for col, task_type in self.task_config.items():
            if col not in self.encoders:
                continue

            encoder = self.encoders[col]
            raw_values = df[col]

            # --- MANEJO DE MÁSCARAS (Crucial en datos clínicos reales) ---
            # 1 = Dato válido, 0 = Dato faltante (No calcular Loss aquí)
            mask = ~raw_values.isna()
            masks[f"{col}_mask"] = torch.from_numpy(mask.values).bool()

            # Rellenamos NaNs temporalmente para no romper el encoder
            # (El modelo ignorará estos valores gracias a la máscara)
            if task_type == "multiclass":
                fill_val = encoder.classes_[0]
                clean_vals = raw_values.fillna(fill_val).astype(str)
                encoded = encoder.transform(clean_vals)
                # Tensor de Enteros (Long) para CrossEntropy
                encoded_targets[col] = torch.tensor(encoded, dtype=torch.long)

            elif task_type == "multilabel":
                # Lógica similar para multilabel
                # Nota: MultiLabelBinarizer es robusto, pero hay que pasar listas vacías en los NaNs
                clean_vals = raw_values.apply(
                    lambda x: x.split(",")
                    if isinstance(x, str)
                    else (x if isinstance(x, list) else [])
                )
                encoded = encoder.transform(clean_vals)
                # Tensor de Floats para BCEWithLogitsLoss
                encoded_targets[col] = torch.tensor(encoded, dtype=torch.float)

        return encoded_targets, masks

    def save_mappings(self, path: str):
        """Guarda los diccionarios para poder decodificar las predicciones luego."""
        mappings = {}
        for col, enc in self.encoders.items():
            if isinstance(enc, LabelEncoder):
                mappings[col] = {"type": "multiclass", "classes": enc.classes_.tolist()}
            else:
                mappings[col] = {"type": "multilabel", "classes": enc.classes_.tolist()}

        with open(path, "w") as f:
            json.dump(mappings, f, indent=2)
