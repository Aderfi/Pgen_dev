# Pharmagen - Pharmacogenetic Prediction and Therapeutic Efficacy
# Copyright (C) 2025 Adrim Hamed Outmani
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


import logging
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

from src.config import get_model_config, get_settings
from src.model.architectures.gnn import PharmagenTwoTower
from src.model.architectures.layers import create_gnn_model

logger = logging.getLogger(__name__)


UNKNOWN_TOKEN = "__UNKNOWN__"


class PGenPredictor:
    """
    Inference engine for a trained PharmagenTwoTower model.
    Loads the model and encoders once; reuses them across predictions.
    """

    def __init__(self, model_name: str, device: str | None = None):
        self.model_name = model_name
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        logger.info("Initialising PGenPredictor for '%s' on %s", model_name, self.device)

        self.config = get_model_config(model_name)
        self.feature_cols = [c.lower() for c in self.config.features]
        self.target_cols = [t.lower() for t in self.config.targets]
        self.params = self.config.params

        self.encoders = self._load_encoders()
        self.model = self._load_model()
        self.model.eval()

    def _load_encoders(self) -> dict[str, LabelEncoder | MultiLabelBinarizer]:
        """Load encoders and patch in the UNKNOWN_TOKEN class."""
        enc_path = get_settings().paths.encoders / f"encoders_{self.model_name}.pkl"

        if not enc_path.exists():
            raise FileNotFoundError(f"Encoders file not found: {enc_path}")

        encoders = joblib.load(enc_path)

        # Patch UNKNOWN_TOKEN into LabelEncoders so unseen values are handled gracefully.
        for col, enc in encoders.items():
            if isinstance(enc, LabelEncoder):
                if UNKNOWN_TOKEN not in enc.classes_:
                    # Extend the numpy array in-place — avoids a full copy.
                    enc.classes_ = np.append(enc.classes_, UNKNOWN_TOKEN)

        return encoders

    def _load_model(self) -> PharmagenTwoTower:
        """Instantiate the architecture and load saved weights."""
        n_features = {
            col: len(self.encoders[col].classes_)
            for col in self.feature_cols
            if col in self.encoders
        }
        target_dims = {
            col: len(self.encoders[col].classes_)
            for col in self.target_cols
            if col in self.encoders
        }

        # Architecture must match the saved checkpoint exactly.
        model = create_gnn_model(
            model_name=self.model_name,
            n_features=n_features,
            target_dims=target_dims,
            params=self.params)

        weights_path = get_settings().paths.models / f"pmodel_{self.model_name}.pth"
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Model weights not found: {weights_path}"
            )

        # map_location ensures CPU fallback when no GPU is available.
        state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(self.device)

        return model

    def _transform_scalar(self, col: str, val: Any) -> torch.Tensor:
        """Encode a single scalar value to a tensor."""
        enc = self.encoders[col]
        val_str = str(val)
        if isinstance(enc, MultiLabelBinarizer):
            raise ValueError(
                f"Column '{col}' is multi-label; use _transform_vectorized instead."
            )

        if val_str not in enc.classes_:
            logger.debug("Unknown value '%s' in column '%s'; falling back to UNKNOWN_TOKEN.", val, col)
            val_str = UNKNOWN_TOKEN

        encoded_arr = cast(np.ndarray, enc.transform([val_str]))
        idx = int(encoded_arr.item())

        return torch.tensor([idx], dtype=torch.long, device=self.device)

    def _transform_vectorized(self, col: str, series: pd.Series) -> torch.Tensor:
        """Vectorised encode of a full pandas Series using numpy."""
        enc = self.encoders[col]
        vals = series.astype(str).to_numpy()

        mask = ~np.isin(vals, enc.classes_)
        if mask.any():
            vals[mask] = UNKNOWN_TOKEN

        encoded = enc.transform(vals)
        return torch.tensor(encoded, dtype=torch.long)

    def predict_single(self, input_dict: dict[str, Any]) -> dict[str, Any] | None:
        """
        Run inference on a single sample dict.
        Example: {'drug': 'Aspirin', 'gene': 'CYP2D6', ...}
        """
        model_inputs = {}

        try:
            for col in self.feature_cols:
                val = input_dict.get(col) or input_dict.get(col.capitalize())

                if val is None:
                    raise ValueError(f"Required feature '{col}' is missing from the input dict.")

                model_inputs[col] = self._transform_scalar(col, val).unsqueeze(0)

            with torch.no_grad():
                outputs = self.model(model_inputs)

            return self._decode_outputs(outputs)[0]

        except Exception as e:
            logger.error("Prediction failed: %s", e)
            return None

    def predict_file(
        self, file_path: str | Path, batch_size: int = 1024
    ) -> list[dict[str, Any]]:
        """
        Batch inference from a CSV/TSV file.
        Uses vectorised preprocessing and mini-batches to keep memory bounded.
        """
        file_path = Path(file_path)
        sep = "\t" if file_path.suffix == ".tsv" else ","

        try:
            df = pd.read_csv(file_path, sep=sep, dtype=str)
            # Normalise column names to lower-case to match feature_cols.
            df.columns = df.columns.str.lower().str.strip()
        except Exception as e:
            logger.error("Failed to read file %s: %s", file_path, e)
            return []

        input_tensors = {}
        try:
            for col in self.feature_cols:
                if col not in df.columns:
                    # Back-compat column aliases for old CSV schemas.
                    aliases = {"genalle": "genotype", "drug": "drug_name"}
                    alias = aliases.get(col)
                    if alias and alias in df.columns:
                        input_tensors[col] = self._transform_vectorized(col, df[alias])
                    else:
                        raise ValueError(
                            f"Required column '{col}' not found in the CSV."
                        )
                else:
                    input_tensors[col] = self._transform_vectorized(col, df[col])
        except Exception as e:
            logger.error("Pre-processing failed: %s", e)
            return []

        num_samples = len(df)
        if num_samples == 0:
            return []

        all_outputs_list = {t: [] for t in self.target_cols}

        with torch.no_grad():
            for i in range(0, num_samples, batch_size):
                end = min(i + batch_size, num_samples)

                batch_inputs = {
                    col: tensor[i:end].to(self.device)
                    for col, tensor in input_tensors.items()
                }

                batch_preds = self.model(batch_inputs)

                # Keep logits on CPU to avoid filling VRAM.
                for t in self.target_cols:
                    all_outputs_list[t].append(batch_preds[t].cpu())

        full_logits = {
            t: torch.cat(all_outputs_list[t], dim=0) for t in self.target_cols
        }

        results_list = self._decode_outputs(full_logits)

        # Merge with original rows for context.
        results_df = pd.DataFrame(results_list)
        final_df = pd.concat([df.reset_index(drop=True), results_df], axis=1)

        return final_df.to_dict(orient="records")

    def _decode_outputs(
        self, outputs: dict[str, torch.Tensor]
    ) -> list[dict[str, Any]]:
        """Convert raw logits to human-readable label strings, handling both single- and multi-label targets."""
        first_out = next(iter(outputs.values()))
        batch_size = first_out.size(0)

        decoded_results = {col: [] for col in self.target_cols}

        for col in self.target_cols:
            enc = self.encoders[col]
            logits = outputs[col]  # [Batch, Classes]

            if col in get_settings().multi_label_set:
                # Multi-label
                probs = torch.sigmoid(logits)
                preds_bin = (probs > 0.5).int().numpy()

                labels_tuples = enc.inverse_transform(preds_bin)
                decoded_results[col] = [list(labels) for labels in labels_tuples]

            else:
                # Single-label
                preds_idx = torch.argmax(logits, dim=1).numpy()
                labels = enc.inverse_transform(preds_idx)

                # Strip internal UNKNOWN_TOKEN before returning to the caller.
                clean_labels = [
                    label if label != UNKNOWN_TOKEN else "Unknown"
                    for label in labels
                ]
                decoded_results[col] = clean_labels

        # Transpose from Dict[List] → List[Dict].
        # From: {'target1': [a, b], 'target2': [c, d]}
        # To:   [{'target1': a, 'target2': c}, {'target1': b, 'target2': d}]
        return [
            {col: decoded_results[col][i] for col in self.target_cols}
            for i in range(batch_size)
        ]
