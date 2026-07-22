"""Inference engine for the Two-Tower GNN.

Composes the same primitives the trainer uses (``DoubleTowerDataset`` for
graph lookup, ``DoubleTowerCollater`` for batching, ``PharmacogenomicCleaner``
for input normalization) so the inference path can never drift from the
training path. The saved checkpoint provides the model weights; the saved
target-encoder bundle provides the label spaces.

Public API (relied on by ``src/api/routers/predict.py``,
``src/interface/cli.py``, and ``main.py``):

    predictor = PGenPredictor(model_name)
    predictor.predict_single({"drugs_cid": "12345", "genotype": "CYP2D6*1"})
    predictor.predict_file("data/patients.csv")
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import joblib
import polars as pl
import torch
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from torch.utils.data import DataLoader

from src.config import get_axes_config, get_model_config, get_settings
from src.core import EncoderError, ModelError
from src.data.cleaning import PharmacogenomicCleaner
from src.data.collator import DoubleTowerCollater
from src.data.datasets import DoubleTowerDataset
from src.data.encoders import UNKNOWN_CATEGORY_LABEL
from src.data.library.geno_store import GenoLibrary
from src.data.loaders import TabularLoader
from src.model.architectures.assembly import infer_axis_specs
from src.model.checkpoint import CheckpointManager
from src.model.engine.base import (
    GENE_COLUMN,
    GENO_LIBRARY_FILE,
    build_gnn_model,
    extract_tower_dims,
    resolve_device,
)

if TYPE_CHECKING:
    from src.data.library.genotype_resolver import GenotypeResolver

logger = logging.getLogger(__name__)


# "CYP2D6*1" or "CYP2D6 *1" → (gene, allele).
_STAR_ALLELE_RE = re.compile(r"^([A-Z0-9]+)\s*(\*[A-Za-z0-9]+)$")


class PGenPredictor:
    """Inference engine for a trained ``PharmagenTwoTower`` model.

    Loads the model, target encoders, and graph cache once per model name;
    each call reuses them. Safe to register in the FastAPI dependency
    container — there is no per-request mutable state.
    """

    def __init__(self, model_name: str, device: str | None = None) -> None:
        self.model_name = model_name
        self.device = resolve_device(device)

        logger.info(
            "Initialising PGenPredictor for '%s' on %s", model_name, self.device
        )

        self.cfg = get_model_config(model_name)
        self.feature_cols = list(self.cfg.features)
        self.target_cols = list(self.cfg.targets)
        self.params = self.cfg.params
        self.multi_label_cols = get_settings().multi_label_set

        self.dims = extract_tower_dims(self.cfg)
        self.encoders, self._saved_drug_dim, self._saved_geno_dim = (
            self._load_training_artifacts()
        )
        self.cleaner = PharmacogenomicCleaner(multi_label_cols=self.multi_label_cols)
        self._resolver: GenotypeResolver | None = None
        self.collater = DoubleTowerCollater(inference_mode=True)

        self.model = self._load_model()
        self.model.eval()

    # ----- artifact loading ------------------------------------------------ #

    def _load_training_artifacts(
        self,
    ) -> tuple[dict[str, LabelEncoder | MultiLabelBinarizer], int | None, int | None]:
        """Load the encoder bundle persisted by the training pipeline.

        Supports two on-disk formats:

        1. **Current bundle** — a dict with keys ``encoders``, ``drug_dim``,
           ``geno_dim``, ``schema_version`` (saved by
           ``_persist_training_artifacts`` in ``src/pipeline.py``).
        2. **Legacy plain dict** — ``{target_col: encoder}`` written by older
           pipelines. The dims are then inferred from ``cfg.extras``, which
           is fragile but preserves backward compatibility.
        """
        enc_path = get_settings().paths.encoders / f"encoders_{self.model_name}.pkl"
        if not enc_path.exists():
            raise FileNotFoundError(f"Encoders file not found: {enc_path}")

        payload = joblib.load(enc_path)
        if not isinstance(payload, dict):
            raise EncoderError(
                f"Encoder artifact at {enc_path} is not a dict "
                f"(got {type(payload).__name__})"
            )

        if "encoders" in payload and isinstance(payload["encoders"], dict):
            encoders = payload["encoders"]
            drug_dim = payload.get("drug_dim")
            geno_dim = payload.get("geno_dim")
        else:
            logger.warning(
                "Loading legacy encoders pickle at %s — drug_dim / geno_dim will "
                "fall back to cfg.extras defaults; retrain to refresh the bundle.",
                enc_path,
            )
            encoders = payload
            drug_dim = None
            geno_dim = None

        for col in self.target_cols:
            if col not in encoders:
                raise EncoderError(
                    f"Encoder for target {col!r} missing from {enc_path}"
                )

        return encoders, drug_dim, geno_dim

    def _infer_axes(self):
        """Rebuild the per-axis specs from the loaded encoders.

        There are no training labels available at inference time, so
        ``train_targets`` is passed empty — ``infer_axis_specs`` tolerates
        this by leaving ``pos_weight`` at ``None`` for binary axes, which is
        fine here since the specs are only used to reconstruct the model
        architecture, not to compute a loss.
        """
        return infer_axis_specs(
            self.encoders,
            {},
            set(self.multi_label_cols),
            get_axes_config(),
        )

    def _resolve_tower_dim(self, saved: int | None, fallback_key: str) -> int:
        """Prefer the dim persisted at training time; fall back to cfg.extras."""
        if saved is not None:
            return saved
        return self.dims[fallback_key]["features"]

    def _load_model(self):
        """Instantiate the architecture and load the best checkpoint."""
        axes = self._infer_axes()
        drug_dim = self._resolve_tower_dim(self._saved_drug_dim, "drugs")
        geno_dim = self._resolve_tower_dim(self._saved_geno_dim, "geno")

        model = build_gnn_model(
            model_name=self.model_name,
            dims=self.dims,
            drug_dim=drug_dim,
            geno_dim=geno_dim,
            axes=axes,
            params=self.params,
            device=self.device,
        )

        manager = CheckpointManager(model_name=self.model_name)
        if manager.get_best_checkpoint_path() is None:
            raise FileNotFoundError(
                f"No best checkpoint for model '{self.model_name}' under {manager.save_dir}"
            )

        try:
            checkpoint = manager.load_checkpoint(load_best=True)
            model.load_state_dict(checkpoint["model_state_dict"])
        except Exception as e:
            raise ModelError(
                f"Failed to load weights for '{self.model_name}': {e}"
            ) from e

        return model

    @property
    def resolver(self) -> GenotypeResolver:
        """Lazily load the genotype library + resolver on first prediction.

        Kept out of ``__init__`` so constructing a predictor (e.g. for a health
        check or artifact inspection) doesn't require the on-disk gene library.
        """
        if self._resolver is None:
            self._resolver = GenoLibrary.load(
                get_settings().paths.library / GENO_LIBRARY_FILE
            ).resolver()
        return self._resolver

    # ----- input normalization -------------------------------------------- #

    def _input_to_dataframe(self, sample: Mapping[str, Any]) -> pl.DataFrame:
        """Coerce a free-form input dict into a 1-row training-shaped frame."""
        normalized = {str(k).lower().strip(): v for k, v in sample.items()}

        drug = normalized.get("drugs_cid") or normalized.get("drug")
        if drug is None:
            raise ValueError("missing required field 'drugs_cid' (or 'drug')")

        gene = normalized.get("gene")
        genotype = normalized.get("genotype", "")
        alleles = normalized.get("alleles", "")

        # API-style single inputs collapse gene + allele into a single label
        # like "CYP2D6*1"; split it back out so the resolver keys (gene, genotype).
        if not gene and isinstance(genotype, str):
            match = _STAR_ALLELE_RE.match(genotype.strip())
            if match:
                gene = match.group(1)
                alleles = match.group(2) if not alleles else alleles

        if not gene:
            raise ValueError(
                "missing 'gene' — pass it explicitly or use the 'GENE*ALLELE' "
                "shorthand in 'genotype'"
            )

        # Ensure the cleaner's drop filter doesn't remove our single row: it
        # requires non-empty gene + genotype strings.
        if not genotype:
            genotype = alleles or gene

        return pl.DataFrame(
            {
                "drugs_cid": [str(drug)],
                "gene": [str(gene)],
                "genotype": [str(genotype)],
                "alleles": [str(alleles)],
            }
        )

    def _build_inference_dataset(self, df: pl.DataFrame) -> DoubleTowerDataset:
        cleaned = self.cleaner.clean(df)
        if len(cleaned) == 0:
            raise ValueError("input produced no usable rows after cleaning")
        return DoubleTowerDataset(
            df=cleaned,
            drug_col=self.feature_cols[0],
            geno_col=self.feature_cols[1],
            target_cols=[],  # no target encoding needed at inference
            multilabel_cols=set(),
            gene_col=GENE_COLUMN,
            genotype_resolver=self.resolver,
            preload_ram=False,
            input_dimensions=self.dims,
            inference_mode=True,
        )

    # ----- forward + decode ----------------------------------------------- #

    @torch.no_grad()
    def _forward_loader(
        self,
        loader: DataLoader,
    ) -> dict[str, torch.Tensor]:
        per_target: dict[str, list[torch.Tensor]] = {t: [] for t in self.target_cols}
        for batch in loader:
            drug_batch = batch["drug_batch"].to(self.device)
            geno_batch = batch["geno_batch"].to(self.device)
            outputs = self.model(drug_batch, geno_batch)
            for t in self.target_cols:
                per_target[t].append(outputs[t].cpu())
        return {t: torch.cat(per_target[t], dim=0) for t in self.target_cols}

    def _decode_logits(self, logits: dict[str, torch.Tensor]) -> list[dict[str, Any]]:
        if not logits:
            return []
        batch_size = next(iter(logits.values())).size(0)
        decoded: dict[str, list[Any]] = {col: [] for col in self.target_cols}

        for col in self.target_cols:
            enc = self.encoders[col]
            tensor = logits[col]

            if col in self.multi_label_cols:
                probs = torch.sigmoid(tensor)
                preds_bin = (probs > 0.5).int().numpy()
                label_tuples = enc.inverse_transform(preds_bin)
                decoded[col] = [list(labels) for labels in label_tuples]
            else:
                idx = torch.argmax(tensor, dim=1).numpy()
                labels = enc.inverse_transform(idx)
                decoded[col] = [
                    label if label != UNKNOWN_CATEGORY_LABEL else "Unknown"
                    for label in labels
                ]

        return [
            {col: decoded[col][i] for col in self.target_cols}
            for i in range(batch_size)
        ]

    # ----- public API ------------------------------------------------------ #

    def predict_single(self, input_dict: Mapping[str, Any]) -> dict[str, Any] | None:
        """Run inference on a single (drug, allele) pair.

        Returns the decoded targets, or None if the input could not be turned
        into a valid graph pair (logged at error level).
        """
        try:
            df = self._input_to_dataframe(input_dict)
            dataset = self._build_inference_dataset(df)
            loader = DataLoader(
                dataset, batch_size=1, shuffle=False, collate_fn=self.collater
            )
            logits = self._forward_loader(loader)
            results = self._decode_logits(logits)
            return results[0] if results else None
        except Exception as e:
            logger.error("Single prediction failed: %s", e)
            return None

    def predict_file(
        self,
        file_path: str | Path,
        batch_size: int = 256,
    ) -> list[dict[str, Any]]:
        """Run batched inference over a CSV/TSV with the training schema."""
        file_path = Path(file_path)
        try:
            raw_df = TabularLoader.load(file_path)
        except Exception as e:
            logger.error("Failed to read input file %s: %s", file_path, e)
            return []

        try:
            dataset = self._build_inference_dataset(raw_df)
        except Exception as e:
            logger.error("Pre-processing failed: %s", e)
            return []

        if len(dataset) == 0:
            return []

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.collater,
        )
        logits = self._forward_loader(loader)
        return self._decode_logits(logits)


__all__ = ["PGenPredictor"]
