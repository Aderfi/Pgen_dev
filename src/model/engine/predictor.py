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
from src.model.architectures.config import AxisSpec
from src.model.architectures.heads.label_table import CompositionalLabelTable
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

        enc_path = get_settings().paths.encoders / f"encoders_{model_name}.pkl"
        bundle = self._load_training_artifacts(enc_path)

        encoders = bundle["encoders"]
        for col in self.target_cols:
            if col not in encoders:
                raise EncoderError(
                    f"Encoder for target {col!r} missing from {enc_path}"
                )
        self.encoders = encoders
        self._saved_drug_dim = bundle.get("drug_dim")
        self._saved_geno_dim = bundle.get("geno_dim")
        self._axis_specs: dict[str, AxisSpec] | None = None
        if bundle.get("schema_version") == 2 and bundle.get("axis_specs"):
            self._axis_specs = {
                name: AxisSpec(**spec) for name, spec in bundle["axis_specs"].items()
            }

        self.cleaner = PharmacogenomicCleaner(multi_label_cols=self.multi_label_cols)
        self._resolver: GenotypeResolver | None = None
        self.collater = DoubleTowerCollater(inference_mode=True)

        self.model = self._load_model()
        self.model.eval()

        self._composable_axes: list[str] = self.model.axis_heads.single_label_axes()
        self.label_table = self._build_label_table(bundle)

    def _build_label_table(
        self, bundle: dict[str, Any]
    ) -> CompositionalLabelTable | None:
        """Rebuild the compositional label table persisted at training time.

        Returns None when the model has no compose head, or the bundle
        carries no (or an empty) label table — decoding then falls back to
        the plain per-axis behavior.
        """
        label_table_data = bundle.get("label_table") or {}
        tuples = label_table_data.get("tuples")
        if not tuples or self.model.compose is None:
            return None

        labels = label_table_data.get("labels", [])
        table = CompositionalLabelTable([tuple(t) for t in tuples], labels)
        table.build(self.model.compose, self.model.axis_heads.axis_embeddings)
        return table

    # ----- artifact loading ------------------------------------------------ #

    @staticmethod
    def _load_training_artifacts(path: Path) -> dict[str, Any]:
        """Load and normalize the encoder bundle persisted by the training pipeline.

        Supports three on-disk formats, always returned as a single dict with
        at least ``encoders`` and ``schema_version`` keys:

        1. **Schema v2** — the full bundle written by
           ``_persist_training_artifacts`` in ``src/pipeline.py``: ``encoders``,
           ``drug_dim``, ``geno_dim``, ``edge_dims``, ``aux_dims``,
           ``axis_specs``, ``label_table``, ``switches``, ``schema_version=2``.
           Returned as-is.
        2. **Schema v1** — the older bundle with just ``encoders``,
           ``drug_dim``, ``geno_dim``, ``schema_version=1``. Returned as-is
           (``schema_version`` normalized to ``1`` if missing/other).
        3. **Legacy plain dict** — ``{target_col: encoder}`` written by even
           older pipelines, with no ``schema_version`` at all. Wrapped as
           ``{"encoders": payload, "schema_version": 1}`` with a warning;
           dims are then inferred from ``cfg.extras`` by the caller, which is
           fragile but preserves backward compatibility.
        """
        if not path.exists():
            raise FileNotFoundError(f"Encoders file not found: {path}")

        # joblib.load unpickles a local artifact written by this project's own
        # `pipeline._persist_training_artifacts` — not attacker-controlled
        # input, so the arbitrary-code-execution risk of pickle does not
        # apply here (same trust boundary as the rest of the checkpoint I/O).
        payload = joblib.load(path)
        if not isinstance(payload, dict):
            raise EncoderError(
                f"Encoder artifact at {path} is not a dict "
                f"(got {type(payload).__name__})"
            )

        if "encoders" in payload and isinstance(payload["encoders"], dict):
            bundle = dict(payload)
            if bundle.get("schema_version") not in (1, 2):
                bundle["schema_version"] = 1
            return bundle

        logger.warning(
            "Loading legacy encoders pickle at %s — drug_dim / geno_dim will "
            "fall back to cfg.extras defaults; retrain to refresh the bundle.",
            path,
        )
        return {"encoders": payload, "schema_version": 1}

    def _infer_axes(self):
        """Rebuild the per-axis specs from the loaded encoders.

        Fallback used when the bundle is v1/legacy and carries no persisted
        ``axis_specs``. There are no training labels available at inference
        time, so ``train_targets`` is passed empty — ``infer_axis_specs``
        tolerates this by leaving ``pos_weight`` at ``None`` for binary axes,
        which is fine here since the specs are only used to reconstruct the
        model architecture, not to compute a loss.
        """
        return infer_axis_specs(
            self.encoders,
            {},
            set(self.multi_label_cols),
            get_axes_config(),
        )

    def _resolve_axes(self) -> dict[str, AxisSpec]:
        """Prefer the axis specs persisted at training time (schema v2)."""
        if self._axis_specs is not None:
            return self._axis_specs
        return self._infer_axes()

    def _resolve_tower_dim(self, saved: int | None, fallback_key: str) -> int:
        """Prefer the dim persisted at training time; fall back to cfg.extras."""
        if saved is not None:
            return saved
        return self.dims[fallback_key]["features"]

    def _load_model(self):
        """Instantiate the architecture and load the best checkpoint."""
        axes = self._resolve_axes()
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
        z_chunks: list[torch.Tensor] = []
        for batch in loader:
            drug_batch = batch["drug_batch"].to(self.device)
            geno_batch = batch["geno_batch"].to(self.device)
            outputs = self.model(drug_batch, geno_batch)
            for t in self.target_cols:
                per_target[t].append(outputs[t].cpu())
            if "_z" in outputs:
                z_chunks.append(outputs["_z"].cpu())

        result = {t: torch.cat(per_target[t], dim=0) for t in self.target_cols}
        if z_chunks:
            result["_z"] = torch.cat(z_chunks, dim=0)
        return result

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

        results = [
            {col: decoded[col][i] for col in self.target_cols}
            for i in range(batch_size)
        ]

        z = logits.get("_z")
        if self.label_table is not None and z is not None:
            self._attach_composed_labels(results, logits, z)

        return results

    def _attach_composed_labels(
        self,
        results: list[dict[str, Any]],
        logits: dict[str, torch.Tensor],
        z: torch.Tensor,
    ) -> None:
        """Augment each result dict with the compositional-table decode.

        Adds ``composed_label`` (top-1), ``composed_topk`` (label, score
        pairs), and ``composed_agreement`` (whether the nearest table row's
        tuple matches the per-axis argmax) — additive, never replaces the
        existing per-axis decode.
        """
        assert self.label_table is not None
        top_k = self.label_table.decode(z, top_k=3)
        argmax_tuple = torch.stack(
            [logits[axis].argmax(dim=-1) for axis in self._composable_axes], dim=1
        )
        agreement = self.label_table.agreement(z, argmax_tuple)

        for i, result in enumerate(results):
            topk_i = top_k[i]
            result["composed_label"] = topk_i[0][0] if topk_i else None
            result["composed_topk"] = [(label, float(score)) for label, score in topk_i]
            result["composed_agreement"] = bool(agreement[i].item())

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
