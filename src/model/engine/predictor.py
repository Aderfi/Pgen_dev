"""Inference engine for the Two-Tower GNN.

Composes the same primitives the trainer uses (``DoubleTowerDataset`` for
graph lookup, ``DoubleTowerCollater`` for batching, ``PharmacogenomicCleaner``
for input normalization) so the inference path can never drift from the
training path. The saved checkpoint provides the model weights; the saved
target-encoder bundle provides the label spaces.

Public API (relied on by ``src/api/routers/predict.py`` and ``src/cli``):

    predictor = PGenPredictor(model_name)
    predictor.predict_single({"drugs_chembl": "CHEMBL1771", "genotype": "CYP2C19*2"})
    predictor.predict_file("data/patients.csv")
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl
import torch
from torch.utils.data import DataLoader

from src.config import get_axes_config, get_model_config, get_settings
from src.core import EncoderError, EvaluationError, ModelError, UnknownDrugError
from src.data.cache import GraphCache, GraphDims
from src.data.cleaning import PharmacogenomicCleaner
from src.data.collator import DoubleTowerCollater
from src.data.datasets import DoubleTowerDataset
from src.data.encoders import UNKNOWN_CATEGORY_LABEL, TargetEncoder
from src.data.genotype import GenotypeGraphBuilder, load_hgvs_catalog
from src.data.loaders import TabularLoader
from src.model.architectures.assembly import infer_axis_specs
from src.model.architectures.config import AxisSpec
from src.model.architectures.label_table import CompositionalLabelTable
from src.model.artifacts import load_training_artifacts
from src.model.checkpoint import CheckpointManager
from src.model.engine.base import (
    GENE_COLUMN,
    build_gnn_model,
    resolve_device,
    tower_dims,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence, Set
    from typing import Protocol

    class _ConformerSource(Protocol):
        def entity_members(self, key: str) -> list[str]: ...
        def n_conformers(self, drug_id: str) -> int: ...

    import numpy as np


logger = logging.getLogger(__name__)


# "CYP2D6*1" or "CYP2D6 *1" → (gene, allele).
_STAR_ALLELE_RE = re.compile(r"^([A-Z0-9]+)\s*(\*[A-Za-z0-9]+)$")


def conformer_passes(
    cache: _ConformerSource, rows: Iterable[Sequence[str]], k: int
) -> int:
    """Forward passes to average: ``min(k, most conformers of any molecule)``.

    ``rows`` are the entity keys of each row; at least one pass (2D-only or
    unknown drugs).
    """
    most = max(
        (
            cache.n_conformers(m)
            for keys in rows
            for key in keys
            for m in cache.entity_members(key)
        ),
        default=0,
    )
    return max(1, min(k, most))


def average_predictions(
    passes: list[dict[str, torch.Tensor]], *, multi_label_cols: Set[str]
) -> dict[str, torch.Tensor]:
    """Average predictions (probabilities) of several passes, back as logits.

    Multi-label heads average sigmoids and return their logit; the others
    average softmaxes and return the log (valid logits); ``_z`` is averaged.
    """
    if len(passes) == 1:
        return passes[0]
    out: dict[str, torch.Tensor] = {}
    for key in passes[0]:
        stack = torch.stack([p[key] for p in passes])
        if key.startswith("_"):
            out[key] = stack.mean(0)
        elif key in multi_label_cols:
            out[key] = torch.logit(torch.sigmoid(stack).mean(0), eps=1e-6)
        else:
            out[key] = torch.log(torch.softmax(stack, dim=-1).mean(0).clamp_min(1e-12))
    return out


class PGenPredictor:
    """Inference engine for a trained ``PharmagenTwoTower`` model.

    Loads the model, target encoders, and graph cache once per model name;
    each call reuses them. Safe to register in the FastAPI dependency
    container — there is no per-request mutable state.
    """

    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        *,
        run_name: str | None = None,
    ) -> None:
        """``model_name`` selects the models.toml section; ``run_name`` (default:
        ``model_name``) the checkpoint + artifact bundle, e.g. a scenario fold."""
        self.model_name = model_name
        self.run_name = run_name or model_name
        self.device = resolve_device(device)

        logger.info(
            "Initialising PGenPredictor for '%s' on %s", model_name, self.device
        )

        self.cfg = get_model_config(model_name)
        self.feature_cols = list(self.cfg.features)
        self.target_cols = list(self.cfg.targets)
        self.params = self.cfg.training_params()
        self.multi_label_cols = get_settings().multi_label_set

        enc_path = get_settings().paths.encoders / f"encoders_{self.run_name}.pkl"
        bundle = self._load_training_artifacts(enc_path)
        # v3 bundles carry the graph widths the model was trained with.
        self.dims = (
            GraphDims(**bundle["dims"]) if "dims" in bundle else tower_dims(self.cfg)
        )

        encoders = bundle["encoders"]
        for col in self.target_cols:
            if col not in encoders:
                raise EncoderError(
                    f"Encoder for target {col!r} missing from {enc_path}"
                )
        self.encoders = encoders
        self._saved_drug_dim = bundle.get("drug_dim")
        self._saved_geno_dim = bundle.get("geno_dim")
        # Reconstruct the exact architecture the checkpoint was trained under:
        # the persisted switches decide whether poly / cross-attention submodules
        # exist, so a strict state_dict load matches.
        self._switches = bundle.get("switches")
        # Scenario-fold runs record the split they were trained on (schema v5+).
        self.split_info: dict[str, Any] | None = bundle.get("split")
        # Gene vocabulary of the HGVS genotype tower (schema v4+).
        self.gene_vocab: dict[str, int] = bundle.get("gene_vocab") or {}
        self._axis_specs: dict[str, AxisSpec] | None = None
        if bundle.get("axis_specs"):
            self._axis_specs = {
                name: AxisSpec(**spec) for name, spec in bundle["axis_specs"].items()
            }

        self.inference_conformers = get_settings().inference_conformers
        self._drug_cache: GraphCache | None = None
        self.cleaner = PharmacogenomicCleaner(multi_label_cols=self.multi_label_cols)
        self._genotype_builder: GenotypeGraphBuilder | None = None
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
        """See :func:`src.model.artifacts.load_training_artifacts`."""
        return load_training_artifacts(path)

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
        if fallback_key == "drugs":
            return self.dims.drug_features
        return self.dims.geno_features

    def _load_model(self):
        """Instantiate the architecture and load the best checkpoint."""
        axes = self._resolve_axes()
        drug_dim = self._resolve_tower_dim(self._saved_drug_dim, "drugs")
        geno_dim = self._resolve_tower_dim(self._saved_geno_dim, "geno")

        model = build_gnn_model(
            dims=self.dims,
            drug_dim=drug_dim,
            geno_dim=geno_dim,
            axes=axes,
            params=self.params,
            device=self.device,
            switches=self._switches,
            gene_vocab=self.gene_vocab or None,
        )

        manager = CheckpointManager(model_name=self.run_name)
        if manager.get_best_checkpoint_path() is None:
            raise FileNotFoundError(
                f"No best checkpoint for model '{self.run_name}' under {manager.save_dir}"
            )

        try:
            checkpoint = manager.load_checkpoint(load_best=True)
            model.load_state_dict(checkpoint["model_state_dict"])
        except Exception as e:
            raise ModelError(
                f"Failed to load weights for '{self.run_name}': {e}"
            ) from e

        return model

    @property
    def genotype_builder(self) -> GenotypeGraphBuilder:
        """Lazily load the HGVS catalog on first prediction.

        Kept out of ``__init__`` so constructing a predictor (e.g. for a health
        check or artifact inspection) doesn't require the catalog on disk.
        """
        if self._genotype_builder is None:
            self._genotype_builder = GenotypeGraphBuilder(
                load_hgvs_catalog(), self.gene_vocab
            )
        return self._genotype_builder

    # ----- input normalization -------------------------------------------- #

    def _input_to_dataframe(self, sample: Mapping[str, Any]) -> pl.DataFrame:
        """Coerce a free-form input dict into a 1-row training-shaped frame.

        ``hgvs`` (a string or a list of HGVS expressions) is the primary way to
        state a genotype; ``genotype`` may also carry rsIDs or star alleles.
        """
        normalized = {str(k).lower().strip(): v for k, v in sample.items()}

        drug = normalized.get("drugs_chembl") or normalized.get("drug")
        if drug is None:
            raise ValueError("missing required field 'drugs_chembl' (or 'drug')")

        gene = normalized.get("gene")
        genotype = normalized.get("genotype", "") or ""
        alleles = normalized.get("alleles", "") or ""
        hgvs = normalized.get("hgvs") or []
        if isinstance(hgvs, str):
            hgvs = [hgvs]
        if hgvs:
            genotype = "|".join([*map(str, hgvs), *([genotype] if genotype else [])])

        # API-style single inputs collapse gene + allele into a single label
        # like "CYP2D6*1"; split it back out into (gene, genotype).
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

        # The cleaner drops rows with an empty genotype; a gene with no stated
        # variants is the reference genotype.
        if not genotype:
            genotype = alleles or "REF_SEQ"

        return pl.DataFrame(
            {
                "drugs_chembl": [str(drug).strip().upper()],
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
            genotype_builder=self.genotype_builder,
            preload_ram=False,
            dims=self.dims,
            inference_mode=True,
            drug_cache=self.drug_cache,
        )

    @property
    def drug_cache(self) -> GraphCache:
        """One library cache for every request (conformer views share it)."""
        if self._drug_cache is None:
            self._drug_cache = GraphCache.from_settings(
                dims=self.dims, inference_mode=True
            )
        return self._drug_cache

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

    def _forward_dataset(
        self, dataset: DoubleTowerDataset, batch_size: int
    ) -> dict[str, torch.Tensor]:
        """Predictions averaged over the lowest-energy conformers.

        Pass ``i`` reads conformer ``i`` of every molecule (clamped to the
        molecule's own count); the number of passes is
        :func:`conformer_passes` over the dataset's drugs.
        """
        base = dataset.cache
        n = conformer_passes(base, dataset.lookup_entities, self.inference_conformers)
        passes: list[dict[str, torch.Tensor]] = []
        try:
            for i in range(n):
                dataset.cache = base.with_conformer(i)
                loader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    collate_fn=self.collater,
                )
                passes.append(self._forward_loader(loader))
        finally:
            dataset.cache = base
        if n > 1:
            logger.debug("Averaged predictions over %d conformers", n)
        return average_predictions(passes, multi_label_cols=self.multi_label_cols)

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
        into a valid graph pair (logged at error level). Raises
        :class:`~src.core.UnknownDrugError` when no named drug is readable —
        a prediction from the placeholder graph would be meaningless.
        """
        try:
            df = self._input_to_dataframe(input_dict)
            dataset = self._build_inference_dataset(df)
            if dataset.rows_without_drug:
                keys = "|".join(dataset.lookup_entities[0]) or "(none)"
                msg = (
                    f"no readable drug in {keys}: not in the library, or a class "
                    "without built members"
                )
                raise UnknownDrugError(msg)
            logits = self._forward_dataset(dataset, batch_size=1)
            results = self._decode_logits(logits)
            return results[0] if results else None
        except UnknownDrugError:
            raise
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

        logits = self._forward_dataset(dataset, batch_size)
        return self._decode_logits(logits)

    # ----- evaluation API -------------------------------------------------- #

    def predict_logits(
        self, df: pl.DataFrame, batch_size: int = 256
    ) -> dict[str, torch.Tensor]:
        """Raw logits per target for already-clean rows, in ``df``'s order."""
        dataset = self._build_inference_dataset(df)
        if len(dataset) != len(df):
            raise EvaluationError(
                f"cleaning dropped {len(df) - len(dataset)} of {len(df)} rows; "
                "evaluation frames must already be clean"
            )
        logits = self._forward_dataset(dataset, batch_size)
        return {t: logits[t] for t in self.target_cols}

    def encode_targets(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        """Targets of ``df`` through the run's fitted encoders."""
        encoder = TargetEncoder(
            self.target_cols, self.multi_label_cols, encoders=self.encoders
        )
        return {k: v.numpy() for k, v in encoder.fit_transform(df).items()}


__all__ = ["PGenPredictor", "average_predictions", "conformer_passes"]
