"""Polypharmacy ablation scaffold: construct + forward both config arms.

**Scope (read before relying on this for a real ablation).** This script is a
*scaffold*, not an end-to-end trainer. It builds two ``PharmagenConfig`` arms
that differ only in ``use_polypharmacy``/``use_cross_attention``
(``poly=off`` vs ``poly=on``), instantiates a ``PharmagenTwoTower`` for each,
logs ``count_parameters()``, and runs each model once on a small hand-built
synthetic batch to confirm both configurations still construct and forward
cleanly after the Task D4 focal-anchored readout change.

It deliberately does **not** train on ``data/processed/train_data.tsv`` (or
any real data) and does **not** report val AUPRC/accuracy. Two things block
that:

1. **No real DDI export.** The polypharmacy (drug-drug interaction) edge
   list this model expects (``ddi_edge_index`` over molecule-global
   indices, one graph per patient) does not exist as a curated artifact yet.
2. **The dataset-side batching is deferred.** ``DoubleTowerDataset.
   _build_poly_drug_data`` packs molecule descriptors into ``x`` today; the
   model expects atom-level ``x`` + a ``global_feats`` attribute. The full
   atom -> molecule -> patient two-level batching (and the focal-flag
   plumbing that feeds ``is_focal``) is a follow-up task, not part of D4.

``--input``/``--epochs`` are accepted (and validated) so the CLI shape
matches the eventual real ablation run, but they are currently no-ops beyond
a warning: this scaffold always runs the synthetic-batch check regardless of
what is passed. Wiring them up to a real training loop is future work, once
(1) and (2) above are resolved.

Usage::

    uv run python -m scripts.ablate_polypharmacy
    uv run python -m scripts.ablate_polypharmacy --input data/processed/train_data.tsv --epochs 3
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running this script directly (`python scripts/ablate_polypharmacy.py`)
# without the repo root on sys.path, matching the pattern in main.py.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.append(str(_PROJECT_ROOT))

import torch  # noqa: E402
from torch_geometric.data import Batch, Data  # noqa: E402

from src.core import setup_logging  # noqa: E402
from src.model.architectures import (  # noqa: E402
    PharmagenConfig,
    PharmagenTwoTower,
    TaskSpec,
)

logger = logging.getLogger("pharmagen.ablate_polypharmacy")

DRUG_IN_FEATURES = 6
DRUG_EDGE_DIM = 4
GENO_IN_FEATURES = 5


def _mol_graph() -> Data:
    """A tiny 3-atom molecule graph. GINEConv requires edge_attr."""
    return Data(
        x=torch.randn(3, DRUG_IN_FEATURES),
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        edge_attr=torch.randn(2, DRUG_EDGE_DIM),
    )


def _geno_graph() -> Data:
    return Data(
        x=torch.randn(3, GENO_IN_FEATURES),
        edge_index=torch.tensor([[0, 1], [1, 2]]),
    )


def build_synthetic_batch(use_polypharmacy: bool) -> tuple[Batch, Batch]:
    """Build a 2-patient synthetic (drug_batch, geno_batch) pair.

    When `use_polypharmacy`, each patient gets 2 molecules (a focal drug plus
    one co-medication neighbour) wired with `mol_to_patient`, `ddi_edge_index`
    and `is_focal`, exercising the Task D4 focal-anchored readout. Otherwise
    each patient gets a single molecule, matching the pre-D4 shape.
    """
    geno_batch = Batch.from_data_list([_geno_graph(), _geno_graph()])

    if not use_polypharmacy:
        drug_batch = Batch.from_data_list([_mol_graph(), _mol_graph()])
        return drug_batch, geno_batch

    mols = [_mol_graph() for _ in range(4)]
    drug_batch = Batch.from_data_list(mols)
    drug_batch.mol_to_patient = torch.tensor([0, 0, 1, 1])
    drug_batch.ddi_edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]])
    drug_batch.is_focal = torch.tensor([1, 0, 1, 0])
    return drug_batch, geno_batch


def build_config(use_polypharmacy: bool) -> PharmagenConfig:
    """Return a config arm: identical to its sibling except the poly switches."""
    return PharmagenConfig(
        drug_in_features=DRUG_IN_FEATURES,
        drug_edge_dim=DRUG_EDGE_DIM,
        drug_hidden_dim=32,
        ddi_edge_dim=None,
        geno_in_features=GENO_IN_FEATURES,
        geno_edge_dim=None,
        geno_hidden_dim=32,
        embedding_dim=32,
        num_layers=2,
        heads=2,
        dropout=0.1,
        drug_global_dim=0,
        drug_admet_dim=0,
        geno_global_dim=0,
        use_polypharmacy=use_polypharmacy,
        use_cross_attention=use_polypharmacy,
        axes={
            "pheno": TaskSpec(dim=3, kind="multiclass"),
            "outcome": TaskSpec(dim=1, kind="binary"),
        },
    )


def run_arm(label: str, use_polypharmacy: bool) -> dict[str, int]:
    """Build, count params, and forward-check one config arm. Returns the counts."""
    cfg = build_config(use_polypharmacy)
    model = PharmagenTwoTower(cfg)
    model.eval()

    counts = model.count_parameters()
    logger.info("[%s] parameter counts: %s", label, counts)

    drug_batch, geno_batch = build_synthetic_batch(use_polypharmacy)
    with torch.no_grad():
        outputs = model(drug_batch, geno_batch)
    logger.info("[%s] forward OK, output keys: %s", label, sorted(outputs))

    return counts


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scaffold: construct poly=off/on PharmagenTwoTower config arms, "
            "log parameter counts, and forward-check each on a synthetic "
            "batch. Does NOT train end-to-end on real data — see module "
            "docstring for what is missing (real DDI export + the deferred "
            "two-level dataset batching)."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help=(
            "Path to a training TSV (e.g. data/processed/train_data.tsv). "
            "Accepted for CLI-shape parity with the eventual real ablation "
            "run; this scaffold does not read it (no real DDI export yet)."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help=(
            "Epoch count for the eventual real training ablation. Accepted "
            "for CLI-shape parity; this scaffold does not train."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    setup_logging(name="ablate_polypharmacy", console_level=logging.INFO)
    args = parse_args(argv)

    if args.input is not None or args.epochs != 3:
        logger.warning(
            "--input/--epochs were provided but this scaffold does not run "
            "real training (see module docstring): input=%s, epochs=%d. "
            "Running the synthetic-batch construct+forward check only.",
            args.input,
            args.epochs,
        )

    logger.info("Running polypharmacy ablation scaffold (synthetic batches only)")
    off_counts = run_arm("poly=off", use_polypharmacy=False)
    on_counts = run_arm("poly=on", use_polypharmacy=True)

    logger.info(
        "Parameter delta (poly=on - poly=off): total=%d",
        on_counts["total"] - off_counts["total"],
    )
    logger.info(
        "Scaffold complete. Both config arms construct and forward "
        "successfully. Real val-metric comparison requires a curated DDI "
        "artifact and the deferred atom->molecule->patient dataset batching."
    )


if __name__ == "__main__":
    main()
