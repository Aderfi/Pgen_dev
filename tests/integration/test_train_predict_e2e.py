"""End-to-end: train on a tiny synthetic library, then predict with the result.

Exercises the real wiring (``prepare_training_data`` → model → artifact bundle
→ ``Trainer`` + checkpoints → ``PGenPredictor``) with every path redirected to
``tmp_path``. Graphs are random but correctly shaped, so this checks plumbing,
not learning.
"""

from __future__ import annotations

import polars as pl
import pytest
import torch
from torch_geometric.data import Data

from src.config import get_settings
from src.config import settings as settings_module
from src.data.cache import GraphDims
from src.data.genotype import HGVS_CATALOG_FILE, HgvsCatalog, VariantInfo

pytestmark = [pytest.mark.integration, pytest.mark.slow]

CIDS = ["101", "202", "303", "404"]
HGVS = "NC_000022.11:g.42128945C>T"
N_ROWS = 160


def _drug_graph(dims: GraphDims) -> Data:
    n = 5
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    g = Data(
        x=torch.randn(n, dims.drug_features),
        edge_index=edge_index,
        edge_attr=torch.randn(edge_index.size(1), dims.drug_edges),
    )
    g.global_feats = torch.randn(1, dims.drug_global)
    g.admet_feats = torch.randn(1, dims.drug_admet)
    return g


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Point every Settings path at ``tmp_path`` and build a tiny library."""
    real_build_paths = settings_module._build_paths

    def build_paths(cfg):
        paths = real_build_paths(cfg)
        return paths.model_copy(
            update={
                "library": tmp_path / "library",
                "models": tmp_path / "models",
                "encoders": tmp_path / "encoders",
                "logs": tmp_path / "logs",
                "reports": tmp_path / "reports",
                "data": tmp_path / "data",
            }
        )

    monkeypatch.setattr(settings_module, "_build_paths", build_paths)
    get_settings.cache_clear()
    library = get_settings().paths.library
    (library / "drugs").mkdir(parents=True)
    dims = GraphDims()
    for cid in CIDS:
        torch.save(_drug_graph(dims), library / "drugs" / f"{cid}_drug.pt")
    HgvsCatalog(
        variants={HGVS: VariantInfo("CYP2D6", ("missense_variant",), None)},
        labels={"rs1": (HGVS,)},
    ).save(library / HGVS_CATALOG_FILE)

    rows = pl.DataFrame(
        {
            "variant_annotation_id": list(range(N_ROWS)),
            "drugs_cid": [CIDS[i % len(CIDS)] for i in range(N_ROWS)],
            "drugs": ["x"] * N_ROWS,
            "genotype": [["rs1", HGVS, "REF_SEQ"][i % 3] for i in range(N_ROWS)],
            "gene": ["CYP2D6"] * N_ROWS,
            "alleles": [["CT", "TT", None][i % 3] for i in range(N_ROWS)],
            "phenotype_category": [
                ["Toxicity", "Efficacy", "Dosage|Other"][i % 3] for i in range(N_ROWS)
            ],
        }
    )
    yield tmp_path, rows
    get_settings.cache_clear()


def _train(source, monkeypatch, split=None):
    from src.model.engine import base
    from src.pipeline import train_pipeline

    # torch.compile needs a C++ toolchain; irrelevant to the wiring under test.
    real_trainer = base.Trainer

    def eager_trainer(*args, **kwargs):
        kwargs["compile_model"] = False
        return real_trainer(*args, **kwargs)

    monkeypatch.setattr(base, "Trainer", eager_trainer)
    return train_pipeline(
        "TwoTowerGAT", source, epochs=2, batch_size=32, patience=2, split=split
    )


def test_train_then_predict(sandbox, monkeypatch):
    tmp_path, rows = sandbox
    data = tmp_path / "train.tsv"
    rows.write_csv(data, separator="\t")

    best = _train(data, monkeypatch)
    assert best < float("inf")
    assert (tmp_path / "encoders" / "encoders_TwoTowerGAT.pkl").exists()
    ckpt = (
        tmp_path / "models" / "checkpoints" / "TwoTowerGAT" / "best_model.safetensors"
    )
    assert ckpt.exists()

    from src.model.engine.predictor import PGenPredictor

    predictor = PGenPredictor("TwoTowerGAT", device="cpu")
    queries = tmp_path / "queries.csv"
    rows.head(7).select("drugs_cid", "gene", "genotype", "alleles").write_csv(queries)
    results = predictor.predict_file(queries)
    assert len(results) == 7
    assert all("phenotype_category" in r for r in results)
    assert predictor.predict_single(
        {"drugs_cid": CIDS[0], "gene": "CYP2D6", "hgvs": [HGVS], "alleles": "CT"}
    )


def test_train_from_kg_snapshot(sandbox, monkeypatch):
    """The parquet written by ``python -m src.kg export`` is a valid source."""
    from src.kg.graph import TRAINING_ROW_SCHEMA
    from src.kg.training import to_training_frame

    tmp_path, rows = sandbox
    raw = pl.DataFrame(
        {
            "annotation_id": [str(i) for i in range(N_ROWS)],
            "source": ["var_drug_ann"] * N_ROWS,
            "pmid": ["1"] * N_ROWS,
            "drug_cids": [[c] for c in rows["drugs_cid"]],
            "drug_names": [["x"]] * N_ROWS,
            "genes": [["CYP2D6"]] * N_ROWS,
            "genotypes": [["rs1"]] * N_ROWS,
            "implicated_hgvs": [[HGVS] if i % 2 else [] for i in range(N_ROWS)],
            "alleles": [None] * N_ROWS,
            "comparison_alleles": [None] * N_ROWS,
            "phenotype_categories": [p.split("|") for p in rows["phenotype_category"]],
            "direction": ["increased"] * N_ROWS,
            "significance": ["yes"] * N_ROWS,
            "associated": [True] * N_ROWS,
        },
        schema=TRAINING_ROW_SCHEMA,
    )
    snapshot = tmp_path / "training_rows.parquet"
    to_training_frame(raw).write_parquet(snapshot)

    assert _train(snapshot, monkeypatch) < float("inf")


def test_split_then_train_a_fold(sandbox, monkeypatch):
    import joblib

    from src.config import get_model_config
    from src.evaluation.entities import ROW_ID, Scenario
    from src.evaluation.runs import FoldRef, create_splits, load_fold_frames
    from src.evaluation.splits import load_split

    tmp_path, rows = sandbox
    data = tmp_path / "train.tsv"
    rows.write_csv(data, separator="\t")
    cfg = get_model_config("TwoTowerGAT")
    create_splits(data, cfg, [Scenario.DRUG_COLD], k=2)
    _, manifest = load_split(Scenario.DRUG_COLD)

    ref = FoldRef(scenario=Scenario.DRUG_COLD, fold=0)
    assert _train(data, monkeypatch, split=ref) < float("inf")
    run = "TwoTowerGAT__drug_cold__fold0"
    assert (
        tmp_path / "models" / "checkpoints" / run / "best_model.safetensors"
    ).exists()
    bundle = joblib.load(tmp_path / "encoders" / f"encoders_{run}.pkl")
    assert bundle["split"]["split_sha256"] == manifest.split_sha256

    train, val, test, _ = load_fold_frames(data, cfg, ref)
    test_ids = set(test[ROW_ID].to_list())
    assert test_ids
    assert not test_ids & (set(train[ROW_ID].to_list()) | set(val[ROW_ID].to_list()))
    assert not set(train["drugs_cid"].cast(str)) & set(test["drugs_cid"].cast(str))


def test_train_split_rejects_changed_data(sandbox, monkeypatch):
    from src.config import get_model_config
    from src.core import EvaluationError
    from src.evaluation.entities import Scenario
    from src.evaluation.runs import FoldRef, create_splits

    tmp_path, rows = sandbox
    data = tmp_path / "train.tsv"
    rows.write_csv(data, separator="\t")
    create_splits(data, get_model_config("TwoTowerGAT"), [Scenario.DRUG_COLD], k=2)
    rows.head(150).write_csv(data, separator="\t")
    with pytest.raises(EvaluationError, match="changed"):
        _train(data, monkeypatch, split=FoldRef(scenario=Scenario.DRUG_COLD, fold=0))
