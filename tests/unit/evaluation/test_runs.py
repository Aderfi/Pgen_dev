"""Run names and fold-frame loading guards."""

from __future__ import annotations

import pytest

from src.config import get_model_config
from src.core import EvaluationError
from src.evaluation.entities import Scenario
from src.evaluation.runs import FoldRef, load_fold_frames, run_name


def test_run_name() -> None:
    assert run_name("M", None) == "M"
    assert (
        run_name("M", FoldRef(scenario=Scenario.DRUG_COLD, fold=2))
        == "M__drug_cold__fold2"
    )


def test_fold_ref_rejects_negative_fold() -> None:
    with pytest.raises(ValueError):
        FoldRef(scenario=Scenario.PMID, fold=-1)


def test_neo4j_source_cannot_use_a_split() -> None:
    with pytest.raises(EvaluationError, match="snapshot"):
        load_fold_frames(
            "neo4j",
            get_model_config("TwoTowerGAT"),
            FoldRef(scenario=Scenario.PMID, fold=0),
        )


def test_variant_scenarios_need_the_catalog(tmp_path, monkeypatch) -> None:
    from src.evaluation import runs

    monkeypatch.setattr(runs, "catalog_sha256", lambda: "absent")
    for scenario in (Scenario.VARIANT_COLD, Scenario.PAIR):
        with pytest.raises(EvaluationError, match="catalog"):
            runs.create_splits(
                tmp_path / "x.tsv", get_model_config("TwoTowerGAT"), [scenario]
            )
