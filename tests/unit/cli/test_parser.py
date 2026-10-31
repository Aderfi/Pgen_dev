"""The CLI parser and its dispatch table must agree on every mode."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.cli.app import _DISPATCH, _source
from src.cli.parser import DEFAULT_INPUT, DEFAULT_MODEL, build_parser


def test_every_mode_has_a_handler() -> None:
    sub = next(a for a in build_parser()._actions if a.dest == "mode")
    assert set(sub.choices) == set(_DISPATCH)


def test_train_defaults() -> None:
    args = build_parser().parse_args(["train"])
    assert args.model == DEFAULT_MODEL
    assert args.input == DEFAULT_INPUT
    assert args.neo4j is False
    assert args.batch_size is None


def test_neo4j_source() -> None:
    args = build_parser().parse_args(["train", "--neo4j"])
    assert _source(args) == "neo4j"


def test_input_and_neo4j_are_exclusive() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["train", "--neo4j", "-i", "x.tsv"])


def test_missing_input_exits(tmp_path: Path) -> None:
    args = build_parser().parse_args(["train", "-i", str(tmp_path / "nope.tsv")])
    with pytest.raises(SystemExit):
        _source(args)


def test_predict_requires_input() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["predict"])


def test_optuna_options() -> None:
    args = build_parser().parse_args(
        ["optuna", "--trials", "5", "--sampler", "random", "--pruner", "none"]
    )
    assert (args.trials, args.sampler, args.pruner) == (5, "random", "none")


def test_split_mode() -> None:
    args = build_parser().parse_args(
        ["split", "--scenarios", "drug_cold,pmid", "--folds", "5"]
    )
    assert args.scenarios == "drug_cold,pmid"
    assert args.folds == 5


def test_split_defaults() -> None:
    args = build_parser().parse_args(["split"])
    assert args.scenarios == "all"
    assert args.folds == 5


def test_train_on_a_fold() -> None:
    args = build_parser().parse_args(["train", "--scenario", "pmid", "--fold", "1"])
    assert args.scenario == "pmid"
    assert args.fold == 1


def test_optuna_on_a_fold() -> None:
    args = build_parser().parse_args(
        ["optuna", "--scenario", "gene_cold", "--fold", "0"]
    )
    assert (args.scenario, args.fold) == ("gene_cold", 0)


@pytest.mark.parametrize(
    "argv", [["train", "--fold", "1"], ["train", "--scenario", "pmid"]]
)
def test_scenario_and_fold_go_together(argv: list[str]) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(argv)


def test_evaluate_modes() -> None:
    p = build_parser()
    assert p.parse_args(["evaluate", "--scenario", "pmid"]).scenario == "pmid"
    assert p.parse_args(["evaluate", "--compare", "a", "b"]).compare == [
        Path("a"),
        Path("b"),
    ]
    assert p.parse_args(["evaluate", "--render", "a"]).render == Path("a")
    args = p.parse_args(
        ["evaluate", "--scenario", "pmid", "--folds", "0,2", "--bootstrap", "100"]
    )
    assert (args.folds, args.bootstrap) == ("0,2", 100)


def test_evaluate_modes_are_exclusive() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["evaluate", "--scenario", "pmid", "--render", "a"])
    with pytest.raises(SystemExit):
        build_parser().parse_args(["evaluate"])
