"""Tests for the multi-format drug catalog reader, failure logging and counter.

Covers :func:`load_drug_records` (TSV + JSON dispatch) and
:class:`DrugGraphBuilder` end-to-end on a tiny in-memory catalog, asserting the
per-:class:`DrugFailureCategory` counter and the ``logs/library`` failure file.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
import torch

from src.data.library.drugs import (
    DRUG_NODE_DIM,
    DrugFailureCategory,
    DrugGraphBuilder,
    FeatureSaturation,
    _one_hot,
    load_drug_records,
    smiles_to_graph,
)
from src.data.library.manifest import BuildManifest

if TYPE_CHECKING:
    from pathlib import Path

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
CAFFEINE = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
# Hypervalent molecules: sulfur in SF6 is SP3D2 / degree 6; phosphorus in PF5 is
# SP3D / degree 5 — all four values fall outside the frozen one-hot bins.
SF6 = "FS(F)(F)(F)(F)F"
PF5 = "FP(F)(F)(F)F"


def _write_tsv(path: Path, rows: list[tuple[str, str, str]]) -> None:
    lines = ["cid\tsmiles\tcmpd_name_cleaned"]
    lines += [f"{cid}\t{smiles}\t{name}" for cid, smiles, name in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class TestLoadDrugRecords:
    def test_reads_tsv(self, tmp_path: Path) -> None:
        tsv = tmp_path / "drugs.tsv"
        _write_tsv(tsv, [("1", ASPIRIN, "aspirin"), ("2", CAFFEINE, "caffeine")])
        records = load_drug_records(tsv)
        assert [r["cid"] for r in records] == [1, 2]
        assert records[0]["smiles"] == ASPIRIN
        assert records[0]["name"] == "aspirin"

    def test_reads_json(self, tmp_path: Path) -> None:
        js = tmp_path / "drugs.json"
        # Keys with stray whitespace, mirroring BACKUPS/cid_smiles_dict.json.
        js.write_text(json.dumps({" 1": ASPIRIN, "2": CAFFEINE}), encoding="utf-8")
        records = load_drug_records(js)
        assert records[0]["cid"] == " 1"  # raw; parsed by the build loop
        assert records[0]["smiles"] == ASPIRIN
        assert records[0]["name"] is None

    def test_rejects_unknown_extension(self, tmp_path: Path) -> None:
        bad = tmp_path / "drugs.parquet"
        bad.write_text("x", encoding="utf-8")
        with pytest.raises(ValueError, match="Unsupported drug catalog format"):
            load_drug_records(bad)


class TestDrugGraphBuilder:
    def test_builds_from_tsv(self, tmp_path: Path) -> None:
        tsv = tmp_path / "drugs.tsv"
        _write_tsv(tsv, [("1", ASPIRIN, "aspirin"), ("2", CAFFEINE, "caffeine")])
        out = tmp_path / "drugs"
        builder = DrugGraphBuilder(out, failures_log=tmp_path / "logs" / "fail.log")

        built, skipped, failed = builder.build(tsv, manifest=BuildManifest())

        assert (built, skipped, failed) == (2, 0, 0)
        assert {p.name for p in out.glob("*.pt")} == {
            "1_aspirin.pt",
            "2_caffeine.pt",
        }

    def test_builds_from_json_with_cid_fallback_name(self, tmp_path: Path) -> None:
        js = tmp_path / "drugs.json"
        js.write_text(json.dumps({" 7": ASPIRIN}), encoding="utf-8")
        out = tmp_path / "drugs"
        builder = DrugGraphBuilder(out, failures_log=tmp_path / "logs" / "fail.log")

        built, _, failed = builder.build(js, manifest=BuildManifest())

        assert (built, failed) == (1, 0)
        # No name column → "<cid>_cid<cid>.pt", still matches the index regex.
        assert (out / "7_cid7.pt").exists()

    def test_resume_skips_existing(self, tmp_path: Path) -> None:
        tsv = tmp_path / "drugs.tsv"
        _write_tsv(tsv, [("1", ASPIRIN, "aspirin")])
        out = tmp_path / "drugs"
        DrugGraphBuilder(out).build(tsv, manifest=BuildManifest())

        built, skipped, failed = DrugGraphBuilder(out).build(
            tsv, manifest=BuildManifest()
        )
        assert (built, skipped, failed) == (0, 1, 0)

    def test_failure_counter_by_nature(self, tmp_path: Path) -> None:
        tsv = tmp_path / "drugs.tsv"
        _write_tsv(
            tsv,
            [
                ("1", ASPIRIN, "ok"),
                ("xyz", ASPIRIN, "bad_cid"),  # non_integer_cid
                ("3", "", "no_smiles"),  # missing_smiles
                ("4", "Q##notReal", "bad_smiles"),  # invalid_smiles
            ],
        )
        log = tmp_path / "logs" / "library" / "fail.log"
        builder = DrugGraphBuilder(tmp_path / "drugs", failures_log=log)

        built, _, failed = builder.build(tsv, manifest=BuildManifest())

        assert built == 1
        assert failed == 3
        assert builder.failure_counts[DrugFailureCategory.NON_INTEGER_CID] == 1
        assert builder.failure_counts[DrugFailureCategory.MISSING_SMILES] == 1
        assert builder.failure_counts[DrugFailureCategory.INVALID_SMILES] == 1

        # Failures were logged to logs/library via the dedicated FileHandler.
        contents = log.read_text(encoding="utf-8")
        assert "nature=non_integer_cid" in contents
        assert "nature=missing_smiles" in contents
        assert "nature=invalid_smiles" in contents
        assert "Build run finished" in contents

    def test_manifest_records_failed_cids(self, tmp_path: Path) -> None:
        tsv = tmp_path / "drugs.tsv"
        _write_tsv(tsv, [("4", "Q##notReal", "bad")])
        manifest = BuildManifest()
        DrugGraphBuilder(tmp_path / "drugs").build(tsv, manifest=manifest)
        assert manifest.failed["drug:4"]


class TestOneHotSaturation:
    """Numeric-bin overflow encoded as all-zeros — the only residual info loss.

    Categorical fields now use explicit "other" buckets, so the redesigned schema
    is saturation-free on real molecules; the counter is a safety net for genuine
    out-of-range numeric values.
    """

    def test_other_bucket_never_saturates(self) -> None:
        sat = FeatureSaturation()
        enc = _one_hot("Xe", ["C", "N"], other=True, sink=sat, feature="element")
        assert enc == [0.0, 0.0, 1.0]  # routed to the trailing "other" slot
        assert sat.total == 0

    def test_numeric_bin_overflow_is_recorded(self) -> None:
        sat = FeatureSaturation()
        enc = _one_hot(9, [0, 1, 2], sink=sat, feature="degree")
        assert enc == [0.0, 0.0, 0.0]  # all-zeros
        assert sat.by_value["degree=9"] == 1
        assert sat.total == 1

    def test_redesign_eliminates_hypervalent_saturation(self) -> None:
        # SF6 (SP3D2, degree 6) and PF5 (SP3D, degree 5) used to saturate; the
        # full-vocab hybridization + wider degree/valence bins encode them losslessly.
        for smi in (SF6, PF5):
            sat = FeatureSaturation()
            g = smiles_to_graph(smi, saturation=sat)
            assert g is not None
            assert sat.total == 0, f"{smi} should no longer saturate"

    def test_schema_and_output_identical_with_or_without_sink(self) -> None:
        # Instrumentation must never alter the produced graph.
        plain = smiles_to_graph(SF6)
        instrumented = smiles_to_graph(SF6, saturation=FeatureSaturation())
        assert plain is not None and instrumented is not None
        assert plain.x.shape[1] == DRUG_NODE_DIM == 61
        assert torch.equal(plain.x, instrumented.x)
        assert torch.equal(plain.edge_attr, instrumented.edge_attr)

    def test_clean_molecule_has_no_saturation(self) -> None:
        sat = FeatureSaturation()
        smiles_to_graph(ASPIRIN, saturation=sat)
        smiles_to_graph(CAFFEINE, saturation=sat)
        assert sat.total == 0

    def test_builder_writes_saturation_summary(self, tmp_path: Path) -> None:
        tsv = tmp_path / "drugs.tsv"
        _write_tsv(tsv, [("1", ASPIRIN, "aspirin"), ("2", SF6, "sf6")])
        log = tmp_path / "logs" / "library" / "saturation.log"
        builder = DrugGraphBuilder(tmp_path / "drugs", saturation_log=log)

        built, _, failed = builder.build(tsv, manifest=BuildManifest())

        assert (built, failed) == (2, 0)
        assert builder.saturation.total == 0  # new schema is saturation-free here
        assert builder.drugs_with_saturation == 0
        contents = log.read_text(encoding="utf-8")
        assert "Build run finished" in contents
        assert "total_events=0" in contents


SODIUM_ACETATE = "CC(=O)[O-].[Na+]"  # acetate (4 atoms) + Na+ counterion


class TestSaltStripping:
    """Reducing multi-fragment SMILES to the largest fragment (salt removal)."""

    def test_strip_keeps_only_largest_fragment(self) -> None:
        stripped = smiles_to_graph(SODIUM_ACETATE, strip_salts=True)
        kept = smiles_to_graph(SODIUM_ACETATE, strip_salts=False)
        acetate = smiles_to_graph("CC(=O)[O-]")
        assert stripped is not None and kept is not None and acetate is not None
        # Stripped graph == bare acetate; unstripped keeps the extra Na+ atom.
        assert stripped.num_nodes == acetate.num_nodes
        assert kept.num_nodes == acetate.num_nodes + 1

    def test_single_fragment_is_untouched(self) -> None:
        plain = smiles_to_graph(ASPIRIN)
        stripped = smiles_to_graph(ASPIRIN, strip_salts=True)
        assert plain is not None and stripped is not None
        assert torch.equal(plain.x, stripped.x)

    def test_counterion_tolerated_but_still_removed(self) -> None:
        # The Na+ counterion no longer *saturates* (it hits "other" buckets), but
        # it is still noise — stripping removes the atom entirely.
        sat = FeatureSaturation()
        smiles_to_graph(SODIUM_ACETATE, saturation=sat, strip_salts=False)
        assert sat.total == 0
        stripped = smiles_to_graph(SODIUM_ACETATE, strip_salts=True)
        kept = smiles_to_graph(SODIUM_ACETATE, strip_salts=False)
        assert stripped is not None and kept is not None
        assert stripped.num_nodes == kept.num_nodes - 1

    def test_builder_counts_stripped_salts(self, tmp_path: Path) -> None:
        tsv = tmp_path / "drugs.tsv"
        _write_tsv(
            tsv,
            [
                ("1", ASPIRIN, "aspirin"),  # single fragment
                ("2", SODIUM_ACETATE, "na_acetate"),  # salt → stripped
            ],
        )
        builder = DrugGraphBuilder(tmp_path / "drugs", strip_salts=True)
        built, _, failed = builder.build(tsv, manifest=BuildManifest())
        assert (built, failed) == (2, 0)
        assert builder.salts_stripped == 1

    def test_builder_can_keep_salts(self, tmp_path: Path) -> None:
        tsv = tmp_path / "drugs.tsv"
        _write_tsv(tsv, [("2", SODIUM_ACETATE, "na_acetate")])
        builder = DrugGraphBuilder(tmp_path / "drugs", strip_salts=False)
        builder.build(tsv, manifest=BuildManifest())
        assert builder.salts_stripped == 0
