"""Tests for src.data.library.manifest."""

import json
from pathlib import Path

from src.data.library.manifest import BuildManifest


class TestBuildManifest:
    def test_round_trip(self, tmp_path: Path) -> None:
        m = BuildManifest()
        m.mark_drug_done(2244)
        m.mark_gene_done("CYP2D6", "*4")
        m.mark_drug_failed(99, "Invalid SMILES")
        m.mark_gene_failed("CYP2D6", "*99", "no FASTA match")

        path = tmp_path / "manifest.json"
        m.save(path)
        assert path.exists()

        loaded = BuildManifest.load_or_empty(path)
        assert loaded.has_drug(2244)
        assert loaded.has_gene("CYP2D6", "*4")
        assert "drug:99" in loaded.failed
        assert "gene:CYP2D6/*99" in loaded.failed

    def test_load_or_empty_missing_file(self, tmp_path: Path) -> None:
        m = BuildManifest.load_or_empty(tmp_path / "does_not_exist.json")
        assert m.completed_drug_cids == []
        assert m.completed_gene_keys == []

    def test_load_or_empty_corrupt_file(self, tmp_path: Path) -> None:
        path = tmp_path / "manifest.json"
        path.write_text("not valid json {{{")
        m = BuildManifest.load_or_empty(path)
        # Should silently fall back to empty rather than crash.
        assert m.completed_drug_cids == []

    def test_no_duplicates_on_replay(self, tmp_path: Path) -> None:
        m = BuildManifest()
        for _ in range(3):
            m.mark_drug_done(2244)
        assert m.completed_drug_cids == [2244]

    def test_atomic_save_leaves_no_temp_files(self, tmp_path: Path) -> None:
        m = BuildManifest()
        m.mark_drug_done(2244)
        m.save(tmp_path / "manifest.json")
        # Hidden temp files used during atomic write should be gone.
        leftovers = list(tmp_path.glob(".manifest.*.tmp"))
        assert leftovers == []

    def test_save_produces_valid_json(self, tmp_path: Path) -> None:
        m = BuildManifest()
        m.mark_gene_done("CYP2D6", "*4")
        path = tmp_path / "manifest.json"
        m.save(path)
        with path.open() as f:
            data = json.load(f)
        assert "completed_gene_keys" in data
        assert "CYP2D6/*4" in data["completed_gene_keys"]
