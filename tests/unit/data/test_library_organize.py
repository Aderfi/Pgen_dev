"""Tests for src.data.library.organize."""

from pathlib import Path

from src.data.library.organize import organize_gene_files


class TestOrganizeGeneFiles:
    def test_moves_files_into_gene_subdirs(self, tmp_path: Path) -> None:
        (tmp_path / "CYP2D6_star4.pt").touch()
        (tmp_path / "CYP2D6_star10.pt").touch()
        (tmp_path / "DPYD_star2A.pt").touch()

        moved = organize_gene_files(tmp_path)

        assert moved == {"CYP2D6": 2, "DPYD": 1}
        assert (tmp_path / "CYP2D6" / "CYP2D6_star4.pt").exists()
        assert (tmp_path / "CYP2D6" / "CYP2D6_star10.pt").exists()
        assert (tmp_path / "DPYD" / "DPYD_star2A.pt").exists()

    def test_ugt1a_subfamily_merged(self, tmp_path: Path) -> None:
        for sub in ("UGT1A1", "UGT1A4", "UGT1A10"):
            (tmp_path / f"{sub}_star1.pt").touch()
        moved = organize_gene_files(tmp_path)
        # All UGT1A* go into the single UGT1A directory.
        assert moved.get("UGT1A") == 3
        for sub in ("UGT1A1", "UGT1A4", "UGT1A10"):
            assert (tmp_path / "UGT1A" / f"{sub}_star1.pt").exists()

    def test_missing_dir_returns_empty(self, tmp_path: Path) -> None:
        assert organize_gene_files(tmp_path / "nope") == {}

    def test_idempotent_when_already_organized(self, tmp_path: Path) -> None:
        (tmp_path / "CYP2D6").mkdir()
        (tmp_path / "CYP2D6" / "CYP2D6_star4.pt").touch()
        # No flat files at the root; organize_gene_files is a no-op.
        moved = organize_gene_files(tmp_path)
        assert moved == {}
        assert (tmp_path / "CYP2D6" / "CYP2D6_star4.pt").exists()
