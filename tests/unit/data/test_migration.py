from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest
import torch

UNKNOWN_CATEGORY_LABEL = "__UNKNOWN__"

# AJUSTA ESTAS IMPORTACIONES AL NOMBRE DE TU ARCHIVO
# Por ejemplo, si tu código está en 'src/data_loader.py':
# from src.data_loader import DoubleTowerDataset, PGenProcessor (o la clase que tenga fit/transform)
# Para este ejemplo, asumo que las clases están disponibles o las simulamos abajo.


# --- MOCKS PARA DEPENDENCIAS EXTERNAS ---
@pytest.fixture
def mock_graph_data():
    """Crea un objeto PyG Data falso."""
    from torch_geometric.data import Data

    return Data(x=torch.randn(5, 10), edge_index=torch.tensor([[0, 1], [1, 0]]))


@pytest.fixture
def sample_df():
    """Crea un DataFrame de Polars representativo."""
    return pl.DataFrame(
        {
            "drug_id": ["1001", "1002", "1003", "1001"],
            "gene": ["CYP2D6", "CYP2C19", "DPYD", "CYP2D6"],
            "genotype": ["rs3892097", "*17", "*2A", "*1"],
            "outcome": [
                "Toxicity",
                "Efficacy",
                "Toxicity",
                "No_Effect",
            ],  # Single-label
            "side_effects": [
                "Headache|Nausea",
                "Nausea",
                None,
                "Vomiting|Headache",
            ],  # Multi-label
        }
    )


class _FakeResolver:
    """Stand-in GenotypeResolver: returns a fixed subgraph for any genotype."""

    def resolve(self, gene: str, genotype: str):  # noqa: ARG002
        from torch_geometric.data import Data

        g = Data(
            x=torch.randn(2, 30),
            edge_index=torch.tensor([[0], [1]]),
            edge_attr=torch.zeros(1, 2),
        )
        g.geno_function = torch.zeros(1, 6)
        g.gene = gene
        return g


# --- TESTS DE PREPROCESAMIENTO (ENCODERS) ---


class TestEncoders:
    """Valida la lógica de fit y transform migrada a Polars."""

    def test_fit_transform_logic(self, sample_df):
        """
        Simula la lógica de tu procesador para verificar:
        1. Single-label: Conversión a enteros.
        2. Multi-label: Conversión a Multi-hot (Listas/Tensores).
        3. Manejo de Nulos.
        """
        # Simulamos la clase que contiene tus métodos fit/transform
        # Aquí inyectamos la lógica que migramos anteriormente
        from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder  # noqa

        # 1. Test Single Label (Outcome)
        # -----------------------------
        # Lógica fit
        target_col = "outcome"
        uniques = (
            sample_df.select(pl.col(target_col).drop_nulls().unique())
            .to_series()
            .to_list()
        )
        le = LabelEncoder().fit(sorted(uniques + [UNKNOWN_CATEGORY_LABEL]))

        # Lógica transform (Polars replace)
        mapping = {k: i for i, k in enumerate(le.classes_)}
        unknown_idx = mapping[UNKNOWN_CATEGORY_LABEL]

        encoded_series = sample_df.select(
            pl.col(target_col).replace(mapping, default=unknown_idx).cast(pl.Int64)
        ).to_series()

        # Assertions
        assert encoded_series.dtype == pl.Int64
        assert encoded_series.null_count() == 0
        assert len(encoded_series) == 4  # noqa
        # Verificar que "Toxicity" tiene el mismo código
        assert encoded_series[0] == encoded_series[2]

        # 2. Test Multi Label (Side Effects)
        # ----------------------------------
        # Lógica fit/transform
        col_multi = "side_effects"

        # Polars: Split string -> List
        parsed_df = sample_df.select(
            pl.col(col_multi)
            .str.split("|")
            .fill_null(pl.lit([], dtype=pl.List(pl.String)))
        )

        mlb = MultiLabelBinarizer().fit(parsed_df.to_series().to_list())

        # Verificar clases detectadas
        assert "Headache" in mlb.classes_
        assert "Nausea" in mlb.classes_
        assert "Vomiting" in mlb.classes_

        # Simular transformación a Tensor
        matrix = mlb.transform(parsed_df.to_series().to_list())
        tensor = torch.tensor(matrix, dtype=torch.float32)

        assert tensor.shape == (4, len(mlb.classes_))
        # La fila 3 es "None" (originalmente nulo) -> debe ser todo ceros
        assert torch.sum(tensor[2]) == 0


# --- TESTS DEL DATASET (DOUBLE TOWER) ---


class TestDoubleTowerDataset:
    """Valida la carga de datos, cache y __getitem__."""

    @patch("torch.load")  # Mock disk load for the drug graph
    @patch("pathlib.Path.exists", return_value=True)
    def test_initialization_and_getitem(
        self, mock_exists, mock_torch_load, sample_df, mock_graph_data
    ):
        from src.data.datasets import DoubleTowerDataset

        mock_torch_load.return_value = mock_graph_data

        # Mock only the drug index; the genotype side is served by an injected
        # resolver, so no on-disk gene library is needed.
        with patch(
            "src.data.graph_indexing.GraphIndexBuilder.build_drug_index",
            return_value={"1001": Path("d1.pt")},
        ):
            dataset = DoubleTowerDataset(
                df=sample_df,
                drug_col="drug_id",
                geno_col="genotype",
                gene_col="gene",
                genotype_resolver=_FakeResolver(),
                target_cols=["outcome", "side_effects"],
                multilabel_cols=["side_effects"],
                preload_ram=True,
            )

            assert len(dataset) == 4  # noqa

            sample = dataset[0]

            assert "drug_data" in sample
            assert "geno_data" in sample
            assert "targets" in sample

            # Genotype subgraph comes from the resolver (30-dim nodes).
            assert sample["geno_data"].x.shape[1] == 30
            assert isinstance(sample["targets"]["outcome"], torch.Tensor)
            assert isinstance(sample["targets"]["side_effects"], torch.Tensor)

            assert sample["targets"]["outcome"].dtype == torch.long
            assert sample["targets"]["side_effects"].dtype == torch.float32

    @patch("pathlib.Path.exists", return_value=True)
    def test_unresolved_genotype_falls_back_to_empty(self, mock_exists, sample_df):
        """A gene the resolver can't place yields a placeholder geno graph."""
        from src.data.datasets import DoubleTowerDataset

        class _NullResolver:
            def resolve(self, gene, genotype):  # noqa: ARG002
                return None

        with patch(
            "src.data.graph_indexing.GraphIndexBuilder.build_drug_index",
            return_value={},
        ):
            dataset = DoubleTowerDataset(
                df=sample_df,
                drug_col="drug_id",
                geno_col="genotype",
                gene_col="gene",
                genotype_resolver=_NullResolver(),
                target_cols=["outcome"],
                multilabel_cols=[],
            )
            geno = dataset[0]["geno_data"]
            assert geno.x.shape == (1, 30)  # placeholder dims
            assert geno.geno_function.shape == (1, 6)

    def test_polars_row_access_optimization(self, sample_df):
        """Verifica el acceso O(1) por listas de lookup (no .iloc)."""
        lookup_drugs = sample_df["drug_id"].to_list()
        lookup_genos = sample_df["genotype"].to_list()

        idx = 1
        assert lookup_drugs[idx] == "1002"
        assert lookup_genos[idx] == "*17"
