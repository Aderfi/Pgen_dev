from src.model.architectures import PharmagenConfig, PharmagenTwoTower, TaskSpec


def _cfg():
    return PharmagenConfig(
        drug_in_features=6,
        drug_edge_dim=4,
        drug_hidden_dim=16,
        geno_in_features=5,
        geno_edge_dim=None,
        geno_hidden_dim=16,
        embedding_dim=16,
        num_layers=2,
        heads=2,
        dropout=0.0,
        use_polypharmacy=False,
        use_cross_attention=False,
        axes={"pheno": TaskSpec(dim=3, kind="multiclass")},
    )


def test_model_builds_and_has_expected_top_level_modules():
    model = PharmagenTwoTower(_cfg())
    names = dict(model.named_children())
    assert "drug_tower" in names and "geno_tower" in names and "axis_heads" in names
    counts = model.count_parameters()
    assert counts["total"] > 0
