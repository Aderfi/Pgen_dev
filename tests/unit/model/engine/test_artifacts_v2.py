import pickle

from src.model.engine.predictor import PGenPredictor


def test_v2_bundle_round_trip(tmp_path):
    bundle = {
        "encoders": {},
        "drug_dim": 6,
        "geno_dim": 5,
        "edge_dims": {"drug_edge": 0, "ddi_edge": 0, "geno_edge": 0},
        "aux_dims": {"drug_global": 0, "drug_admet": 0, "geno_global": 0},
        "axis_specs": {"pheno": {"name": "pheno", "dim": 3, "kind": "multiclass"}},
        "label_table": {"tuples": [], "labels": []},
        "switches": {"use_polypharmacy": False, "use_cross_attention": False},
        "schema_version": 2,
    }
    p = tmp_path / "encoders_TwoTowerGAT.pkl"
    p.write_bytes(pickle.dumps(bundle))
    loaded = PGenPredictor._load_training_artifacts(p)  # staticmethod
    assert loaded["schema_version"] == 2
    assert loaded["axis_specs"]["pheno"]["dim"] == 3


def test_v1_plain_dict_still_loads(tmp_path):
    legacy = {"pheno": object()}  # legacy encoders-only dict
    p = tmp_path / "encoders_old.pkl"
    p.write_bytes(pickle.dumps(legacy))
    loaded = PGenPredictor._load_training_artifacts(p)
    assert loaded["schema_version"] == 1
