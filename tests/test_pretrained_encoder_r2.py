"""Serialized numerical fixtures, not imported or evaluated public weights."""
from copy import deepcopy

import numpy as np
import pytest
import torch

from opal2.model import GroupedProfileEncoder
from opal2.training import load_pretrained_encoder, fit_model, load_model
from opal2.provenance import assert_unseen_outcomes, bind_fitting_provenance


def checkpoint_fixture(encoder, names=None, center=None, scale=None):
    d = encoder.feature_dim
    return {
        "schema_version": 1,
        "encoder_state_dict": {k: v.detach().clone() for k, v in encoder.state_dict().items()},
        "feature_names": list(map(str, names)) if names is not None else [f"numeric_coordinate_{i}" for i in range(d)],
        "feature_groups": deepcopy(encoder.feature_groups),
        "encoder_config": {"hidden_dim": encoder.hidden_dim,
                           "attention_layers": encoder.attention_layers,
                           "attention_heads": encoder.attention_heads},
        "input_transform": {"center": np.zeros(d).tolist() if center is None else np.asarray(center).tolist(),
                            "scale": np.ones(d).tolist() if scale is None else np.asarray(scale).tolist()},
        "pretraining_ids": ["numerical_pretraining_fixture_0", "numerical_pretraining_fixture_1"],
        "provenance": {"source": "pytest serialized numerical weights fixture; not a public model",
                       "training_data": "generated numerical tensor fixture only; no biological observations",
                       "identity_namespace": "opal2_dataset_compound_ids",
                       "pretraining_ids_complete": True},
    }


def encoder_fixture():
    torch.set_num_threads(1)
    return GroupedProfileEncoder({"group_A": [0, 2], "group_B": [1, 3]}, 16,
                                 attention_layers=2, attention_heads=4).eval()


def import_fixture(encoder, path):
    return load_pretrained_encoder(encoder, path,
        feature_names=[f"numeric_coordinate_{i}" for i in range(4)],
        input_center=np.zeros(4), input_scale=np.ones(4), protected_ids=["held_out_0"])


def test_real_serialized_compatible_weights_import_changes_predictions_exactly(tmp_path):
    torch.manual_seed(801)
    source, target = encoder_fixture(), encoder_fixture()
    payload = checkpoint_fixture(source)
    path = tmp_path / "explicit_numerical_encoder_fixture.pt"
    torch.save(payload, path)
    x = torch.tensor([[.1, -.7, .3, 2.], [.3, .2, .4, -.5]])
    assert not torch.equal(source(x), target(x))
    metadata = import_fixture(target, path)
    torch.testing.assert_close(source(x), target(x), rtol=0, atol=0)
    assert metadata["pretraining_ids"] == payload["pretraining_ids"]
    assert "encoder_state_dict" not in metadata
    assert "not a public model" in metadata["provenance"]["source"]


@pytest.mark.parametrize("fault,match", [
    ("feature_order", "feature coordinates"),
    ("same_shape_group_assignment", "group coordinates"),
    ("same_shape_group_order", "ordered feature groups"),
    ("spoofed_index_buffer", "index buffers"),
    ("same_shape_attention_heads", "architecture"),
    ("wrong_input_scale", "input transform"),
    ("empty_provenance", "provenance"),
    ("empty_source", "provenance"),
    ("empty_training_data", "provenance"),
    ("unknown_identity_mapping", "provenance"),
    ("incomplete_ids", "provenance"),
    ("empty_ids", "Pretraining IDs"),
    ("string_ids", "Pretraining IDs"),
    ("duplicate_ids", "Pretraining IDs"),
    ("held_out_overlap", "overlaps held-out"),
    ("nonfinite_weights", "incompatible"),
    ("missing_state_key", "state keys"),
    ("unmapped_sources", "source groups"),
])
def test_reject_incompatible_or_unauditable_serialized_weights_without_mutation(tmp_path, fault, match):
    source, target = encoder_fixture(), encoder_fixture()
    p = checkpoint_fixture(source)
    if fault == "feature_order":
        p["feature_names"] = p["feature_names"][::-1]
    elif fault == "same_shape_group_assignment":
        p["feature_groups"] = {"group_A": [0, 1], "group_B": [2, 3]}
    elif fault == "same_shape_group_order":
        p["feature_groups"] = dict(reversed(list(p["feature_groups"].items())))
    elif fault == "spoofed_index_buffer":
        p["encoder_state_dict"]["indices_0"] = torch.tensor([0, 1])
    elif fault == "same_shape_attention_heads":
        p["encoder_config"]["attention_heads"] = 2
    elif fault == "wrong_input_scale":
        p["input_transform"]["scale"][1] = 2
    elif fault == "empty_provenance":
        p["provenance"] = ""
    elif fault == "empty_source":
        p["provenance"]["source"] = "  "
    elif fault == "empty_training_data":
        p["provenance"]["training_data"] = ""
    elif fault == "unknown_identity_mapping":
        p["provenance"]["identity_namespace"] = "unmapped_image_ids"
    elif fault == "incomplete_ids":
        p["provenance"]["pretraining_ids_complete"] = False
    elif fault == "empty_ids":
        p["pretraining_ids"] = []
    elif fault == "string_ids":
        p["pretraining_ids"] = "not an identity list"
    elif fault == "duplicate_ids":
        p["pretraining_ids"] *= 2
    elif fault == "held_out_overlap":
        p["pretraining_ids"].append("held_out_0")
    elif fault == "nonfinite_weights":
        p["encoder_state_dict"]["group_embedding"][0, 0] = float("nan")
    elif fault == "missing_state_key":
        del p["encoder_state_dict"]["group_embedding"]
    elif fault == "unmapped_sources":
        p["pretraining_source_groups"] = [17]
    path = tmp_path / "incompatible_numerical_fixture.pt"
    torch.save(p, path)
    before = {k: v.clone() for k, v in target.state_dict().items()}
    with pytest.raises(ValueError, match=match):
        import_fixture(target, path)
    for k, v in target.state_dict().items():
        torch.testing.assert_close(v, before[k], rtol=0, atol=0)


@pytest.mark.parametrize("known_sources", [False, True])
def test_pretrained_fit_reload_retains_identity_and_source_boundaries(tmp_path, known_sources):
    from test_training_integration import schema_fixture
    from opal2.config import TrainConfig
    from opal2.data import TrainScaler
    ds = schema_fixture(d=8)
    splits = {"train": np.arange(6), "validation": np.arange(6, 8),
              "calibration": np.arange(8, 10), "evaluation": np.arange(10, 12)}
    scaler = TrainScaler.fit(ds, splits["train"])
    source = GroupedProfileEncoder(ds.feature_groups, 16, attention_layers=2, attention_heads=4)
    payload = checkpoint_fixture(source, ds.feature_names, scaler.y_center, scaler.y_scale)
    if known_sources:
        payload["pretraining_source_groups"] = [17]
        payload["provenance"]["source_namespace"] = "opal2_dataset_source_groups"
    path = tmp_path / "actual_serialized_numeric_encoder.pt"
    torch.save(payload, path)
    cfg = TrainConfig(encoder_policy="pretrained_frozen", pretrained_encoder_checkpoint=str(path),
                      use_library=False, hidden_dim=16, latent_rank=4, residual_rank=2,
                      epochs=1, batch_size=6, threads=1)
    directory = tmp_path / "numerical_integration_training"
    model, _ = fit_model(ds, splits, cfg, directory)
    restored, _, _, artifact = load_model(directory)
    for loaded in (model, restored):
        assert loaded.pretraining_ids == tuple(payload["pretraining_ids"])
        assert set(payload["pretraining_ids"]).issubset(loaded.fitting_ids)
        assert loaded.training_ids == tuple(ds.ids[splits["train"]])
        assert loaded.fitting_sources_known is known_sources
        assert loaded.pretraining_sources_unknown is (not known_sources)
        assert loaded.fitting_source_groups == ({0, 17} if known_sources else {0})
        assert not any(p.requires_grad for p in loaded.profile_encoder.parameters())
        with pytest.raises(ValueError, match="pretraining"):
            assert_unseen_outcomes(loaded, [payload["pretraining_ids"][0]])
        for key, value in loaded.profile_encoder.state_dict().items():
            torch.testing.assert_close(value, source.state_dict()[key], rtol=0, atol=0)
    assert artifact["encoder_pretraining"]["pretraining_ids"] == payload["pretraining_ids"]


def test_legacy_fitting_sources_are_unknown_not_empty_means_unseen():
    class Model:
        pass
    model = bind_fitting_provenance(Model(), ["train"], ["validation"])
    assert model.fitting_source_groups == frozenset()
    assert model.fitting_sources_known is False
