"""Engineering fixtures for biological metadata; no experimental data are read."""
from dataclasses import asdict, replace
import json

import numpy as np
import pytest
import torch

from opal2.biology import (BiologyRecord, TypedValue, Provenance, MechanismRelation,
    EvidenceAwareMechanismPrior, fit_vocabulary, encode_relations, load_biology,
    save_biology, validate_records, validate_vocabulary, TOKEN_FIELDS)
from opal2.config import TrainConfig
from opal2.data import (TrainScaler, load_dataset, save_dataset, make_inference_batch,
    make_episode, collate_episodes)
from opal2.model import MeasurementWorldModel
from opal2.training import model_kwargs, fixed_batch, load_model
from test_data import records as measurement_fixture


def provenance(source="fixture-curation"):
    return Provenance(source, "urn:engineering-fixture:not-biological-evidence", "fixture-v1", "2026-09-12")


def value(kind, value, **kwargs):
    return TypedValue(kind, "known", value, provenance=provenance(), availability="decision", **kwargs)


def relation(subject="PERTURBATION:self", target="HGNC:TEST1", **kwargs):
    base = dict(subject=subject, predicate="targets", object=target, direction="inhibition",
                evidence_family="curatorial", evidence_code="not_supplied", confidence=None,
                provenance=provenance(), availability="decision")
    base.update(kwargs)
    return MechanismRelation(**base)


def annotated(unit="unit_0", relations=None, **kwargs):
    base = dict(unit_id=unit, perturbation_type="small_molecule",
                perturbation={"entities": value("entities", [f"FIXTURE:{unit}"]),
                              "dose": value("quantity", 10., unit="uM", interpretation="nominal_protocol")},
                relation_coverage="partial", relations=(relation(subject=f"FIXTURE:{unit}"),)
                if relations is None else tuple(relations))
    base.update(kwargs)
    return BiologyRecord(**base)


def tensors(encoded):
    return {key: torch.as_tensor(array) for key, array in encoded.items()}


def dataset_with_annotations():
    ds = measurement_fixture()
    return replace(ds, biology_records=tuple(annotated(str(unit)) for unit in ds.ids))


def configuration(**kwargs):
    base = dict(hidden_dim=16, latent_rank=3, residual_rank=2, use_jepa=False,
                group_attention_layers=0, attention_heads=4, use_library=False, threads=1)
    base.update(kwargs)
    return TrainConfig(**base).validate()


def test_typed_metadata_and_evidence_reject_invented_values():
    with pytest.raises(ValueError, match="invented"):
        TypedValue("quantity", "unknown", 10., "uM")
    with pytest.raises(ValueError, match="unit"):
        value("quantity", 10.)
    with pytest.raises(ValueError, match="provenance"):
        TypedValue("category", "known", "U2OS", provenance="not-a-source")
    with pytest.raises(ValueError, match="documented control"):
        BiologyRecord("drug", "small_molecule", relation_coverage="known_empty", coverage_provenance=provenance())
    control = BiologyRecord("control", "vehicle_control", relation_coverage="known_empty", coverage_provenance=provenance())
    assert control.relations == ()
    for code, family in [("IDA", "experimental"), ("IBA", "phylogenetic"), ("ISS", "computational"),
                         ("TAS", "author"), ("IC", "curatorial"), ("IEA", "automatic")]:
        assert relation(evidence_code=code, evidence_family=family).confidence is None
        with pytest.raises(ValueError, match="inconsistent"):
            relation(evidence_code=code, evidence_family="unknown")
    assert relation(evidence_family="curated").evidence_family == "curatorial"
    with pytest.raises(ValueError, match="Confidence"):
        relation(confidence=1.1)


def test_relation_graph_keeps_target_pathway_subject_and_normalizes_own_drug():
    a = annotated(relations=[relation(subject="FIXTURE:unit_0"),
        relation(subject="HGNC:TEST1", target="GO:000TEST", predicate="participates_in", direction="membership")])
    b = annotated("unit_1")
    vocab = fit_vocabulary([a, b], ["unit_0"])
    assert vocab["fields"]["subject"] == ["HGNC:TEST1", "PERTURBATION:self"]
    assert all("unit_0" not in token for values in vocab["fields"].values() for token in values)
    assert encode_relations([b], vocab)["biology_mask"].sum() == 1
    with pytest.raises(ValueError, match="connect"):
        annotated(relations=[relation(subject="HGNC:UNCONNECTED", target="GO:OTHER")])


def test_roles_missing_confidence_oov_and_train_only_vocabulary():
    a = annotated(relations=[relation(), relation(target="HGNC:AUDIT", role="audit"),
        relation(target="HGNC:VALIDATION", role="validation"), relation(target="HGNC:FUTURE", availability="after_measurement")])
    b = annotated("unit_1", relations=[relation(target="HGNC:UNSEEN")])
    vocab = fit_vocabulary([a, b], ["unit_0"])
    assert vocab["fields"]["object"] == ["HGNC:TEST1"]
    encoded = encode_relations([a, b], vocab)
    assert encoded["biology_mask"].tolist() == [[True], [False]]
    assert encoded["biology_oov_count"].tolist() == [0, 1]
    assert encoded["biology_support_weight"][0, 0] == 1.
    assert encoded["biology_confidence"][0, 0] == 0.
    assert not encoded["biology_confidence_mask"].any()
    strict = fit_vocabulary([a, b], ["unit_0"], "supplied_confidence_only")
    assert not encode_relations([a, b], strict)["biology_mask"].any()
    assert all(not values for values in strict["fields"].values())
    supplied = annotated(relations=[relation(confidence=.4)])
    enc = encode_relations([supplied], vocab)
    assert enc["biology_support_weight"][0, 0] == .4
    assert enc["biology_confidence"][0, 0] == .4 and enc["biology_confidence_mask"][0, 0]
    with pytest.raises(ValueError, match="independent validation"):
        annotated(relations=[relation(), relation(role="validation")])
    with pytest.raises(ValueError, match="independent validation"):
        annotated(relations=[relation(subject="FIXTURE:unit_0"), relation(role="validation")])
    with pytest.raises(ValueError, match="identities"):
        validate_vocabulary(dict(vocab, train_ids="unit_0"))


def test_dedup_retains_provenance_but_does_not_multiply_evidence():
    a = annotated(relations=[relation(confidence=.4), relation(confidence=.4, provenance=provenance("alias-source"))])
    vocab = fit_vocabulary([a], [a.unit_id])
    enc = encode_relations([a], vocab)
    assert len(a.relations) == 2 and enc["biology_mask"].sum() == 1
    assert enc["biology_support_weight"][0, 0] == .4
    with pytest.raises(ValueError, match="Duplicate"):
        annotated(relations=[relation(), relation()])


@pytest.mark.parametrize("bridge_kwargs", [{"availability": "after_measurement"},
    {"availability": "unknown"}, {"confidence": 0.}, {"evidence_family": "unknown"},
    {"evidence_code": "ND", "evidence_family": "curatorial", "confidence": 1.}])
def test_unavailable_or_nonpositive_bridge_cannot_unlock_pathway(bridge_kwargs):
    a = annotated(relations=[relation(**bridge_kwargs),
        relation(subject="HGNC:TEST1", target="GO:PATHWAY", predicate="participates_in", direction="membership")])
    vocab = fit_vocabulary([a], [a.unit_id])
    assert all(not values for values in vocab["fields"].values())
    enc = encode_relations([a], vocab)
    assert enc["biology_mask"].sum() == 0
    assert enc["biology_unreachable_count"].tolist() == [1]


def test_oov_bridge_cannot_unlock_known_pathway_and_strict_policy_applies_to_paths():
    downstream = relation(subject="HGNC:TEST1", target="GO:PATHWAY", predicate="participates_in",
                          direction="membership", confidence=.5)
    train = annotated(relations=[relation(confidence=.5), downstream])
    heldout = annotated("unit_1", relations=[relation(predicate="OOV_association", confidence=.5), downstream])
    vocab = fit_vocabulary([train, heldout], [train.unit_id])
    enc = encode_relations([heldout], vocab)
    assert not enc["biology_mask"].any()
    assert enc["biology_oov_count"].tolist() == [1]
    assert enc["biology_unreachable_count"].tolist() == [1]
    strict = annotated(relations=[relation(), downstream])
    vocab = fit_vocabulary([strict], [strict.unit_id], "supplied_confidence_only")
    assert not encode_relations([strict], vocab)["biology_mask"].any()


def test_distinct_evidence_channels_share_one_semantic_support_budget():
    experiment = relation(evidence_family="experimental", evidence_code="IDA")
    automated = relation(evidence_family="automatic", evidence_code="IEA")
    second = relation(target="HGNC:TEST2", confidence=.5)
    a, b = annotated(relations=[experiment, second]), annotated("unit_1", relations=[experiment, automated, second])
    vocab = fit_vocabulary([a, b], [a.unit_id, b.unit_id])
    enc = encode_relations([a, b], vocab)
    assert enc["biology_mask"].sum(1).tolist() == [2, 3]
    for i in range(2):
        for group in np.unique(enc["biology_group_ids"][i][enc["biology_mask"][i]]):
            selection = enc["biology_mask"][i] & (enc["biology_group_ids"][i] == group)
            assert enc["biology_support_weight"][i][selection].sum() <= 1.
    # Hold the learned channel representation constant to isolate aggregation:
    # extra evidence channels must neither amplify nor attenuate semantic edges.
    prior = EvidenceAwareMechanismPrior(vocab, 8, 3)
    with torch.no_grad():
        prior.relation[0].weight.zero_()
        prior.relation[0].bias.fill_(1.)
    mean, var = prior(tensors(enc), torch.zeros(2, 3), torch.ones(2, 3))
    torch.testing.assert_close(mean[0], mean[1], rtol=0, atol=0)
    torch.testing.assert_close(var[0], var[1], rtol=0, atol=0)


def test_metadata_roundtrip_subset_scaler_and_episode_padding(tmp_path):
    ds = dataset_with_annotations()
    a = replace(ds.biology_records[0], measurement_metadata={str(ds.well_ids[0, 0]): {
        "cell_count": TypedValue("quantity", "unknown", unit="cells", availability="after_measurement"),
        "plate": value("category", "fixture-plate")}})
    b = annotated("unit_1", relations=[relation(), relation(target="HGNC:TEST2")])
    ds = replace(ds, biology_records=(a, b, *ds.biology_records[2:]))
    source = tmp_path / "biology.json"
    save_biology(ds.biology_records, source)
    loaded = load_biology(source)
    assert loaded == ds.biology_records
    assert loaded[0].perturbation["dose"].interpretation == "nominal_protocol"
    with pytest.raises(FileExistsError):
        save_biology(loaded, source)
    path, _ = save_dataset(ds, tmp_path / "data")
    rebuilt = load_dataset(path)
    assert rebuilt.biology_records == ds.biology_records
    assert rebuilt.subset([3, 0]).biology_records == (ds.biology_records[3], a)
    with pytest.raises(ValueError, match="physical well"):
        validate_records((replace(a, measurement_metadata={"wrong-well": {}}), *ds.biology_records[1:]), ds.ids, ds.well_ids)
    scaler = TrainScaler.fit(ds, np.arange(8), biology_enabled=True)
    scaler.save(tmp_path / "scaler.json")
    normalized = TrainScaler.load(tmp_path / "scaler.json").transform(ds)
    assert normalized.biology_vocabulary["train_ids"] == ds.ids[:8].tolist()
    batch = collate_episodes([make_episode(normalized, 0, (0,), (1, 2)),
                              make_episode(normalized, 1, (0, 1), (2, 3, 4))])
    assert batch["inputs"]["biology_tokens"].shape == (2, 2, len(TOKEN_FIELDS))
    assert batch["inputs"]["biology_mask"].sum(1).tolist() == [1, 2]
    assert batch["inputs"]["biology_oov_count"].shape == (2,)
    assert batch["target_y"].shape == (2, 3, 4)


def test_default_off_never_tensorizes_annotations_or_future_qc():
    ds = dataset_with_annotations()
    scaled = TrainScaler.fit(ds, np.arange(8)).transform(ds)
    batch = make_inference_batch(scaled, [0, 1], (0,), (1, 2))
    assert not any(key.startswith("biology_") for key in batch)
    ds2 = replace(ds, biology_records=tuple(replace(r, measurement_metadata={str(ds.well_ids[i, 1]): {
        "cell_count": value("quantity", 987654321, unit="cells", interpretation="observed", role="audit")}})
        for i, r in enumerate(ds.biology_records)))
    scaler = TrainScaler.fit(ds, np.arange(8), biology_enabled=True)
    a, b = scaler.transform(ds), scaler.transform(ds2)
    before = make_inference_batch(a, [0, 1], (0,), (1, 2))
    after = make_inference_batch(b, [0, 1], (0,), (1, 2))
    for key in before:
        assert torch.equal(before[key], after[key]), key


def test_prior_is_operational_differentiable_permutation_invariant_and_safe_for_oov():
    a = annotated(relations=[relation(), relation(target="HGNC:TEST2", direction="activation", confidence=.5)])
    vocab = fit_vocabulary([a], [a.unit_id])
    batch = tensors(encode_relations([a, BiologyRecord("unknown")], vocab))
    torch.manual_seed(123)
    prior = EvidenceAwareMechanismPrior(vocab, 8, 3)
    mean, var = torch.zeros(2, 3), torch.ones(2, 3)
    m, v = prior(batch, mean, var)
    assert not torch.equal(m[0], mean[0]) and not torch.equal(v[0], var[0])
    assert torch.equal(m[1], mean[1]) and torch.equal(v[1], var[1])
    flipped = {key: array.flip(1) if array.ndim >= 2 else array for key, array in batch.items()}
    m2, v2 = prior(flipped, mean, var)
    torch.testing.assert_close(m, m2, rtol=0, atol=0)
    torch.testing.assert_close(v, v2, rtol=0, atol=0)
    (m.square().sum() + v.square().sum()).backward()
    assert prior.mean.weight.grad.abs().sum() > 0
    assert prior.embeddings[1].weight.grad.abs().sum() > 0
    masked = {key: array.clone() for key, array in batch.items()}
    masked["biology_tokens"][1] = 999999
    m3, v3 = prior(masked, mean, var)
    assert torch.equal(m3, m) and torch.equal(v3, v)
    masked["biology_mask"][1, 0] = True
    masked["biology_support_weight"][1, 0] = 1.
    with pytest.raises(ValueError, match="Unknown"):
        prior(masked, mean, var)


def test_unknown_prior_preserves_original_model_rng_weights_and_predictions():
    torch.set_num_threads(1)
    ds = measurement_fixture()
    scaler = TrainScaler.fit(ds, np.arange(8), biology_enabled=True)
    normalized = scaler.transform(ds)
    off = configuration()
    on = configuration(use_biology_prior=True)
    torch.manual_seed(861)
    base = MeasurementWorldModel(**model_kwargs(ds, off)).eval()
    rng_after_base = torch.get_rng_state().clone()
    torch.manual_seed(861)
    enabled = MeasurementWorldModel(**model_kwargs(normalized, on)).eval()
    assert torch.equal(torch.get_rng_state(), rng_after_base)
    for name, weight in base.state_dict().items():
        assert torch.equal(weight, enabled.state_dict()[name]), name
    inputs, _, _ = fixed_batch(normalized, [0, 1], on)
    x, y = base(inputs), enabled(inputs)
    for name in ("mean", "diag_var", "factors"):
        assert torch.equal(getattr(x, name), getattr(y, name)), name
    assert not any(name.startswith("biological_prior") for name in base.state_dict())
    assert "use_biology_prior" not in base.config


def test_known_relations_change_actual_world_model_prior_and_output():
    ds = dataset_with_annotations()
    cfg = configuration(use_biology_prior=True)
    scaler = TrainScaler.fit(ds, np.arange(8), biology_enabled=True)
    normalized = scaler.transform(ds)
    model = MeasurementWorldModel(**model_kwargs(normalized, cfg)).eval()
    batch, _, _ = fixed_batch(normalized, [0, 1], cfg)
    populated = model(batch)
    empty = dict(batch, biology_mask=torch.zeros_like(batch["biology_mask"]))
    without = model(empty)
    assert not torch.equal(populated.mean, without.mean)
    assert populated.mean.shape[-1] == ds.Y.shape[-1]
    (populated.mean.square().mean() + populated.diag_var.mean()).backward()
    assert model.biological_prior.mean.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("biology", [False, True])
def test_checkpoint_compatibility_without_any_training(tmp_path, biology):
    ds = dataset_with_annotations()
    cfg = configuration(use_biology_prior=biology)
    scaler = TrainScaler.fit(ds, np.arange(8), biology_enabled=biology)
    normalized = scaler.transform(ds)
    kwargs = model_kwargs(normalized, cfg)
    model = MeasurementWorldModel(**kwargs).eval()
    model.set_outcome_transform(scaler.y_center, scaler.y_scale)
    batch, _, _ = fixed_batch(normalized, [0, 1], cfg)
    before = model(batch)
    train_config = asdict(cfg)
    if not biology:
        # A checkpoint written before the optional interface existed.
        train_config.pop("use_biology_prior")
        train_config.pop("biology_evidence_weight_policy")
    payload = {"state_dict": model.state_dict(), "model_config": kwargs, "train_config": train_config,
               "train_ids": ds.ids[:8].tolist(), "validation_ids": ds.ids[8:10].tolist(),
               "feature_names": ds.feature_names.tolist()}
    torch.save(payload, tmp_path / "best.pt")
    scaler.save(tmp_path / "scaler.json")
    if not biology:
        old = json.loads((tmp_path / "scaler.json").read_text())
        old.pop("biology_vocabulary")
        (tmp_path / "scaler.json").write_text(json.dumps(old))
    loaded, restored_scaler, _, _ = load_model(tmp_path)
    after = loaded(fixed_batch(restored_scaler.transform(ds), [0, 1], cfg)[0])
    for name in ("mean", "diag_var", "factors"):
        assert torch.equal(getattr(before, name), getattr(after, name)), name
    if biology:
        payload["train_config"]["biology_evidence_weight_policy"] = "supplied_confidence_only"
        torch.save(payload, tmp_path / "best.pt")
        with pytest.raises(ValueError, match="policy"):
            load_model(tmp_path)


def test_attach_cli_writes_new_export_only(tmp_path, capsys):
    from opal2.cli import main
    ds = dataset_with_annotations()
    source, _ = save_dataset(replace(ds, biology_records=None), tmp_path / "source")
    original_bytes = source.read_bytes()
    sidecar = tmp_path / "annotations.json"
    save_biology(ds.biology_records, sidecar)
    dest = tmp_path / "attached.npz"
    main(["attach-biology", "--dataset", str(source), "--annotations", str(sidecar), "--output", str(dest)])
    status = json.loads(capsys.readouterr().out)
    assert status["training_started"] is False and status["fitted_vocabulary"] is False
    result = load_dataset(dest)
    assert result.biology_records == ds.biology_records
    assert result.biology_vocabulary is None
    np.testing.assert_array_equal(result.Y, ds.Y)
    assert source.read_bytes() == original_bytes
    with pytest.raises(FileExistsError):
        main(["attach-biology", "--dataset", str(source), "--annotations", str(sidecar), "--output", str(dest)])
