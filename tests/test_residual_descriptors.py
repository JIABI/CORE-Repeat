"""Data-lineage and numerical tests; synthetic inputs do not test efficacy."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from opal2.residual_descriptors import (
    CONTROL_NAMES, ResidualDescriptorTransformer, information_availability,
    load_normalized_plate_controls, normalized_control_summary,
)


def fixture_data():
    rng = np.random.default_rng(871)
    names = [f"{comp}_{family}_{j}" for comp in ("Cells", "Cytoplasm", "Nuclei")
             for family in ("Intensity_MADIntensity", "RadialDistribution_RadialCV", "Texture_Variance")
             for j in range(2)]
    state = rng.normal(size=(24, len(names)))
    y = state[:, None, :]+rng.normal(size=(24, 4, len(names)))*.3
    ids = np.asarray([f"i{i}" for i in range(24)])
    data = dict(Y=y, ids=ids, groups=np.asarray([f"g{i//2}" for i in range(24)]),
                feature_names=np.asarray(names))
    units = [dict(id=str(i), actual_dose_uM=10., exposure_hours_protocol_nominal=48., cell_line="A549",
                  roles={role:dict(cell_count=200+j*5+(role!="X")*10000, plate=f"p{j//6}", well="B07")
                         for role in ("X", "Z1", "Z2", "V")}) for j, i in enumerate(ids)]
    controls = {f"p{k}": normalized_control_summary(rng.normal(size=(24, len(names)))) for k in range(4)}
    return data, dict(units=units), controls


def fit_fixture(data=None, meta=None, controls=None):
    if data is None:
        data, meta, controls = fixture_data()
    return ResidualDescriptorTransformer.fit(data, meta, np.arange(16), plate_controls=controls,
                                             pca_dim=8, latent_pca_dim=5, seed=15)


def test_shapes_finite_schema_and_interpretation():
    data, meta, controls = fixture_data()
    fitted = fit_fixture(data, meta, controls)
    out = fitted.transform(data, meta)
    assert out.values.shape == (24, len(out.names))
    assert out.latent_input.shape == (24, 6)
    assert np.isfinite(out.values).all()
    assert sorted(sum(out.blocks.values(), [])) == list(range(len(out.names)))
    assert out.select_blocks("amplitude", "cell_count").shape == (24, 5)
    assert out.select_blocks(["amplitude", "cell_count"]).shape == (24, 5)
    assert fitted.fit_ids == data["ids"][:16].tolist()
    assert "not cell-to-cell" in out.report["availability"]["heterogeneity_interpretation"]
    assert out.report["availability"]["potency_annotated_objects"] == 0
    assert out.report["availability"]["normalized_dmso_query_coverage"] == 24
    assert "not independence" in out.report["ratio_interpretation"]


def test_query_future_profiles_never_enter_fit_or_transform():
    data, meta, controls = fixture_data()
    first = fit_fixture(data, meta, controls)
    changed = deepcopy(data)
    changed["Y"][16:, 1:] = np.nan
    second = fit_fixture(changed, meta, controls)
    a, b = first.transform(data, meta), second.transform(changed, meta)
    np.testing.assert_array_equal(a.values, b.values)
    np.testing.assert_array_equal(a.latent_input, b.latent_input)
    np.testing.assert_array_equal(first.reliability_vectors, second.reliability_vectors)
    # Even every fit future profile is irrelevant after fitting has finished.
    changed["Y"][:, 1:] = np.nan
    np.testing.assert_array_equal(a.values, first.transform(changed, meta).values)


def test_query_first_profiles_do_not_fit_preprocessing():
    data, meta, controls = fixture_data()
    a = fit_fixture(data, meta, controls)
    changed = deepcopy(data)
    changed["Y"][16:, 0] *= 1000
    b = fit_fixture(changed, meta, controls)
    for key in ("x_center", "x_scale", "raw_center", "raw_scale", "reliability_values", "reliability_vectors"):
        np.testing.assert_array_equal(getattr(a, key), getattr(b, key))
    np.testing.assert_array_equal(a.pca.components_, b.pca.components_)
    np.testing.assert_array_equal(a.latent_pca.components_, b.latent_pca.components_)


def test_reliability_ratios_are_scale_invariant_not_declared_independent():
    data, meta, controls = fixture_data()
    model = fit_fixture(data, meta, controls)
    first = model.transform(data, meta)
    changed = deepcopy(data)
    changed["Y"][:, 0] *= np.linspace(.2, 7., len(data["ids"]))[:, None]
    second = model.transform(changed, meta)
    np.testing.assert_allclose(first.raw_values[:, first.blocks["reliability"]],
                               second.raw_values[:, second.blocks["reliability"]], atol=2e-14)
    assert not np.allclose(first.values[:, 0], second.values[:, 0])


def test_latent_direction_pca_preserves_per_object_scale_invariance():
    data, meta, controls = fixture_data()
    model = fit_fixture(data, meta, controls)
    first = model.transform(data, meta)
    changed = deepcopy(data)
    scales = np.linspace(.1, 17., len(data["ids"]))
    changed["Y"][:, 0] *= scales[:, None]
    second = model.transform(changed, meta)
    np.testing.assert_allclose(first.latent_input[:, :-1], second.latent_input[:, :-1], atol=2e-14)
    np.testing.assert_allclose(second.latent_input[:, -1]-first.latent_input[:, -1],
                               np.log(scales)/model.latent_amp_scale, atol=2e-14)
    assert model.latent_pca is not model.pca


def test_only_x_count_conditions_and_plate_are_used():
    data, meta, controls = fixture_data()
    model = fit_fixture(data, meta, controls)
    a = model.transform(data, meta)
    changed = deepcopy(meta)
    for unit in changed["units"]:
        for role in ("Z1", "Z2", "V"):
            unit["roles"][role].update(cell_count=-100, plate="never_observed", well="A01")
    b = model.transform(data, changed)
    np.testing.assert_array_equal(a.values, b.values)
    rawcount = a.raw_values[:, a.names.index("log_cell_count_X")]
    np.testing.assert_allclose(rawcount, np.log([200+j*5 for j in range(24)]))
    changed["units"][22]["roles"]["X"]["cell_count"] = None
    c = model.transform(data, changed)
    assert c.raw_values[22, c.names.index("cell_count_missing")] == 1
    assert np.isfinite(c.values).all()


def test_missing_controls_unknown_conditions_and_schema():
    data, meta, _ = fixture_data()
    model = fit_fixture(data, meta, {})
    changed = deepcopy(meta)
    changed["units"][20]["cell_line"] = "unseen_cell_line"
    changed["units"][20]["actual_dose_uM"] = None
    result = model.transform(data, changed)
    assert result.raw_values[20, result.names.index("cell_background_unseen")] == 1
    assert result.raw_values[20, result.names.index("dose_missing")] == 1
    assert result.raw_values[:, result.names.index("normalized_dmso_unavailable")].all()
    bad = deepcopy(data); bad["feature_names"] = bad["feature_names"][::-1]
    with pytest.raises(ValueError, match="Feature order"):
        model.transform(bad, meta)


def test_group_self_exclusion_and_fit_outcomes_used_only_for_reliability():
    data, meta, controls = fixture_data()
    a = fit_fixture(data, meta, controls)
    transformed = a.transform(data, meta)
    nearest = transformed.raw_values[:, transformed.names.index("log1p_nearest_fit_distance")]
    assert (nearest>0).all()
    changed = deepcopy(data)
    changed["Y"][:16, 1:] = np.random.default_rng(31).normal(size=changed["Y"][:16, 1:].shape)
    b = fit_fixture(changed, meta, controls)
    assert not np.allclose(a.reliability_values, b.reliability_values)
    np.testing.assert_array_equal(a.pca.components_, b.pca.components_)
    assert "same-chemical-group" in a.report["learned_scaling_scope"]


def test_control_reader_only_reads_x_plates_and_dmso(tmp_path):
    data, meta, _ = fixture_data()
    names = data["feature_names"].tolist()
    rng = np.random.default_rng(189)
    for plate in {u["roles"]["X"]["plate"] for u in meta["units"]}:
        frame = pd.DataFrame(rng.normal(size=(7, len(names))), columns=names)
        frame["Metadata_broad_sample"] = ["DMSO"]*6+["DRUG"]
        frame.to_csv(tmp_path/f"{plate}_normalized_dmso.csv.gz", index=False)
    for u in meta["units"]:
        u["roles"]["V"]["plate"] = "future_only"
    (tmp_path/"future_only_normalized_dmso.csv.gz").write_bytes(b"MUST NOT READ")
    controls = load_normalized_plate_controls(meta, names, tmp_path)
    assert len(controls) == 4
    assert all(v.shape == (len(CONTROL_NAMES),) for v in controls.values())
    assert all(v[0] == np.log1p(6) for v in controls.values())
    availability = information_availability(data, meta, controls)
    assert availability["future_plate_context_used"] is False


def test_potency_metadata_does_not_infer_mechanism_from_target_identity():
    data, meta, controls = fixture_data()
    meta["units"][0]["chemistry"] = {"target_set":["EGFR"], "name":"gefitinib"}
    report = information_availability(data, meta, controls)
    assert report["potency_annotated_objects"] == 0
    meta["units"][0]["chemistry"]["EC50"] = 2.
    report = information_availability(data, meta, controls)
    assert report["potency_annotated_objects"] == 1
    assert report["matched_morphology_potency_verified"] is False
