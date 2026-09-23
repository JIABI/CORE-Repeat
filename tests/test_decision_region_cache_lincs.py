"""CAL label isolation and unchanged QUERY semantics, without scientific runs."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from opal2 import decision_region_cache_lincs as adapter


def _calibration_arrays():
    rng = np.random.default_rng(916)
    n = 16
    ids = np.asarray([f"id_{i:02d}" for i in range(n)])
    groups = np.asarray([f"g{i:02d}" for i in range(n)])
    groups[1] = groups[0]
    inner = np.arange(n) % 5
    inner[1] = inner[0]
    mean = rng.normal(0, .07, (n, 9))
    scatter = np.broadcast_to(.04 * np.eye(9), (n, 9, 9)).copy()
    residual = rng.normal(0, .15, (n, 9))
    logamp = np.linspace(1., 3., n)
    stats = dict(u_scale=np.ones(9), u_center=np.zeros(9))
    return ids, groups, mean, scatter, residual, logamp, .7, stats, inner


def test_core_predictions_exclude_every_outcome_in_the_held_chemical_fold():
    arrays = _calibration_arrays()
    result = adapter.crossfit_core_radial(*arrays, samples=256, seed=27)
    changed = list(deepcopy(arrays))
    held = arrays[-1] == 0
    # Poison both aliases and every other held chemical group, not just one row.
    changed[4][held] *= 8.
    poisoned = adapter.crossfit_core_radial(*changed, samples=256, seed=27)
    np.testing.assert_array_equal(result["cal_p"][held], poisoned["cal_p"][held])
    np.testing.assert_array_equal(result["cal_mean"][held], poisoned["cal_mean"][held])
    assert np.any(result["cal_mean"][~held] != poisoned["cal_mean"][~held])
    for record in result["audit"]:
        assert not set(record["held_groups"]) & set(record["fit_groups"])
        assert not set(record["held_ids"]) & set(record["radial_representative_ids"])
        # The two aliases contribute one ID-min representative whenever in fit.
        representatives = record["radial_representative_ids"]
        assert "id_01" not in representatives
        assert len(representatives) == len(record["fit_groups"])
    json.dumps(result["audit"])


def test_core_rejects_an_alias_group_split_between_inner_folds():
    arrays = list(_calibration_arrays())
    arrays[-1] = arrays[-1].copy()
    arrays[-1][1] = 1
    with pytest.raises(ValueError, match="chemical group crosses"):
        adapter.crossfit_core_radial(*arrays, samples=8, seed=1)


class _FixedRegression:
    def predict(self, x):
        return x[:, 0]

    def fit(self, *args, **kwargs):
        raise AssertionError("No direct estimator may be fitted by cache construction")


class _FixedClassifier:
    classes_ = np.array([0, 1])

    def predict_proba(self, x):
        p = 1. / (1. + np.exp(-x[:, 1]))
        return np.column_stack((1-p, p))

    def fit(self, *args, **kwargs):
        raise AssertionError("No direct estimator may be fitted by cache construction")


def _context():
    cal_ids, cal_groups, mean, scatter, residual, logamp, bandwidth, stats, _ = _calibration_arrays()
    ids = np.r_[np.array(["train", "valid", "ref"]), cal_ids, ["query_a", "query_b"]]
    groups = np.r_[np.array(["tg", "vg", "rg"]), cal_groups, ["qa", "qb"]]
    parts = dict(TRAIN=np.array([0]), VALIDATION=np.array([1]), REF_FIT=np.array([2]),
                 DIST_CAL=np.arange(3, 19), DEV_EVAL=np.array([19, 20]))
    cell = dict(fold=0, half=1, budget=1)
    query = {name: dict(query_p=np.array([.12 + i*.01, .72-i*.01]),
                       query_mean=np.array([.08-i*.01, -.07+i*.01]))
             for i, name in enumerate(adapter.ARM_FILES)}
    return dict(cell=cell, folder=Path("/synthetic-cell"), manifest=dict(samples=100000),
        ids=ids, groups=groups, parts=parts, cal_mean_u=mean, cal_scatter=scatter,
        cal_residual=residual, cal_log_amplitude=logamp, radial_bandwidth=bandwidth,
        stats=stats, cal_x=np.column_stack((np.linspace(-.1, .2, 16), logamp)),
        cal_actual=np.linspace(-.15, .25, 16), query_actual=np.array([.1, -.03]),
        query_layout=np.array(["layout1", "layout2"]), original_query=query,
        sources=dict(manifest="synthetic"))


def _install_context(monkeypatch, context):
    monkeypatch.setattr(adapter, "_load_cell", lambda name: deepcopy(context))

    def load_model(path):
        family = Path(path).stem
        return dict(regression=_FixedRegression(), classifier=_FixedClassifier(),
            provenance=dict(scope="ACCESS_MATCHED", family=family,
                            train_ids=["train", "ref"], validation_ids=["valid"]))
    monkeypatch.setattr(adapter.joblib, "load", load_model)


def test_build_retains_saved_query_vectors_without_estimator_refit(monkeypatch):
    context = _context()
    _install_context(monkeypatch, context)
    result = adapter.build_cell("LINCS", "cell_0_1", samples=128, seed=111)
    assert set(result["arms"]) == set(adapter.ARM_FILES)
    assert result["query_budget"] == 1
    for name, values in result["arms"].items():
        for key in ("query_mean", "query_p"):
            np.testing.assert_array_equal(values[key], context["original_query"][name][key])
        assert values["cal_mean"].shape == values["cal_p"].shape == (16,)
    for group in np.unique(result["cal_groups"]):
        assert len(np.unique(result["cal_inner_fold"][result["cal_groups"] == group])) == 1
    assert result["audit"]["query_samples"] == 100000
    assert result["audit"]["cal_samples"] == 128
    assert result["audit"]["mean_refits"] == result["audit"]["direct_estimator_refits"] == 0
    json.dumps(result["audit"])


def test_all_arm_calibration_outputs_ignore_held_fold_label_mutations(monkeypatch):
    context = _context()
    _install_context(monkeypatch, context)
    first = adapter.build_cell("LINCS", "cell_0_1", samples=128, seed=111)
    held = first["cal_inner_fold"] == 0
    poisoned = deepcopy(context)
    poisoned["cal_actual"][held] = .8
    poisoned["cal_residual"][held] *= 8.
    poisoned["query_actual"][:] = -.9
    _install_context(monkeypatch, poisoned)
    second = adapter.build_cell("LINCS", "cell_0_1", samples=128, seed=111)
    for arm in adapter.ARM_FILES:
        for field in ("cal_mean", "cal_p"):
            np.testing.assert_array_equal(first["arms"][arm][field][held],
                                          second["arms"][arm][field][held])
        for field in ("query_mean", "query_p"):
            np.testing.assert_array_equal(first["arms"][arm][field], second["arms"][arm][field])


def test_original_cell_roles_reject_alias_leakage():
    ids = np.asarray(list("abcde"))
    roles = ("TRAIN", "VALIDATION", "REF_FIT", "DIST_CAL", "DEV_EVAL")
    cell = dict(ids={role: [ids[i]] for i, role in enumerate(roles)},
                counts={role: 1 for role in roles})
    adapter._roles(ids, ids.copy(), cell)
    groups = ids.copy()
    groups[-1] = groups[-2]
    with pytest.raises(ValueError, match="Chemical group crosses"):
        adapter._roles(ids, groups, cell)


@pytest.mark.parametrize("dataset,samples,seed", [("EU", 128, 1), ("LINCS", 0, 1),
                                                ("LINCS", 128, -1)])
def test_rejects_invalid_scope_or_integration_budget(dataset, samples, seed):
    with pytest.raises(ValueError):
        adapter.build_cell(dataset, "cell_0_0", samples=samples, seed=seed)
