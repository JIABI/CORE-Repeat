"""Identity, cache and direct-calibration checks on synthetic arrays only."""
import json

import joblib
import numpy as np
import pytest

from scripts import run_lincs_r2_completion_20260918 as runner
from opal2.eu_r2_direct_baselines import fit_direct_baselines


def test_cell_roles_exclude_chemical_aliases_and_preserve_outer_membership():
    ids = np.array(list("abcdefghi"))
    groups = ids.copy()
    record = dict(fit_ids=list("abc"), inner_validation_ids=list("de"), test_ids=list("fghi"))
    cell = dict(fit_ids=["f"], calibration_ids=["g"], query_ids=list("hi"), budget=1)
    parts = runner.validate_cell(ids, groups, record, cell)
    np.testing.assert_array_equal(parts["DEV_EVAL"], [7, 8])
    poisoned = groups.copy()
    poisoned[7] = poisoned[5]
    with pytest.raises(ValueError, match="Chemical identity overlap"):
        runner.validate_cell(ids, poisoned, record, cell)
    with pytest.raises(ValueError, match="outer test membership"):
        runner.validate_cell(ids, groups, record, dict(cell, query_ids=["h"]))


def test_fixed_cell_budget_retains_remainder_and_deterministic_ids():
    ids = np.array(["c", "a", "b", "d"])
    chosen = runner.select_budget(ids, np.zeros(4), np.zeros(4), .2, 2)
    np.testing.assert_array_equal(chosen, [False, True, True, False])
    # Old 118-object half has k=15; recomputing floor(.25*N)/2 would lose one.
    ids = np.asarray([f"id{i:03d}" for i in range(118)])
    assert runner.select_budget(ids, np.zeros(118), np.zeros(118), 0., 15).sum() == 15


def test_shared_train_estimator_and_cal_only_refit_match_original_full_recipe(tmp_path, monkeypatch):
    rng = np.random.default_rng(119)
    x = rng.normal(size=(69, 6))
    y = np.clip(.08*x[:, 0]-.06*x[:, 1]+rng.normal(0, .025, len(x)), -.5, .5)
    ids = np.array([f"i{i:03d}" for i in range(len(x))])
    t, v, c, q = np.arange(35), np.arange(35, 47), np.arange(47, 59), np.arange(59, 69)
    selected = runner.fit_selected(x, y, t, v, "RIDGE", tmp_path, ids, ids, 91, "TRAIN_MATCHED")
    separate = runner.calibrate_selected(selected, x[c], y[c], x[q], 91)
    complete = fit_direct_baselines(x[t], y[t], x[v], y[v], x[c], y[c], x[q], seed=91)["RIDGE"]
    for key in ("predicted", "p_null", "p_null_calibrated", "gamma_residuals",
                "gamma_distribution_mean", "p_null_from_gamma"):
        np.testing.assert_allclose(separate[key], complete[key], atol=1e-14, rtol=1e-14)
    def no_fit(*args, **kwargs):
        raise AssertionError("Cached estimator must not be fitted twice")
    monkeypatch.setattr(runner, "_select", no_fit)
    reused = runner.fit_selected(x, y, t, v, "RIDGE", tmp_path, ids, ids, 91, "TRAIN_MATCHED")
    altered = runner.calibrate_selected(reused, x[c], -y[c], x[q], 91)
    np.testing.assert_array_equal(altered["predicted"], separate["predicted"])
    np.testing.assert_array_equal(altered["p_null"], separate["p_null"])
    assert not np.array_equal(altered["gamma_residuals"], separate["gamma_residuals"])
    with pytest.raises(ValueError, match="provenance changed"):
        runner.fit_selected(x, y, t[:-1], v, "RIDGE", tmp_path, ids, ids, 91, "TRAIN_MATCHED")


def test_atomic_cache_roundtrip_and_block_bootstrap(tmp_path):
    path = tmp_path/"values.npz"
    runner.save_npz(path, ids=np.array(["a", "b"]), values=np.array([1., 2.]))
    np.testing.assert_array_equal(runner.read_npz(path)["values"], [1., 2.])
    assert not (tmp_path/"values.tmp.npz").exists()
    path = tmp_path/"model.joblib"
    runner.save_model(path, {"setting": 1})
    assert joblib.load(path) == {"setting": 1}
    mean, interval = runner.paired_intervals(np.array([[2., 0.], [2., 0.], [2., 0.]]),
        np.array(["same", "same", "other"]), replicates=100)
    np.testing.assert_array_equal(mean, [2., 0.])
    np.testing.assert_array_equal(interval, [[2., 2.], [0., 0.]])


def test_resource_and_control_budget_accounting():
    cell = dict(counts=dict(TRAIN=700, VALIDATION=190, REF_FIT=79, DIST_CAL=40, DEV_EVAL=118), budget=15)
    train = runner.resource_counts(cell, "DIRECT_TRAIN_MATCHED_RIDGE_COHERENT")
    access = runner.resource_counts(cell, "DIRECT_ACCESS_MATCHED_RIDGE_COHERENT")
    assert train["reference_all_new_wells"] == 0
    assert train["calibration_all_new_wells"] == 160
    assert access["reference_all_new_wells"] == 316
    assert access["combined_setup_all_new_wells"] == 476
    assert runner.resource_counts(cell, "HR_REF_AMP_GAUSSIAN")["calibration_all_new_wells"] == 0
    assert runner.resource_counts(cell, "DIRECT_TRAIN_MATCHED_RIDGE_CLASSIFIER_RAW")["calibration_all_new_wells"] == 0
    assert train["bank_anchors_already_in_backbone"] == 64
    constant = runner.resource_counts(cell, "CONSTANT_ACCESS_MATCHED")
    assert constant["calibration_objects"] == 0
    ids = np.asarray([f"x{i}" for i in range(8)])
    output = runner.fixed_controls(ids, np.ones((8, 3)), np.ones(8)*.1,
        np.repeat([0, 1], 4), [dict(budget=1), dict(budget=2)])
    assert output["same_budget"]["ID_FIXED"]["activated"] == 3
    assert output["random_same_budget"]["fixed_k"] == 3
    assert output["random_same_budget"]["total_value_mean"] == pytest.approx(.3)


def test_protocol_and_real_joint_arms_are_distinct():
    assert len(runner.JOINT_ARMS) == 12
    assert len(runner.ARMS) == 31 and len(set(runner.ARMS)) == 31
    assert runner.SAMPLES == 100000 and runner.THREADS <= 2


def test_all_arm_aggregation_preserves_identity_policy_and_scalar_scope(tmp_path, monkeypatch):
    root, report = tmp_path/"runs", tmp_path/"reports"
    root.mkdir(); report.mkdir()
    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.setattr(runner, "REPORT", report)
    ids = np.array([f"i{i}" for i in range(8)])
    actual = np.linspace(-.15, .2, 8)
    # Noncanonical query order exercises identity joining of main and MC arrays.
    cell_rows = [np.array([2, 0, 3, 1]), np.array([7, 5, 4, 6])]
    cells = []
    for h, q in enumerate(cell_rows):
        folder = root/f"cell_0_{h}"
        folder.mkdir()
        cell = dict(fold=0, half=h, budget=1, ids={"DEV_EVAL":ids[q].tolist()},
            counts=dict(TRAIN=8, VALIDATION=2, REF_FIT=2, DIST_CAL=2, DEV_EVAL=4))
        cells.append(cell)
        for name in runner.ARMS:
            out = dict(ids=ids[q], actual=actual[q], predicted=actual[q]/2,
                p_null=np.linspace(.2, .8, 8)[q], crps=np.ones(4)*.1,
                gamma_coverage_by_level=np.ones((4,5),bool))
            if name in runner.JOINT_ARMS:
                out.update(mean_u=np.zeros((4,9)), actual_u=np.ones((4,9)),
                    nll=np.ones(4), energy=np.ones(4), joint_coverage_by_level=np.ones((4,5)))
                for offset in runner.OFFSETS:
                    runner.save_npz(folder/f"{name}_mc{offset}.npz", ids=ids[q],
                                    predicted=out["predicted"], p_null=out["p_null"])
            runner.save_npz(folder/(name+".npz"), **out)
        (folder/"complete.json").write_text(json.dumps(dict(complete=True)))
    data = dict(ids=ids, groups=ids.copy(), Y=np.ones((8,4,6)))
    metadata = dict(units=[dict(layout_block="layout"+str(i//4)) for i in range(8)])
    runner.aggregate(data, metadata, dict(cells=cells))
    result = json.loads((root/"summary.json").read_text())
    assert result["complete"] and len(result["metrics"]) == 31
    for name in runner.ARMS:
        row = runner.read_npz(root/(name+".npz"))
        np.testing.assert_array_equal(row["ids"], ids)
        assert row["selected_lambda_0.2"].sum() == 2
        assert result["selected_overlap_with_CORE"][name]["lambda_0.2"]["overlap_fraction"] == 1.
    assert "nll" not in result["metrics"][runner.DIRECT_ARMS[0]]
    assert result["metrics"]["CORE_ORIGINAL"]["full_joint_available"]
    assert "HR_REF minus RIDGE_REF" in result["paired_mean_and_law_attribution"]
    assert "absolute_gap_intervals_per_candidate" in result["calibration"]["CORE_ORIGINAL"]
    assert not any("NULL_gap" in key for key in result["paired_vs_CORE"]["CORE_ORIGINAL"])
    assert result["monte_carlo"]["CORE_ORIGINAL"][0]["policies"]["lambda_0.2"]["symmetric_difference_from_main"] == 0
