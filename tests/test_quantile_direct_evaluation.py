import numpy as np
import pytest

from opal2.quantile_direct_evaluation import (
    ARMS, _attach_historical, aggregate, dataset_analysis,
    paired_block_statistics, validate_cell,
)


def make_cell(prefix="a", n=8, k=2):
    actual = np.linspace(-.12, .3, n)
    out = dict(ids=np.array([prefix+str(i) for i in range(n)]),
               groups=np.array(["g"+str(i//2) for i in range(n)]),
               layout=np.array(["L"+str(i % 3) for i in range(n)]),
               actual=actual, original_k=np.asarray(k))
    for arm in ARMS:
        out[arm+"_expected"] = np.linspace(0., .2, n)
        out[arm+"_p_null"] = np.linspace(.2, .5, n)
        out[arm+"_crps"] = np.full(n, .05)
        out[arm+"_coverage"] = np.ones((n, 5), bool)
        out[arm+"_width"] = np.full((n, 5), .2)
    return out


def test_validation_requires_exact_integer_quota_and_finite_complete_vectors():
    cell = make_cell()
    assert validate_cell(cell) == (8, 2)
    for invalid in (2., True, 0, 9, [2]):
        bad = dict(cell, original_k=np.asarray(invalid))
        with pytest.raises(ValueError, match="original_k"): validate_cell(bad)
    bad = dict(cell); bad["raw_p_null"] = np.full(8, 1.1)
    with pytest.raises(ValueError, match="outside"): validate_cell(bad)
    bad = dict(cell); bad["actual"] = np.full(8, np.nan)
    with pytest.raises(ValueError, match="finite"): validate_cell(bad)


def test_saved_quota_random_expectation_and_duplicate_chemical_groups_preserved():
    one = make_cell("a", 8, 2); two = make_cell("b", 10, 3)
    arrays, tables = dataset_analysis("TEST", [("cell_0", one), ("cell_1", two)], replicates=40)
    assert len(arrays["ids"]) == 18
    assert len(np.unique(arrays["groups"])) == 5  # groups shared across doses/cells
    expected_random = one["actual"].sum()*2/8 + two["actual"].sum()*3/10
    for row in tables["summary"]:
        assert row["selected"] == 5
        assert row["random_expected_total_Gamma"] == pytest.approx(expected_random)
    for contrast in tables["paired_intervals"]:
        assert contrast["difference"] == 0
        assert contrast["ci95"] == [0., 0.]
        assert contrast["clusters"] == (5 if contrast["block"] == "chemical_group" else 3)
    assert {row["lambda_value"] for row in tables["budget_curves"]} == {0., .2}
    assert len(tables["lambda0_secondary"]) == 12


def test_only_query_aligned_fields_are_concatenated():
    one = make_cell("a", 8, 2); two = make_cell("b", 10, 3)
    for cell in (one, two):
        # Eight knots accidentally match the first cell but not the second.
        cell["probability_knots"] = np.linspace(0., 1., 8)
        cell["nominal_coverage"] = np.array([.5, .8, .9, .95, .99])
        cell["raw_quantiles"] = np.zeros((len(cell["actual"]), 8))
    arrays, _ = dataset_analysis("TEST", [("cell_0", one), ("cell_1", two)], replicates=20)
    assert "probability_knots" not in arrays
    assert "nominal_coverage" not in arrays
    assert arrays["raw_quantiles"].shape == (18, 8)
    assert all(len(value) == 18 for value in arrays.values())


def test_ids_determine_ties_and_query_outcomes_do_not_change_any_selection():
    cell = make_cell()
    for arm in ARMS:
        cell[arm+"_expected"] = np.ones(8)
        cell[arm+"_p_null"] = np.full(8, .1)
    arrays, _ = dataset_analysis("TEST", [("cell_0", cell)], replicates=20)
    changed = dict(cell, actual=-cell["actual"])
    after, _ = dataset_analysis("TEST", [("cell_0", changed)], replicates=20)
    for arm in ARMS:
        np.testing.assert_array_equal(np.flatnonzero(arrays[arm+"_selected"]), [0, 1])
        np.testing.assert_array_equal(arrays[arm+"_selected"], after[arm+"_selected"])
        np.testing.assert_array_equal(arrays[arm+"_selected_lambda0"], after[arm+"_selected_lambda0"])


def test_shared_group_bootstrap_is_unchanged_by_repeated_dose_rows():
    values = np.array([[1., 2.], [3., 4.], [1., 2.], [3., 4.]])
    den = np.ones_like(values)
    duplicated = paired_block_statistics(values, den, ["a", "b", "a", "b"], replicates=100)
    original = paired_block_statistics(values[:2], den[:2], ["a", "b"], replicates=100)
    assert duplicated == original


def test_zero_selected_denominators_are_reported_and_one_layout_has_no_interval():
    values = np.array([[1.], [0.]])
    result = paired_block_statistics(values, values, ["a", "b"], replicates=200)[0]
    assert result["invalid_zero_denominator_replicates"] > 0
    assert result["valid_replicates"] + result["invalid_zero_denominator_replicates"] == 200
    single = paired_block_statistics(values, values, ["a", "a"], replicates=20)[0]
    assert single["ci95"] is None


def test_historical_prediction_or_selection_changes_are_rejected():
    cell = make_cell()
    historical = {}
    for arm in ("core", "histgb"):
        historical[arm] = dict(ids=cell["ids"], actual=cell["actual"],
            predicted=cell[arm+"_expected"], p_null=cell[arm+"_p_null"], crps=cell[arm+"_crps"],
            selected_lambda_0_2=None)
        historical[arm]["selected_lambda_0.2"] = np.array([False]*6+[True]*2)
    _attach_historical(cell, historical, "cell_0")
    changed = dict(cell, core_expected=cell["core_expected"]+.01)
    with pytest.raises(AssertionError): _attach_historical(changed, historical, "cell_0")
    changed = dict(cell, original_k=np.asarray(1))
    with pytest.raises(ValueError, match="quota"): _attach_historical(changed, historical, "cell_0")


def test_historical_lists_are_preserved_even_if_rank_reconstruction_differs():
    cell = make_cell()
    for arm in ("core", "histgb"):
        cell[arm+"_selected"] = np.array([True]*2+[False]*6)
    arrays, tables = dataset_analysis("TEST", [("cell_0", cell)], replicates=20)
    np.testing.assert_array_equal(arrays["core_selected"], cell["core_selected"])
    audit = next(row for row in tables["selection_audit"] if row["arm"] == "CORE_ORIGINAL" and row["lambda_value"] == .2)
    assert audit["reconstruction_mismatches"] == 4
    assert audit["original_saved_list_reused"]


def test_incomplete_runs_never_claim_all_sixty_cells_complete(tmp_path):
    run = tmp_path/"runs"/"trial"
    status = aggregate(tmp_path, run)
    assert not status["complete"]
    assert status["expected_cells"] == 60
    assert status["pending_datasets"] == ["EU", "JUMP", "LINCS", "RxRx3"]
