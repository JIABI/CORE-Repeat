"""Exact numerical checks on calibration mathematics and split discipline."""
import numpy as np
import pytest

from opal2.calibration import (RiskContract, SplitConformalUtility, assert_disjoint,
                               bernoulli_confidence_sequence, bounded_mean_confidence_sequence,
                               bounded_mean_interval, clopper_pearson, evaluate_contract)


def test_conformal_finite_sample_order_statistic_and_tiny_sample_infinity():
    ids = [f"c{i}" for i in range(9)]
    model = SplitConformalUtility.fit(np.arange(1., 10.), np.zeros(9), ids, alpha=.2)
    assert model.radius == 8  # ceil((9+1)*.8) = 8, not an interpolated quantile
    lo, hi = model.interval(np.array([2.]), ["test"])
    np.testing.assert_allclose([lo[0], hi[0]], [-6., 10.])
    tiny = SplitConformalUtility.fit(np.ones(3), np.zeros(3), ["a", "b", "c"], alpha=.05)
    assert np.isinf(tiny.radius)
    lo, hi = tiny.interval(np.zeros(1), ["test"], support=(-1.02, 1.))
    np.testing.assert_allclose([lo[0], hi[0]], [-1.02, 1.])


def test_conformal_simultaneous_fixed_action_collection():
    y = np.array([[1., 4.], [2., 6.], [3., 8.], [4., 10.]])
    scale = np.array([[1., 2.]] * 4)
    model = SplitConformalUtility.fit(y, np.zeros_like(y), list("abcd"), alpha=.4, predicted_scale=scale)
    np.testing.assert_allclose(model.calibration_scores, [2, 3, 4, 5])
    assert model.radius == 4
    lo, hi = model.interval(np.zeros((1, 2)), ["test"], predicted_scale=np.array([[1., 2.]]))
    np.testing.assert_allclose(hi, [[4., 8.]])
    np.testing.assert_allclose(lo, [[-4., -8.]])


def test_id_overlap_duplicate_ids_and_unseen_output_shape_rejected():
    with pytest.raises(ValueError, match="overlap"):
        assert_disjoint(["x"], ["x"])
    with pytest.raises(ValueError, match="unique"):
        SplitConformalUtility.fit(np.ones(2), np.zeros(2), ["x", "x"])
    with pytest.raises(ValueError, match="overlap"):
        SplitConformalUtility.fit(np.ones(2), np.zeros(2), ["x", "y"], training_ids=["x"])
    model = SplitConformalUtility.fit(np.ones(2), np.zeros(2), ["x", "y"])
    with pytest.raises(ValueError, match="overlap"):
        model.interval(np.zeros(1), ["x"])
    with pytest.raises(ValueError, match="shape"):
        model.interval(np.zeros((1, 2)), ["new"])


def test_clopper_pearson_zero_failures_and_empty_denominator():
    lower, upper = clopper_pearson(0, 20, alpha=.05, side="upper")
    assert lower == 0
    np.testing.assert_allclose(upper, 1 - .05 ** (1 / 20), rtol=1e-13)
    np.testing.assert_allclose(clopper_pearson(20, 20, alpha=.05, side="lower")[0], .05 ** (1 / 20))
    assert clopper_pearson(0, 0) == (0., 1.)


def test_hoeffding_sequence_exact_alpha_spending_and_bounds():
    x = np.array([.2, .4, .5, .8, .2])
    result = bounded_mean_confidence_sequence(x, 0., 1., alpha=.05, assume_iid_units=True)
    t = np.arange(1, 6)
    np.testing.assert_allclose(result["alpha_spent_at_time"], .05 / (t * (t+1)))
    np.testing.assert_allclose(result["alpha_spent_at_time"].sum(), .05 * (1 - 1/6))
    expected_radius = np.sqrt(np.log(2 * t * (t+1) / .05) / (2*t))
    np.testing.assert_allclose(result["lower"], np.maximum(0, np.cumsum(x)/t - expected_radius))
    np.testing.assert_allclose(result["upper"], np.minimum(1, np.cumsum(x)/t + expected_radius))
    with pytest.raises(ValueError, match="IID"):
        bounded_mean_confidence_sequence(x, 0., 1.)
    with pytest.raises(ValueError, match="bounds"):
        bounded_mean_interval(np.array([2.]), 0., 1., assume_iid_units=True)


def test_bernoulli_sequence_uses_spent_alpha_not_repeated_fixed_point_zero_five():
    result = bernoulli_confidence_sequence(np.zeros(20), assume_iid_units=True)
    expected = clopper_pearson(0, 20, alpha=.05/(20*21))[1]
    np.testing.assert_allclose(result["upper"][-1], expected)
    assert result["upper"][-1] > clopper_pearson(0, 20)[1]


def _evaluation_inputs():
    n = 40
    selected = np.zeros(n, dtype=bool); selected[:10] = True
    gains = np.full(n, .05); gains[:2] = -.01
    null = gains <= 0
    positive = gains >= .005
    return (selected, gains, null, positive, [f"e{i}" for i in range(n)])


def test_contract_development_status_no_false_certification_and_disjoint_ids():
    args = _evaluation_inputs()
    result = evaluate_contract(*args, RiskContract(min_activations=5), gain_support=(-1.02, 1.),
                               wells_if_selected=np.ones(40), calibration_ids=["cal"],
                               assume_iid_units=True)
    assert result["status"] == "DEVELOPMENT_DIAGNOSTIC_PASS"
    assert result["is_prospective_certification"] is False
    assert result["activations"] == 10 and result["false_activations"] == 2
    np.testing.assert_allclose(result["observed_mean_net_gain"], (.05*8 - .01*2)/40)
    with pytest.raises(ValueError, match="overlap"):
        evaluate_contract(*args, RiskContract(min_activations=5), gain_support=(-1.02, 1.),
                          wells_if_selected=np.ones(40), calibration_ids=["e1"], assume_iid_units=True)


def test_contract_conservative_multiple_criteria_and_conflicting_labels():
    args = _evaluation_inputs()
    contract = RiskContract(max_fdp=.35, max_fpr=.075, min_coverage=.05, min_activations=100)
    result = evaluate_contract(*args, contract, gain_support=(-1.02, 1.),
                               wells_if_selected=np.ones(40), assume_iid_units=True)
    np.testing.assert_allclose(result["alpha_per_declared_statistical_criterion"], .05/3)
    assert not result["passed"] and not result["checks"]["min_activations"]
    bad = list(args); bad[2] = np.zeros(40, dtype=bool)
    with pytest.raises(ValueError, match="conflict"):
        evaluate_contract(*bad, contract, gain_support=(-1.02, 1.),
                          wells_if_selected=np.ones(40), assume_iid_units=True)
    with pytest.raises(ValueError, match="IID"):
        evaluate_contract(*args, contract, gain_support=(-1.02, 1.), wells_if_selected=np.ones(40))


def test_explicit_pointwise_design_preserves_default_contract_results():
    args = _evaluation_inputs()
    contract = RiskContract(max_fdp=.35, min_coverage=.05)
    options = dict(gain_support=(-1.02, 1.), wells_if_selected=np.ones(40), assume_iid_units=True)
    default = evaluate_contract(*args, contract, **options)
    explicit = evaluate_contract(*args, contract, selection_design="pointwise", **options)
    assert explicit == default
    assert explicit["selection_design"] == "pointwise"


@pytest.mark.parametrize("design", ["cohort_coupled", "knapsack", "top_k", "within_independent_cluster"])
def test_cohort_coupled_allocation_cannot_receive_compound_CP_bounds(design):
    # An IID input declaration does not make a jointly allocated output IID.
    with pytest.raises(ValueError, match="cohort-coupled.*IID compound CP.*within_independent_cluster"):
        evaluate_contract(*_evaluation_inputs(), RiskContract(max_fdp=.35),
                          gain_support=(-1.02, 1.), wells_if_selected=np.ones(40),
                          assume_iid_units=True, developmental=False, selection_design=design)
