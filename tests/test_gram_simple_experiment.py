import numpy as np
import pytest

from opal2.gram_simple_experiment import paired_policy, coordinate_scores


def _report(selected):
    row = dict(label="common__Z1Z2__expected_gain__0.5", action="Z1Z2",
               ranking="expected_gain", fraction=.5, selected_n=2,
               used_wells=4, budget_wells=4, selected_ids=selected)
    return {"policy": {"within_action": [], "common_budget": [row]}}


def test_pair_global_uses_uniform_expectation_not_lexical_mask(tmp_path):
    actual = np.array([[0., 0., -.1], [0., 0., -.2], [0., 0., .3], [0., 0., .4]])
    ids = np.array(["a", "b", "c", "d"])
    for side in ("left", "right"):
        np.savez(tmp_path/f"{side}.npz", ids=ids, actual=actual)
    result = paired_policy(_report(["c", "d"]), _report(["a", "b"]),
        tmp_path/"left.npz", tmp_path/"right.npz", right_global=True,
        n_bootstrap=20)
    row = result["rows"][0]
    assert row["value_per_selected"]["mean"] == pytest.approx(.25)
    assert row["fdp"]["mean"] == pytest.approx(-.5)
    assert row["fpr"]["mean"] == pytest.approx(-.5)
    assert row["selected_overlap"] is None


def test_paired_policy_rejects_different_id_order(tmp_path):
    for side, ids in (("left", ["a", "b"]), ("right", ["b", "a"])):
        np.savez(tmp_path/f"{side}.npz", ids=ids, actual=np.zeros((2,3)))
    with pytest.raises(ValueError, match="identical ordered"):
        paired_policy(_report(["a","b"]),_report(["a","b"]),
                      tmp_path/"left.npz", tmp_path/"right.npz",n_bootstrap=2)


def test_coordinate_score_uses_training_reference():
    actual = np.ones((3,9))*2
    score = coordinate_scores(actual,np.ones((3,9)),np.zeros(9))
    assert score["r2_vs_train_mean"] == .75
