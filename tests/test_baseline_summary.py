"""Aggregation tests on saved synthetic fixtures, never real measurements."""
import json

import numpy as np
import pytest

from opal2.baseline_policy import evaluate_predictions
from opal2.baseline_summary import summarize, compact
from opal2.representation_probe import regression_metrics


def save_json(path, value):
    def default(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        raise TypeError(type(x))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, default=default))


def write_forecast(folder, ids, offset=0.):
    folder.mkdir(parents=True, exist_ok=True)
    n = len(ids)
    actual = np.column_stack((np.linspace(-.02, .03, n), np.linspace(.03, -.03, n),
                              np.linspace(-.04, .04, n))) + offset
    predicted = actual + .003
    pn = np.where(actual <= 0, .8, .2)
    report = evaluate_predictions(predicted, pn, actual, ids, fractions=(.5,),
                                  n_random=3, n_bootstrap=3)
    np.savez(folder / "predictions.npz", ids=np.array(ids), actual=actual, predicted=predicted,
             p_null=pn, mc_se=np.ones((n, 3)) * .001, raw_nll=np.ones(n),
             standardized_nll=np.ones(n) * 2, standardized_mse=np.arange(1, n + 1) / 6,
             standardized_sse=np.arange(1, n + 1))
    save_json(folder / "policy.json", report)
    return report


def metrics(y, pred):
    mean = np.zeros(y.shape[1:])
    raw = regression_metrics(y, pred, mean)
    centered = regression_metrics(y, pred, y.mean(0))
    return dict(standardized=raw, physical=raw,
                r2_evaluation_mean=centered["overall"]["r2_training_mean"],
                slot_r2_evaluation_mean=[r["r2_training_mean"] for r in centered["slots"]])


def write_probe(folder, ids, arm_number=0, fixed_splits=None):
    folder.mkdir(parents=True, exist_ok=True)
    y = np.arange(len(ids) * 6, dtype=float).reshape(len(ids), 3, 2) + 1
    pred = y + (arm_number + 1) * .01
    parts = {"all_test": metrics(y, pred)}
    if fixed_splits:
        for name, names in fixed_splits.items():
            ix = [ids.index(unit) for unit in names]
            parts[name] = metrics(y[ix], pred[ix])
    report = dict(test_ids=ids, train_ids=["train"], input_dimension=4,
                  selected_alphas=[.1, .1, .1], partitions=parts)
    np.savez(folder / "predictions.npz", ids=np.array(ids), predictions=pred)
    save_json(folder / "metrics.json", report)


def fixture(root):
    names = [f"u{i:02d}" for i in range(12)]
    splits = dict(train=names[:6], validation=names[6:8], calibration=names[8:10], evaluation=names[10:])
    folds = [dict(repeat=0, fold=i, test_ids=names[i * 4:(i + 1) * 4]) for i in range(3)]
    probe_folds = [dict(repeat=0, fold=i, test_ids=names[6 + i * 2:8 + i * 2]) for i in range(3)]
    config = dict(final_opened=False, fifth_repeat_opened=False, biology_kernel_active=False,
                  original_contract_changed=False, split_ids=splits, folds=folds,
                  probe_folds=probe_folds, jepa_arms=["J1", "J2"], seed=5, n_bootstrap=7)
    save_json(root / "config.json", config)
    for split in ("validation", "calibration", "evaluation"):
        write_forecast(root / "fixed_baseline" / split, splits[split])
    for task in folds:
        write_forecast(root / "stability_baseline" / f"repeat_0_fold_{task['fold']}", task["test_ids"], .1 * task["fold"])
    for number, arm in enumerate(("raw", "reliability", "J1", "J2")):
        write_probe(root / "fixed_probes" / "fixed383" / arm, names[6:], number,
                    {key: splits[key] for key in ("validation", "calibration", "evaluation")})
        for task in probe_folds:
            write_probe(root / "stability_probes" / f"repeat_0_fold_{task['fold']}" / arm,
                        task["test_ids"], number)
    return config


def test_saved_summary_preserves_counts_fold_selections_and_error_denominators(tmp_path):
    cfg = fixture(tmp_path)
    result = summarize(tmp_path)
    assert not result["pending"]
    assert result["fixed_baseline"]["evaluation"]["n"] == 2
    pooled = result["stability_baseline"][0]
    assert pooled["n"] == 12
    selected = next(r for r in pooled["policies"] if r["section"] == "within_action" and
                    r["action"] == "Z1" and r["ranking"] == "expected_gain")
    # Each four-object fold selects two, rather than globally selecting the
    # highest six scores (which would favor the offset .2 fold).
    assert selected["selected_ids"] == ["u02", "u03", "u06", "u07", "u10", "u11"]
    assert selected["selected_n"] == selected["used_wells"] == 6
    assert selected["paired_value_vs_fold_random"]["n"] == 12
    assert selected["per_eligible_net_gain"] == pytest.approx(.5 * selected["per_selected_net_gain"])
    assert pooled["measurement_metrics"]["max_object_sse_fraction"] == pytest.approx(4 / 30)
    probes = result["stability_probes"][0]
    assert probes["n"] == 6
    assert len(probes["paired"]) == 6
    assert all(r["n"] == 6 and not r["formal_certificate"] for r in probes["paired"])
    raw = probes["arms"]["raw"]["metrics"]["standardized"]
    assert raw["sse"] == pytest.approx(6 * 6 * .01 ** 2)
    assert raw["mse"] == pytest.approx(.01 ** 2)
    assert raw["r2_training_mean"] == pytest.approx(1 - raw["sse"] / raw["baseline_sse"])
    assert probes["arms"]["raw"]["r2_evaluation_mean"] is None
    assert result["fixed_probes"]["raw"]["partitions"]["evaluation"]["r2_evaluation_mean"] is not None
    assert "Pending" not in compact(result)
    assert json.loads((tmp_path / "summary.json").read_text())["no_new_measurements_read"]


def test_incomplete_repeat_is_pending_not_partially_pooled(tmp_path):
    fixture(tmp_path)
    (tmp_path / "stability_baseline" / "repeat_0_fold_2" / "policy.json").unlink()
    result = summarize(tmp_path)
    assert result["stability_baseline"] == []
    assert "stability_baseline/repeat_0" in result["pending"]
    assert "Pending" in compact(result)


@pytest.mark.parametrize("problem", ["duplicate_population", "changed_selected_ids", "changed_utility", "probe_order", "forbidden_scope"])
def test_misaligned_or_relabelled_evidence_fails(tmp_path, problem):
    fixture(tmp_path)
    if problem in {"duplicate_population", "forbidden_scope"}:
        path = tmp_path / "config.json"
        cfg = json.loads(path.read_text())
        if problem == "duplicate_population":
            cfg["split_ids"]["evaluation"][0] = cfg["split_ids"]["validation"][0]
        else:
            cfg["fifth_repeat_opened"] = True
        save_json(path, cfg)
    elif problem == "probe_order":
        path = tmp_path / "fixed_probes" / "fixed383" / "raw" / "predictions.npz"
        np.savez(path, ids=np.array(["u11", "u10", "u09", "u08", "u07", "u06"]))
    else:
        path = tmp_path / "stability_baseline" / "repeat_0_fold_0" / "policy.json"
        report = json.loads(path.read_text())
        if problem == "changed_selected_ids":
            report["within_action"][0]["selected_ids"] = ["not_here"]
        else:
            report["row_trace"][0]["actual"]["Z1"] = 500
        save_json(path, report)
    with pytest.raises(ValueError):
        summarize(tmp_path)
