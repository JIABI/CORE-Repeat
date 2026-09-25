"""Check released observations, identities and saved outcomes without fitting."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
from release_paths import DATA, DATA_ROOT, RESEARCH, QA


def cosine(a, b):
    return np.einsum("ij,ij->i", a, b) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))


def gamma(y):
    return .5 * (cosine(y[:, :3].mean(axis=1), y[:, 3]) - cosine(y[:, 0], y[:, 3])) - .02


def main():
    paths = {
        "EU": (904, "reports/eu_core_development_20260917_v1/prepared_data_cc904/data.npz"),
        "JUMP": (639, "data/source5_primary_fullcontrols/measurements.npz"),
        "LINCS": (1188, "data/lincs_pilot1_biology_20260915/data.npz"),
        "RxRx3": (10410, "data/rxrx3_r2_20260918/prepared_r2/data.npz"),
    }
    saved = pd.read_csv(DATA / "measurement_fig2_object_predictions.csv", float_precision="round_trip")
    assert len(saved) == 13141
    checks = {}
    for dataset, (n, rel) in paths.items():
        with np.load(RESEARCH / rel, allow_pickle=False) as z:
            ids, y = z["ids"], z["Y"]
        assert y.shape[:2] == (n, 4) and len(ids) == n and len(np.unique(ids)) == n
        actual = gamma(y)
        group = saved.loc[saved.dataset == dataset]
        assert len(group) == n
        ix = pd.Index(ids).get_indexer(group.candidate_id.astype(str))
        assert (ix >= 0).all()
        np.testing.assert_allclose(actual[ix], group.actual_gamma.to_numpy(), rtol=0, atol=1e-12)
        checks[dataset] = {"conditions": n, "features": y.shape[2],
                           "gamma_max_abs_error": float(np.max(np.abs(actual[ix]-group.actual_gamma)))}
    root = RESEARCH / "runs/r4_confirmation_20260921_v1"
    with np.load(root / "ingest/x/query.npz", allow_pickle=False) as x, np.load(root / "ingest/outcomes/outcomes.npz", allow_pickle=False) as o:
        np.testing.assert_array_equal(x["ids"], o["ids"])
        ids, eligible = x["ids"].copy(), x["eligible"].copy()
        y = np.concatenate((x["X"][:, None, :], o["future"]), axis=1)
    complete = np.isfinite(y).all(axis=(1, 2)) & (np.linalg.norm(y, axis=2) > 0).all(axis=1)
    observed = np.full(len(ids), np.nan)
    observed[complete] = gamma(y[complete])
    q = pd.read_csv(RESEARCH / "reports/r4_execution_20260921_v1/primary/campaign/object_results.csv", float_precision="round_trip")
    np.testing.assert_array_equal(ids, q.object_id)
    assert len(ids) == 1539 and eligible.sum() == 1527 and np.isfinite(observed).sum() == 1520
    np.testing.assert_allclose(observed, q.gamma.to_numpy(), rtol=0, atol=1e-12, equal_nan=True)
    with np.load(root / "selections/selections.npz", allow_pickle=False) as sel:
        np.testing.assert_array_equal(ids, sel["ids"])
        for policy in ["CORE", "HISTGB_CAL"]:
            m = sel[policy + "__selected"]
            assert m.sum() == 192 and (eligible[m]).all()
            np.testing.assert_array_equal(m, q[policy + "_selected"])
    tables = pd.read_csv(DATA_ROOT / "tables/table_index.csv")
    assert len(tables) == 64
    main_tables = tables.loc[tables.document == "main"].sort_values("table_number")
    assert main_tables.primary_label.tolist() == [
        "tab:r4_primary", "tab:r2_four_policy", "tab:eu_measurement_main"]
    assert int((tables.document == "supplementary").sum()) == 61
    pairs = pd.read_csv(DATA / "plot_inputs/fig6_residual_pairs.csv")
    assert len(pairs) == 93558 and (pairs.target_overlap > 0).sum() == 3088
    checks["confirmation"] = {"qualified": 1539, "eligible_X": 1527, "observed_gamma": 1520,
        "selected_each": 192, "gamma_max_abs_error": float(np.nanmax(np.abs(observed-q.gamma)))}
    folder = DATA / "eu_direct_measurement_20260924"
    summary = pd.read_csv(folder / "summary.csv")
    family = summary.loc[summary.target == "three_target_mean"].set_index("arm")
    assert set(family.index) == {"CORE", "CAL", "RAW"} and (summary.n == 1520).all()
    paired = pd.read_csv(folder / "paired_intervals.csv")
    primary = paired.loc[(paired.comparator == "CAL") & (paired.target == "three_target_mean")]
    assert set(primary.resampling) == {"chemical_identity", "library_layout"}
    assert primary.primary.all() and (primary.repetitions == 10000).all()
    np.testing.assert_allclose(primary.crps_difference,
        family.loc["CAL", "crps"] - family.loc["CORE", "crps"], rtol=0, atol=1e-14)
    assert (primary.difference_lower < 0).all() and (primary.difference_upper > 0).all()
    assert (primary.width95_lower > 0).all()
    width_reduction = 100 * (1 - family.loc["CORE", "width_95"] / family.loc["CAL", "width_95"])
    assert round(width_reduction, 1) == 21.8
    assert round(100 * family.loc["CORE", "coverage_95"], 1) == 93.9
    assert round(100 * family.loc["CAL", "coverage_95"], 1) == 94.2
    checks["eu_direct_measurement"] = {"complete_objects": 1520, "primary": "CAL minus CORE",
        "nominal_95_width_reduction_percent": float(width_reduction), "post_hoc": True}
    checks["display_tables"] = {"main": 3, "supplementary": 61, "total": 64}
    checks["status"] = "PASS"
    (QA / "release_data_validation.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
