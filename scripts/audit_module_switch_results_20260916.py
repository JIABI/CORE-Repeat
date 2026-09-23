"""Independent, read-only audit of the two declared module-switch comparisons.

Run with the project .venv Python. The script reads completed saved predictions,
role declarations, and training summaries; it does not import experiment runners,
fit a model, open raw query profiles, or modify experiment artifacts. JSON is
emitted to stdout. An explicit --output creates a new audit file exclusively.
Exit 0: all requested runs pass; 2: at least one is pending; 1: failure.
Developmental score/risk changes are not acquisition certification.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS = [PROJECT/"runs/module_switches_20260916_v1"/name
                for name in ("lincs_v2", "lkcp_transfer_v1")]
ARMS = ("CORE", "BIO", "PCA_STATE", "DIRECT_STATE", "CONDITIONAL_STATE")
LEVELS = (.5, .8, .9, .95, .99)
SCORES = ("nll", "energy", "crps", "brier", "single_crps", "pair_crps",
          "average_crps", "absolute_pair_crps")
METADATA = {"ids", "groups", "layout", "fold", "actual", "selected",
            "resource_support", "effective_mixing", "changed_weights", "policy_value"}


def read_json(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def same(actual, expected, name):
    require(np.array_equal(actual, expected), name+" differs bitwise")


def near(actual, expected, name):
    require(np.allclose(actual, expected, atol=1e-12, rtol=1e-11, equal_nan=False),
            name+" does not reproduce")


def read_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def paired_cluster_difference(a, b, labels):
    """Independent copy of the specified 2,000-draw cluster bootstrap recipe."""
    delta = np.asarray(a)-np.asarray(b)
    labels = np.asarray(labels)
    values = np.unique(labels)
    totals = np.array([delta[labels == g].sum() for g in values])
    sizes = np.array([(labels == g).sum() for g in values])
    rng = np.random.default_rng(20260916)
    indices = rng.integers(len(values), size=(2000, len(values)))
    means = totals[indices].sum(1)/sizes[indices].sum(1)
    return dict(difference=float(delta.mean()), ci95=np.quantile(means, [.025, .975]).tolist())


def recompute_metrics(out, core, actual):
    selected = out["selected"].astype(bool)
    require(selected.any(), "No selected objects despite the declared nonzero budget")
    return dict(n=len(actual), selected_n=int(selected.sum()),
        selected_null=int(np.count_nonzero(actual[selected] <= 0)),
        selected_mean=float(actual[selected].mean()),
        actual_null_rate=float(np.mean(actual <= 0)),
        p_null_mean=float(out["p_null"].mean()),
        null_auc=float(roc_auc_score(actual <= 0, out["p_null"])),
        gamma_spearman=float(spearmanr(actual, out["predicted"]).statistic),
        scores={key: float(out[key].mean()) for key in SCORES},
        joint_coverage={str(level): float(out["joint_coverage_by_level"][:, j].mean())
                        for j, level in enumerate(LEVELS)},
        eligible_reference_queries=int(out["resource_support"].sum()),
        effective_nonzero_queries=int(np.count_nonzero(out["effective_mixing"] > 0)),
        changed_weight_queries=int(out["changed_weights"].sum()),
        changed_selected_membership=int(np.count_nonzero(out["selected"] != core["selected"])),
        original_mean_preserved=True,
        covariance_multiplier_mean=float(out["radial_variance_multiplier"].mean()))


def compare_tree(actual, expected, prefix):
    require(set(actual) == set(expected), prefix+" fields differ")
    for key, value in actual.items():
        name = prefix+"."+key
        if isinstance(value, dict):
            compare_tree(value, expected[key], name)
        elif isinstance(value, (bool, int, str)):
            require(value == expected[key], name+" differs")
        else:
            near(value, expected[key], name)


def group_sets(ids, lookup):
    require(len(ids) == len(set(ids)), "Duplicate IDs inside a declared role")
    require(set(ids).issubset(lookup), "Role ID absent from the declared source population")
    return {lookup[name] for name in ids}


def audit_switch_selection(module):
    choice = module["selection"]
    if choice is None:
        return
    if choice["reason"] != "calibration_group_cv":
        require(choice["reason"] == "insufficient_calibration_groups", "Unexpected module selector")
        require(not any(module["coefficients"]), "Unsupported calibration enabled a module")
        return
    eligible = []
    for i, row in enumerate(choice["candidates"]):
        accepted = i == 0 or row["group_mean_paired_difference"] < -row["group_se"]-1e-12
        require(row["eligible"] == accepted, "Saved module admission rule differs")
        if accepted and i:
            eligible.append(i)
    selected = min(eligible, key=lambda j: (sum(choice["candidates"][j]["coefficients"]),
                    choice["candidates"][j]["group_mean_paired_difference"],
                    tuple(choice["candidates"][j]["coefficients"]))) if eligible else 0
    require(choice["selected_index"] == selected, "Module strength selected using a different rule")
    same(module["coefficients"], choice["candidates"][selected]["coefficients"], "selected module coefficients")
    require(choice["calibrated_admission"] == bool(selected), "Module admission status differs")


def audit_run(path):
    path = Path(path).resolve()
    if not (path/"status.json").exists():
        return dict(run=str(path), state="PENDING", reason="No status record")
    status = read_json(path/"status.json")
    if status.get("state") != "COMPLETE":
        return dict(run=str(path), state="PENDING", source_state=status.get("state"),
                    completed_cells=status.get("cells_complete", 0))
    require((path/"summary.json").exists(), "COMPLETE run has no committed summary")
    summary = read_json(path/"summary.json")
    require(summary["state"] == "COMPLETE", "Summary/status disagree")
    require(summary["samples"] == 100000, "Declared Monte Carlo budget changed")
    require(summary["epochs_max"] <= 60, "Representation epoch budget expanded")
    require(not summary["endpoint_changed"] and not summary["mean_changed"], "Fixed core endpoint/mean changed")
    require(not summary["formal_certificate"], "Developmental result mislabeled a certificate")
    lkcp = (path/"roles_before_target_scoring.json").exists()
    if lkcp:
        declared = read_json(path/"roles_before_target_scoring.json")
        all_ids, all_groups = declared["ids"], declared["groups"]
        expected_ids = all_ids[declared["source_count"]:]
        expected_cells = 5
        old_cells = None
        require(summary["unknown_dose_blocks_biology"], "LKCP unknown dose must block biology")
        require(summary["core_fold"] == 0 and declared["core_fold"] == 0, "Frozen source fold changed")
    else:
        source = read_json(Path(summary["source"])/"summary.json")
        manifest = read_json(Path(source["reference_run"])/"run_manifest.json")
        all_ids, all_groups = manifest["ids"], manifest["groups"]
        expected_ids, expected_cells = all_ids, 10
        old_cells = {(c["fold"], c["half"]): c for c in source["cells"]}
    require(len(all_ids) == len(set(all_ids)), "Declared source IDs are not unique")
    population = dict(zip(all_ids, all_groups))
    require(len(population) == len(all_ids), "Population mapping is incomplete")
    stores = {arm: read_npz(path/(arm+".npz")) for arm in ARMS}
    core = stores["CORE"]
    if not lkcp:
        original_core = read_npz(Path(summary["source"])/"AMP_EMP_LOCAL.npz")
        for key in ("ids", "groups", "fold", "mean_u", "actual_u"):
            same(core[key], original_core[key], "unchanged source core "+key)
        near(core["actual"], original_core["actual"], "unchanged realized endpoint")
        near(core["covariance_u"], original_core["covariance_u"], "unchanged CORE joint covariance")
    ids = core["ids"].astype(str)
    same(ids, expected_ids, "complete ordered query scope")
    n, actual = len(ids), core["actual"]
    require(len(set(ids)) == n and n == summary["n"], "Query duplicates/count mismatch")
    require(np.isfinite(actual).all(), "Nonfinite realized Gamma")
    same(core["groups"], [population[name] for name in ids], "query chemical groups")
    lookup = {name: i for i, name in enumerate(ids)}
    require(len(summary["cells"]) == expected_cells, "Wrong number of query cells")
    seen, budget_sum, training_checked = np.zeros(n, int), 0, set()
    lambda0_selected = np.zeros(n, bool)
    random_value, random_null = 0., 0.
    activation_cells = {arm: 0 for arm in ARMS}
    for cell in summary["cells"]:
        fold, half = cell["fold"], cell["half"]
        file = path/(f"cell_{fold}.json" if lkcp else f"cell_{fold}_{half}.json")
        require(read_json(file) == cell, "Per-cell record differs from committed summary")
        q = np.array([lookup[name] for name in cell["query_ids"]])
        same(core["fold"][q], np.full(len(q), fold), "query fold labels")
        role_names = ["model_fit_ids", "calibration_ids", "query_ids"]
        if lkcp:
            role_names.insert(1, "covfit_ids")
        else:
            original = old_cells[(fold, half)]
            same(cell["query_ids"], original["query_ids"], "original query roles")
            same(cell["calibration_ids"], original["representative_ids"], "original radial calibration representatives")
            role_names.insert(1, "covfit_ids")
            cell = {**cell, "covfit_ids": original["fit_ids"]}
        parts = [group_sets(cell[key], population) for key in role_names]
        require(all(not parts[i] & parts[j] for i in range(len(parts)) for j in range(i+1, len(parts))),
                "Chemical identity crosses new model/reference/calibration/query roles")
        if lkcp:
            declared_row = next(row for row in declared["roles"] if row["fold"] == fold)
            for role, key in (("fit", "model_fit_ids"), ("covfit", "covfit_ids"),
                              ("calibration", "calibration_ids"), ("query", "query_ids")):
                same([all_ids[i] for i in declared_row[role]], cell[key], "predeclared LKCP "+role)
            expected_budget = int(np.floor(.25*len(q)))//2
        else:
            expected_budget = old_cells[(fold, half)]["budget"]
        require(cell["budget"] == expected_budget, "Cell budget changed")
        budget_sum += expected_budget
        lambda0_order = np.lexsort((ids[q], -core["predicted"][q]))
        lambda0_selected[q[lambda0_order[:expected_budget]]] = True
        random_value += expected_budget*float(actual[q].mean())
        random_null += expected_budget*float(np.mean(actual[q] <= 0))
        seen[q] += 1
        if fold not in training_checked:
            folder = path/f"fold_{fold}"
            pca = read_npz(folder/"pca_state.npz")
            same(pca["train_ids"], cell["model_fit_ids"], "PCA fitting identities")
            for arm in ("DIRECT_STATE", "CONDITIONAL_STATE"):
                report = read_json(folder/(arm+"_training.json"))
                fit_ids, val_ids = report["inner_fit_ids"], report["inner_validation_ids"]
                require(set(fit_ids+val_ids) == set(cell["model_fit_ids"]), "Representation internal split is not MODEL_FIT")
                require(not group_sets(fit_ids, population) & group_sets(val_ids, population), "Representation inner group leakage")
                require(1 <= report["best_epoch"] <= report["epochs_completed"] <= report["epochs_requested"] <= 60,
                        "Invalid representation checkpoint/epoch budget")
                require(not report["original_endpoint_changed"], "Auxiliary representation changed endpoint")
            training_checked.add(fold)
        weights_file = path/(f"cell_{fold}_weights.npz" if lkcp else f"cell_{fold}_{half}_weights.npz")
        weights = read_npz(weights_file)
        same(weights["query_ids"], cell["query_ids"], "weight query order")
        same(weights["reference_ids"], cell["calibration_ids"], "weight reference order")
        allowed = core["groups"][q, None] != np.array([population[x] for x in cell["calibration_ids"]])[None]
        for arm in ARMS:
            w = weights[arm]
            require(w.shape == allowed.shape and np.isfinite(w).all() and (w >= 0).all(), "Invalid reference weights")
            near(w.sum(1), np.ones(len(q)), "reference normalization")
            require(np.all(w[~allowed] == 0), "Query group receives its own reference outcomes")
            changed = np.any(w != weights["CORE"], axis=1)
            same(stores[arm]["changed_weights"][q], changed, arm+" changed-weight flag")
            same(stores[arm]["effective_mixing"][q], cell["modules"][arm]["gate"], arm+" effective gate")
            same(stores[arm]["resource_support"][q], cell["modules"][arm]["support"], arm+" resource support")
            audit_switch_selection(cell["modules"][arm])
            activation_cells[arm] += int(changed.any())
            values = stores[arm]["predicted"][q]-.2*stores[arm]["p_null"][q]
            order = np.lexsort((ids[q], -values))
            selected = np.zeros(len(q), dtype=int)
            selected[order[:expected_budget]] = 1
            same(stores[arm]["selected"][q], selected, arm+" exact score/budget/stable-ID selection")
    same(seen, np.ones(n, int), "one query appearance per object")
    checked_off = {}
    recomputed = {}
    for arm, out in stores.items():
        require(set(out) == set(core), arm+" arrays differ from CORE schema")
        for key in ("ids", "groups", "layout", "fold", "actual", "mean_u", "actual_u"):
            same(out[key], core[key], arm+" "+key)
        for key, value in out.items():
            require(value.shape[0] == n, arm+" array count mismatch: "+key)
            if value.dtype.kind in "biufc":
                require(np.isfinite(value).all(), arm+" nonfinite "+key)
        require(np.all((out["p_null"] >= 0) & (out["p_null"] <= 1)), arm+" invalid NULL probability")
        require(np.all(np.isin(out["selected"], (0, 1))), arm+" nonbinary decisions")
        require(int(out["selected"].sum()) == budget_sum, arm+" wrong total acquisition count")
        same(out["brier"], (out["p_null"]-(actual <= 0))**2, arm+" Brier")
        same(out["policy_value"], out["selected"]*actual, arm+" realized policy value")
        off = ~out["changed_weights"].astype(bool)
        numeric_keys = [key for key in out if key not in METADATA]
        for key in numeric_keys:
            same(out[key][off], core[key][off], arm+" off-query "+key)
        if off.all():
            same(out["selected"], core["selected"], arm+" completely-off decisions")
        checked_off[arm] = dict(objects=int(off.sum()), prediction_arrays=len(numeric_keys),
                               note="Individual off-query membership need not stay fixed if other cohort scores change")
        recomputed[arm] = recompute_metrics(out, core, actual)
        compare_tree(recomputed[arm], summary["metrics"][arm], arm+" summary")
    if lkcp:
        require(not stores["BIO"]["resource_support"].any(), "LKCP BIO used unknown/mismatched contexts")
        require(not stores["BIO"]["effective_mixing"].any(), "LKCP BIO has nonzero gate")
        require(not stores["BIO"]["changed_weights"].any(), "LKCP BIO unexpectedly changed distribution")
    references = dict(core_lambda0=dict(selected_n=int(lambda0_selected.sum()),
        selected_null=int(np.count_nonzero(actual[lambda0_selected] <= 0)),
        selected_mean=float(actual[lambda0_selected].mean())),
        uniform_random_expectation=dict(selected_n=budget_sum, selected_null=random_null,
                                       selected_mean=random_value/budget_sum),
        no_additional_measurements=dict(net_value=0.),
        all_add_two_budget_unmatched=dict(selected_n=n, selected_null=int(np.count_nonzero(actual <= 0)),
                                         selected_mean=float(actual.mean())))
    compare_tree(references, summary["references"], "fixed and random reference arithmetic")
    # Recompute all declared paired cluster intervals from fixed predictions.
    comparisons = {arm: (arm, "CORE") for arm in ARMS if arm != "CORE"}
    comparisons.update(CONDITIONAL_minus_DIRECT=("CONDITIONAL_STATE", "DIRECT_STATE"),
                       CONDITIONAL_minus_PCA=("CONDITIONAL_STATE", "PCA_STATE"))
    for name, (a, b) in comparisons.items():
        for metric, scopes in summary["comparisons"][name].items():
            for scope, recorded in scopes.items():
                labels = core["groups"] if scope == "chemistry" else core["layout"]
                compare_tree(paired_cluster_difference(stores[a][metric], stores[b][metric], labels),
                             recorded, name+" "+metric+" "+scope)
    highlights = {}
    for arm in ARMS:
        out = stores[arm]
        highlights[arm] = dict(selected_n=int(out["selected"].sum()),
            selected_null=recomputed[arm]["selected_null"], selected_mean=recomputed[arm]["selected_mean"],
            population_policy_value=float(out["policy_value"].mean()),
            gamma_crps=recomputed[arm]["scores"]["crps"], null_brier=recomputed[arm]["scores"]["brier"],
            geometry_nll=recomputed[arm]["scores"]["nll"], null_auc=recomputed[arm]["null_auc"],
            nonzero_query_count=int(out["changed_weights"].sum()), activation_cells=activation_cells[arm],
            selected_symmetric_difference=recomputed[arm]["changed_selected_membership"])
    return dict(run=str(path), state="PASS", query_n=n, query_cells=expected_cells,
        chemical_groups=len(np.unique(core["groups"])), layout_groups=len(np.unique(core["layout"])),
        total_activations=budget_sum, future_queries_used_for_new_module_fit=False,
        mean_preservation_verified=True, off_query_replay=checked_off,
        metrics_and_cluster_intervals_recomputed=True, audit_only=True,
        uncertainty_vs_decision="Geometry NLL/coverage and Gamma CRPS/Brier are predictive scores; actual selected Gamma, NULL count and policy value are separate decision results",
        noncertification="Repeated-development fixed-prediction comparisons; no iid CP or new independent certificate",
        layout_interval_scope=("Descriptive sensitivity only: fewer than 10 layout groups; no reliable independent deployment inference"
                               if len(np.unique(core["layout"])) < 10 else
                               "Fixed-prediction layout-cluster sensitivity, not deployment certification"),
        core_scope=("Transferred frozen Pilot1 fold0 STATE50 mean; not a newly fitted LKCP-native model"
                    if lkcp else "Opened LINCS fixed source mean"),
        source_core_replay=("Within-arm means fixed; complete saved predictor separately golden-replayed by test_frozen_state50_transfer.py"
                            if lkcp else "Means, targets and identity/fold arrays match prior source bitwise; covariance agrees numerically"),
        references=references,
        highlights=highlights)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="*", type=Path)
    parser.add_argument("--output", type=Path, help="Create a NEW audit JSON; refuse overwrite")
    args = parser.parse_args()
    results = []
    for path in args.runs or DEFAULT_RUNS:
        try:
            result = audit_run(path)
        except Exception as exc:
            result = dict(run=str(path.resolve()), state="FAIL", error=type(exc).__name__+": "+str(exc))
        results.append(result)
    encoded = json.dumps(dict(audits=results), indent=2, allow_nan=False)
    if args.output is not None:
        with args.output.open("x") as stream:
            stream.write(encoded+"\n")
    print(encoded)
    raise SystemExit(1 if any(r["state"] == "FAIL" for r in results)
                     else 2 if any(r["state"] == "PENDING" for r in results) else 0)


if __name__ == "__main__":
    main()
