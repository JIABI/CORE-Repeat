#!/usr/bin/env python3
"""Render saved M3 tables into a compact evidence report; no fitting."""
from pathlib import Path
import json
import shutil

import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT/"reports/m3_amplitude_controls_20260922_v1"
RUN = PROJECT/"runs/m3_amplitude_controls_20260922_v1"


def table(frame, columns, formats=None):
    formats = formats or {}
    rows = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for _, row in frame.iterrows():
        values = []
        for key in columns:
            v = row[key]
            values.append(formats[key].format(v) if key in formats else str(v))
        rows.append("| " + " | ".join(values) + " |")
    return rows


def main():
    dev = pd.read_csv(ROOT/"development_summary.csv")
    # The literal endpoint name NULL is a metric, not a missing CSV value.
    conf = pd.read_csv(ROOT/"confirmation/summary.csv", keep_default_na=False)
    final = json.loads((RUN/"confirmation/dispersion_selection_frozen.json").read_text())
    if not (RUN/"complete.json").exists():
        raise RuntimeError("Development execution is not complete")
    lines = ["# M3 amplitude and dispersion allocation controls", "",
        "All 60 original development deployment cells are included. New controls use the same identities, "
        "role assignments and fixed quotas as the original comparisons. CORE and calibrated HistGB are unchanged.", "",
        "## Development", "",
        "Gamma includes two added-well costs of 0.01 each; higher total Gamma and fewer NULL selections are preferable. "
        "The random row is the exact expectation under each deployment cell's quota.", ""]
    lines += table(dev, ["dataset", "arm", "selected", "total_Gamma", "NULL", "excess_over_random"],
                   {"selected":"{:.0f}", "total_Gamma":"{:.4f}", "NULL":"{:.2f}", "excess_over_random":"{:.4f}"})
    lines += ["", "The learned AMP_HISTGB arm uses only log first-well amplitude. DISPERSION_DESC instead uses "
        "the full ACCESS_MATCHED first-well and chemical inputs, predicting log W and ranking it in the declared "
        "descending direction. It is therefore a target-based control, not another amplitude-only method.", "",
        "## Post-hoc confirmation", "",
        "There are 1,539 metadata-qualified candidates and 1,527 eligible first wells; every list selects 192. "
        "AMP_ASC, AMP_DESC, AMP_HISTGB and DISPERSION_DESC are post-hoc controls. The two learned controls use only "
        "the existing final DEV904 fitting roles and the same configurations declared for development.", ""]
    for endpoint in ("Gamma", "NULL", "paired_cross_site"):
        lines += ["### " + endpoint, ""]
        subset = conf[conf.endpoint == endpoint]
        cols = ["arm", "observed_selected", "missing_selected", "observed_sum", "total_lower", "total_upper"]
        if endpoint == "paired_cross_site":
            cols = ["arm", "observed_selected", "missing_selected", "available_case_mean",
                    "fixed_list_mean_lower", "fixed_list_mean_upper"]
        lines += table(subset, cols, {key:"{:.4f}" for key in cols if key != "arm"})
        lines += [""]
    lines += ["Missing Gamma lies in [-1.02, 0.98], NULL in [0, 1], and the paired external increment in [-2, 2]. "
        "These are finite-cohort identification bounds, not confidence intervals. A random row's observed and missing "
        "selection counts are expected counts. Its inclusion probability is 192/1527 for every eligible object.", "",
        "The paired external endpoint averages MEDINA and USC neighbourhood-Spearman improvements and is observed "
        "only when both sites are observed. Available-case means have their own reported denominators; they are not "
        "full-list means. Gamma availability is never used to filter the cross-site endpoint.", "",
        "## Comparisons and reproducibility", "",
        "`paired_intervals.csv` contains 2,000 fixed-list paired chemical-group and layout resamples for development. "
        "`confirmation/paired_identification_bounds.csv` contains signed contrasts that cancel shared missing objects. "
        "`confirmation/paired_intervals.csv` reports block-resampling sensitivity for each lower/upper bound endpoint. "
        "These conditional-on-fitted-model comparisons are not equivalence tests.", "",
        "`deployment_units.csv`, `selection_overlap.csv`, `compute_and_resource_costs.csv`, dataset `per_object.csv` "
        "files and confirmation `per_object.csv` retain all reportable quantities. Resource costs are per deployment "
        "cell; cross-validation fitting pools must not be summed as one physical campaign. Historical CORE/HistGB "
        "fitting costs are not zero because their saved predictions were reused.", "",
        "Fitted models, validation losses, role IDs, predictions and frozen new selections are saved under "
        "`runs/m3_amplitude_controls_20260922_v1/`. Original R4 artifacts are unchanged. Protocols are "
        "`protocols/m3_amplitude_controls_20260922.md` and its confirmation-dispersion extension.", "",
        "A fitted amplitude-only baseline tests the specified nonlinear predictor. It does not establish an upper "
        "bound over all possible one-dimensional amplitude functions. Descending profile norm is a defined heuristic; "
        "this experiment does not establish how often laboratories use it for biological hit confirmation.", ""]
    (ROOT/"REPORT.md").write_text("\n".join(lines))
    (ROOT/"method_identity.json").write_text(json.dumps(dict(
        status="confirmed", configuration="M3 HistGB 200 iterations, fixed 3-setting grid; exact original 60 cells",
        implementation="scikit-learn 1.9.1", authorization="user request 22 September 2026",
        confirmation_dispersion_frozen=final["frozen_at"],
        protocol="protocols/m3_amplitude_controls_20260922.md",
        extension="protocols/m3_amplitude_controls_20260922_confirmation_extension.md"), indent=2)+"\n")
    paper_source = PROJECT.parent/"OPAL2_NC_DATA_RICH_REVISION_20260921/source_data"
    exports = {
        "development_summary.csv": "m3_development_summary.csv",
        "paired_intervals.csv": "m3_development_paired_intervals.csv",
        "selection_overlap.csv": "m3_development_selection_overlap.csv",
        "deployment_units.csv": "m3_deployment_units.csv",
        "compute_and_resource_costs.csv": "m3_compute_and_resource_costs.csv",
        "confirmation/summary.csv": "m3_confirmation_summary.csv",
        "confirmation/paired_identification_bounds.csv": "m3_confirmation_paired_identification_bounds.csv",
        "confirmation/paired_intervals.csv": "m3_confirmation_paired_intervals.csv",
        "confirmation/selection_overlap.csv": "m3_confirmation_selection_overlap.csv",
        "confirmation/per_object.csv": "m3_confirmation_per_object.csv",
    }
    for dataset in ("EU", "JUMP", "LINCS", "RxRx3"):
        exports[f"{dataset}/per_object.csv"] = f"m3_{dataset}_per_object.csv"
    paper_source.mkdir(parents=True, exist_ok=True)
    for source, destination in exports.items():
        shutil.copy2(ROOT/source, paper_source/destination)
    (ROOT/"paper_source_exports.json").write_text(json.dumps(
        {source: str(paper_source/destination) for source, destination in exports.items()}, indent=2)+"\n")
    print(ROOT/"REPORT.md")


if __name__ == "__main__":
    main()
