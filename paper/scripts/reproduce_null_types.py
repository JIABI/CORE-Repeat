"""Decompose observed NULL outcomes without changing frozen R4 selections.

Run with the repository environment's Python. Inputs are released frozen
confirmation outputs; generated files go to paper/qa/null_types_20260925
(or OPAL2_QA_OUT/null_types_20260925). Missing outcomes remain untyped.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


from release_paths import DATA, RESEARCH, QA

OUT = QA / "null_types_20260925"
RUN = RESEARCH / "runs/r4_confirmation_20260921_v1"
SOURCES = {
    "completed_run": RUN / "complete.json",
    "query": RUN / "ingest/x/query.npz",
    "outcomes": RUN / "ingest/outcomes/outcomes.npz",
    "selections": RUN / "selections/selections.npz",
    "selected_lists": RUN / "selections/selected_lists.csv",
    "object_diagnostics": DATA / "layout_diagnostics_20260923/object_diagnostics.csv",
}
POLICIES = ("CORE", "HISTGB_CAL")
COST = 0.02


def rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def cosine(a, b):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.sum(a * b, axis=-1) / np.sqrt(
            np.sum(a * a, axis=-1) * np.sum(b * b, axis=-1)
        )


def save_csv(name, data):
    with (OUT / name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    state = json.loads(SOURCES["completed_run"].read_text())
    assert state["stage"] == "ALL_R4_COMPLETE" and state["selections_frozen"]
    source_rows = rows(SOURCES["object_diagnostics"])
    by_id = {row["object_id"]: row for row in source_rows}
    assert len(by_id) == len(source_rows) == 1539
    frozen_lists = rows(SOURCES["selected_lists"])
    with np.load(SOURCES["query"], allow_pickle=True) as q, np.load(
        SOURCES["outcomes"], allow_pickle=True
    ) as o, np.load(SOURCES["selections"], allow_pickle=True) as s:
        ids = q["ids"].astype(str)
        assert np.array_equal(ids, o["ids"].astype(str))
        assert np.array_equal(ids, s["ids"].astype(str))
        assert set(ids) == set(by_id)
        assert tuple(o["role_order"].astype(str)) == ("Z1", "Z2", "V")
        eligible = q["eligible"].astype(bool) & q["x_valid"].astype(bool)
        observed = eligible & o["valid"].astype(bool).all(axis=1)
        x = q["X"]
        z1, z2, verifier = (o["future"][:, j] for j in range(3))
        raw_increment = cosine((x + z1 + z2) / 3, verifier) - cosine(x, verifier)
        gamma = 0.5 * raw_increment - COST
        selected = {policy: s[policy + "__selected"].astype(bool) for policy in POLICIES}
        layout = q["layout"].astype(str)
    assert len(ids) == 1539 and eligible.sum() == 1527 and observed.sum() == 1520
    diagnostics = [by_id[object_id] for object_id in ids]
    old_gamma = np.array([float(row["gamma"]) if row["gamma"] else np.nan for row in diagnostics])
    assert np.array_equal(observed, np.isfinite(old_gamma))
    max_error = float(np.max(np.abs(gamma[observed] - old_gamma[observed])))
    assert max_error < 1e-12
    assert np.array_equal(layout, np.array([row["layout"] for row in diagnostics]))
    summaries, exceptions = [], []
    for policy in POLICIES:
        mask = selected[policy]
        assert mask.sum() == 192 and not np.any(mask & ~eligible)
        assert set(ids[mask]) == {row["object_id"] for row in frozen_lists if row["policy"] == policy}
        assert np.array_equal(mask, np.array([row[policy + "_selected"] == "True" for row in diagnostics]))
        known_null = mask & observed & (gamma <= 0)
        nonpositive = known_null & (gamma <= -COST)
        insufficient = known_null & (gamma > -COST)
        missing = mask & ~observed
        low_layout_null = known_null & np.isin(layout, ("B1004", "B1007"))
        both = np.array([
            bool(row["MEDINA_delta"] and row["USC_delta"])
            and float(row["MEDINA_delta"]) > 0 and float(row["USC_delta"]) > 0
            for row in diagnostics
        ])
        summaries.append(dict(
            policy=policy, selected=192, observed=int((mask & observed).sum()),
            nonpositive_cosine_increment=int(nonpositive.sum()),
            positive_increment_below_or_equal_cost=int(insufficient.sum()),
            known_null=int(known_null.sum()), positive_net_gain=int((mask & observed & (gamma > 0)).sum()),
            missing_untyped=int(missing.sum()), low_layout_known_null=int(low_layout_null.sum()),
            low_layout_null_both_external_sites_positive=int((low_layout_null & both).sum()),
        ))
        for i in np.flatnonzero(known_null | missing):
            row = diagnostics[i]
            category = (
                "missing_untyped" if missing[i] else
                "nonpositive_cosine_increment" if nonpositive[i] else
                "positive_increment_below_or_equal_cost"
            )
            exceptions.append(dict(
                policy=policy, object_id=ids[i], layout=layout[i], outcome_type=category,
                gamma=float(gamma[i]) if observed[i] else "",
                half_cosine_increment=float(gamma[i] + COST) if observed[i] else "",
                cosine_increment=float(raw_increment[i]) if observed[i] else "",
                MEDINA_delta=row["MEDINA_delta"], USC_delta=row["USC_delta"],
                both_sites_positive=row["both_sites_positive"],
            ))
    assert [(r["observed"], r["nonpositive_cosine_increment"], r["positive_increment_below_or_equal_cost"], r["known_null"], r["missing_untyped"]) for r in summaries] == [
        (190, 13, 3, 16, 2), (191, 4, 2, 6, 1)
    ]
    assert summaries[0]["low_layout_known_null"] == 8
    assert summaries[0]["low_layout_null_both_external_sites_positive"] == 7
    assert all(r["gamma"] < -COST for r in exceptions if r["policy"] == "CORE" and r["layout"] in ("B1004", "B1007"))
    save_csv("summary.csv", summaries)
    save_csv("selected_exception_objects.csv", exceptions)
    audit = dict(
        sources={key: str(value) for key, value in SOURCES.items()},
        run_state=state["stage"], normalization_population=1539,
        selected_per_policy=192, cost=COST,
        definition="Gamma = 0.5 * cosine_increment - 0.02; NULL iff Gamma <= 0",
        nonpositive_increment="Gamma <= -0.02, equivalent to cosine_increment <= 0",
        positive_cost_insufficient="-0.02 < Gamma <= 0, equivalent to 0 < cosine_increment <= 0.04",
        missing="Retained in the 192 selections, but no observed endpoint type assigned",
        max_absolute_gamma_difference_from_previous_source=max_error,
        scope="Descriptive decomposition of unchanged frozen selections; no model fitting or new selection",
    )
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"summary": summaries, "max_gamma_error": max_error}, indent=2))


if __name__ == "__main__":
    main()
