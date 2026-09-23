"""Close R1 using already-open DEV measurements and saved predictions only.

No fitting, sampling, source downloads, cohort changes, or protected-data reads.
The common table describes a fixed endpoint and complete role reassignments;
it is not a ranking of datasets or a model-independent predictability bound.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_r1_completion_20260917 import observables, gamma_from_gram

OUT = ROOT / "reports/r1_closure_20260918_v1"
SOURCES = (
    ("JUMP source_5 DEV", "data/source5_primary_fullcontrols/measurements.npz", 639, 3617,
     "runs/gram_oof_20260914_v1/RIDGE_GEOMETRY_oof_predictions.npz"),
    ("LINCS Pilot1 DEV", "data/lincs_pilot1_biology_20260915/data.npz", 1188, 1694,
     "runs/lincs_empirical_radial_20260916_v1/AMP_EMP_LOCAL.npz"),
    ("EU FMP-HepG2 DEV", "reports/eu_core_development_20260917_v1/prepared_data_cc904/data.npz", 904, 2810,
     "runs/eu_core_cc904_20260917_v1/AMP_EMP_LOCAL.npz"),
)


def main():
    if (OUT / "summary.json").exists():
        print("R1 closure already saved; no repeated computation.")
        return
    started = time.monotonic()
    manifest = json.loads((ROOT / "data/source5_primary_fullcontrols/manifest.json").read_text())
    assert manifest["shape"] == [639, 4, 3617]
    assert manifest["fifth_repeat_read"] is False and manifest["old_final_opened"] is False
    eu_meta = json.loads((ROOT / "reports/eu_core_development_20260917_v1/prepared_data_cc904/metadata.json").read_text())
    assert eu_meta["n"] == 904 and eu_meta["confirmation_data_loaded"] is False
    rows, role_rows, audit = [], [], []
    for label, source, n, dim, prediction_path in SOURCES:
        with np.load(ROOT / source, allow_pickle=False) as z:
            ids, y = z["ids"], z["Y"]
            # JUMP groups are source/batch/plate indices, not chemical identities.
            chemical_groups = z["groups"] if z["groups"].ndim == 1 else None
        assert y.shape == (n, 4, dim) and np.isfinite(y).all()
        assert len(set(ids.tolist())) == n
        if label.startswith("EU"):
            with np.load(ROOT / "runs/r1_completion_20260917_v1/observables.npz", allow_pickle=False) as z:
                np.testing.assert_array_equal(ids, z["ids"])
                obs = {k: z[k] for k in ("gram", "W", "role_indices", "role_gamma")}
            observables_reused = True
        else:
            obs = observables(y)
            observables_reused = False
        gamma = gamma_from_gram(obs["gram"], 0, 3)
        direct_mean = y[:, :3].mean(1)
        def cos(a, b):
            return np.einsum("nd,nd->n", a, b) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
        direct = .5 * (cos(direct_mean, y[:, 3]) - cos(y[:, 0], y[:, 3])) - .02
        np.testing.assert_allclose(gamma, direct, rtol=1e-9, atol=1e-10)
        with np.load(ROOT / prediction_path, allow_pickle=False) as z:
            saved_ids = z["ids"]
            saved_actual = z["actual"]
            if saved_actual.ndim == 2:
                assert label.startswith("JUMP") and saved_actual.shape == (639, 3)
                saved_actual = saved_actual[:, 2]  # original ADD_TWO action
            lookup = {str(v): i for i, v in enumerate(saved_ids)}
            assert set(lookup) == set(ids.tolist())
            ordered = saved_actual[[lookup[str(v)] for v in ids]]
        np.testing.assert_allclose(gamma, ordered, rtol=1e-8, atol=1e-9)
        fixed = obs["role_gamma"][:, obs["role_indices"][:, 0] == 0]
        flip = (fixed <= 0).any(1) & (fixed > 0).any(1)
        rms = np.sqrt(np.mean(y[:, 0] ** 2, axis=1))
        rows.append(dict(
            dataset=label, n=n, coordinates=dim,
            chemical_groups=(int(len(np.unique(chemical_groups))) if chemical_groups is not None else None),
            mean_actual_gamma=float(gamma.mean()), sd_actual_gamma=float(gamma.std(ddof=1)),
            null_n=int((gamma <= 0).sum()), null_fraction=float((gamma <= 0).mean()),
            positive_n=int((gamma >= .005).sum()), gray_n=int(((gamma > 0) & (gamma < .005)).sum()),
            fixed_X_three_roles_any_null_flip_n=int(flip.sum()),
            fixed_X_three_roles_any_null_flip_fraction=float(flip.mean()),
            log_amplitude_log_W_descriptive_spearman=float(spearmanr(np.log(rms), np.log(obs["W"])).statistic),
        ))
        for j, (x, v) in enumerate(obs["role_indices"]):
            g = obs["role_gamma"][:, j]
            role_rows.append(dict(dataset=label, x=int(x), v=int(v), n=n,
                                  mean_gamma=float(g.mean()), null_n=int((g <= 0).sum()),
                                  flip_vs_primary_n=int(((g <= 0) != (gamma <= 0)).sum())))
        audit.append(dict(dataset=label, source=source, saved_actual=prediction_path,
                          ids_and_endpoint_match=True, eu_observables_reused=observables_reused,
                          max_endpoint_difference=float(np.max(np.abs(gamma - ordered)))))
    OUT.mkdir(parents=True, exist_ok=False)
    with (OUT / "common_observed_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    with (OUT / "role_sensitivity.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(role_rows[0])); writer.writeheader(); writer.writerows(role_rows)
    summary = dict(
        state="COMPLETE", purpose="R1 common descriptive closure; reuse completed mechanism evidence",
        endpoint="0.5*(cos(mean(X,Z1,Z2),V)-cos(X,V))-0.02",
        primary_roles=["X", "Z1", "Z2", "V"], null_definition="Gamma <= 0", positive_margin=.005,
        metrics=rows, endpoint_audit=audit, elapsed_seconds=time.monotonic() - started,
        fits_performed=0, monte_carlo_draws=0, protected_measurements_opened=False,
        scope="Already-open development objects, each existing fixed measurement space",
        cautions=["Role swaps change V and the acquired pair together; they are not independent replicates.",
                  "Observed amplitude-W association is descriptive, not cross-validated prediction skill.",
                  "JUMP source/batch/plate groups are not chemical-identity groups; unique chemical-group count not inferred.",
                  "Different preprocessing and populations preclude an information-ceiling or dataset-quality ranking."])
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
