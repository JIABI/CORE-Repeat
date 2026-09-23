"""Read-only checkpoint audit of original RxRx3 right branches.

No retraining, no protected evaluation data, no change to saved checkpoints.
The output separates missing-channel reparameterization from removal and checks
the optimization history against the claim that the right branch never moved.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import joblib
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from opal2.dual_branch_biology import BOUND


def describe(values):
    a = np.asarray(values, float)
    return dict(min=float(a.min()), median=float(np.median(a)),
                mean=float(a.mean()), max=float(a.max()))


def main():
    torch.set_num_threads(1)
    run = ROOT / "runs/r3_rxrx3_modules_20260920_v1"
    output = ROOT / "runs/r3_biology_support_pilot_20260920_v1/audit_training.json"
    cells, all_right, all_removed_delta = [], [], []
    for index in range(40):
        folder = run / f"fold_{index}"
        model = joblib.load(folder / "BIO_STRUCTURED.joblib")
        report = json.loads((folder / "BIO_STRUCTURED_training.json").read_text())
        predictions = np.load(folder / "module_predictions.npz", allow_pickle=False)
        records = np.load(folder / "adapter_reference/records.npz", allow_pickle=False)
        right = model.right
        names = right.names
        moa = np.array([name.startswith("moa_") for name in names])
        supported = predictions["query_support"]
        biology = predictions["query_biology"]
        assert np.all(biology[:, moa] == 0)
        assert np.all(records["biology_values"][:, moa] == 0)
        assert np.all(model.biological_center.numpy()[moa] == 0)
        history = report["history"]
        with torch.no_grad():
            raw = torch.as_tensor(biology, dtype=torch.float64)
            basis = right.basis(raw, (raw-model.biological_center)/model.biological_scale)
            local = (basis * right.local_coefficients).sum(-1)
            preactivation = right.readout(local)
            moa_contribution = local[:, moa] @ right.readout.weight[:, moa].T
            assert torch.max(torch.abs(moa_contribution-moa_contribution[:1])) < 1e-14
            original = BOUND*torch.tanh(preactivation/BOUND)
            removed = BOUND*torch.tanh((preactivation-moa_contribution)/BOUND)
            # Same parameters, target channels only, exact algebraic bias
            # absorption. Floating summation order can differ at roundoff.
            absorbed = (local[:, ~moa] @ right.readout.weight[:, ~moa].T
                        + right.readout.bias + moa_contribution[0])
            restored = BOUND*torch.tanh(absorbed/BOUND)
            original[~supported] = 0
            removed[~supported] = 0
            restored[~supported] = 0
            recorded = predictions["BIO_STRUCTURED_query"]-predictions["GELU_query"]
            forward_error = float(np.max(np.abs(original.numpy()-recorded)))
            restoration_error = float(torch.max(torch.abs(original-restored)))
            assert forward_error < 2e-14 and restoration_error < 2e-14
            all_right.append(original.numpy()[supported])
            all_removed_delta.append((removed-original).numpy()[supported])
            row = dict(
                cell=index, fitting_supported_n=report["fitting_supported_n"],
                query_supported_n=int(supported.sum()), query_n=len(supported),
                projection_nll_epoch1=history[0]["training_projection_nll"],
                projection_nll_epoch60=history[-1]["training_projection_nll"],
                nll_reduction_epoch1_to60=history[0]["training_projection_nll"]-history[-1]["training_projection_nll"],
                nll_reduction_epoch50_to60=history[-2]["training_projection_nll"]-history[-1]["training_projection_nll"],
                final_readout_weight_norm=float(torch.linalg.vector_norm(right.readout.weight)),
                final_readout_bias_norm=float(torch.linalg.vector_norm(right.readout.bias)),
                final_local_coefficient_norm=float(torch.linalg.vector_norm(right.local_coefficients)),
                final_preclip_gradient_norm=history[-1]["preclip_gradient_norm_mean"],
                all_recorded_gradient_clipping_fractions=[h["clipped_step_fraction"] for h in history],
                learning_rate_epoch60=history[-1]["learning_rate"],
                train_right_increment_rms_epoch1=history[0]["right_increment_rms"],
                train_right_increment_rms_epoch60=history[-1]["right_increment_rms"],
                query_right_increment_rms_supported=float(torch.sqrt(original[supported].square().mean())),
                query_right_nonzero_n=int(torch.any(original != 0, dim=1).sum()),
                missing_moa_nonzero_local_fields=[names[j] for j in np.flatnonzero(moa)
                    if bool(torch.any(local[:, j] != 0))],
                missing_moa_preactivation_constant=moa_contribution[0].tolist(),
                missing_moa_constant_rms=float(torch.sqrt(moa_contribution[0].square().mean())),
                removal_without_bias_absorption_output_delta_rms_supported=float(torch.sqrt((removed[supported]-original[supported]).square().mean())),
                absorption_output_max_abs_error=restoration_error,
                saved_forward_max_abs_error=forward_error,
                history=history)
            cells.append(row)
    right = np.concatenate(all_right)
    removed = np.concatenate(all_removed_delta)
    fields = ["fitting_supported_n", "nll_reduction_epoch1_to60",
              "nll_reduction_epoch50_to60", "final_readout_weight_norm",
              "final_readout_bias_norm", "final_local_coefficient_norm",
              "final_preclip_gradient_norm", "train_right_increment_rms_epoch60",
              "query_right_increment_rms_supported", "missing_moa_constant_rms",
              "removal_without_bias_absorption_output_delta_rms_supported"]
    aggregate = {field: describe([row[field] for row in cells]) for field in fields}
    aggregate.update(
        cells=len(cells), training_loss_improved_cells=sum(row["nll_reduction_epoch1_to60"] > 0 for row in cells),
        nonzero_final_readout_cells=sum(row["final_readout_weight_norm"] > 0 for row in cells),
        query_n=sum(row["query_n"] for row in cells),
        supported_query_n=sum(row["query_supported_n"] for row in cells),
        nonzero_right_query_n=sum(row["query_right_nonzero_n"] for row in cells),
        supported_right_increment_rms=float(np.sqrt(np.mean(right**2))),
        supported_missing_moa_removal_rms=float(np.sqrt(np.mean(removed**2))),
        maximum_bias_absorption_error=max(row["absorption_output_max_abs_error"] for row in cells),
        maximum_saved_forward_error=max(row["saved_forward_max_abs_error"] for row in cells),
        any_recorded_gradient_clipping=any(any(row["all_recorded_gradient_clipping_fractions"]) for row in cells))
    report = dict(
        mode="numeric-audit", source_run=str(run), aggregate=aggregate, cells=cells,
        interpretation={
            "averaging_order": "Each donor residual is whitened/projected and squared before weights @ energies. No mean-vector-before-square implementation bug.",
            "jensen": "Energy(weighted mean)-weighted mean energy is nonpositive and equals negative weighted dispersion; compare normalized/aligned versions against matched random donors, not sign alone.",
            "optimization": "Saved right readouts and outputs are nonzero; compare epoch1-to60 losses. Small end-epoch changes occur under a learning-rate schedule reaching exactly zero and do not prove convergence or absence of signal.",
            "moa": "Twelve zero-valued MoA fields do not yield twelve active constants: four uncertainty fields have nonzero constant bases. Their learned contribution is a constant pre-tanh offset absorbable into existing bias without changing predictions (roundoff aside). Removing without absorption changes a fitted model, and retraining a smaller model is not a diagnostic of semantic contamination.",
            "invariants": "All-unsupported right output and whole-adapter-off invariants remain valid. Per-relation missingness has no explicit mask in this historical right branch.",
            "no_training": True, "protected_evaluation_read": False})
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(aggregate, indent=2))
    print(output)


if __name__ == "__main__":
    main()
