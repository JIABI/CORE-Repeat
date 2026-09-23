"""Post-hoc description of fitted fold noise, without refitting or removing rows."""
import argparse
import json
from pathlib import Path

import numpy as np

from .baseline_experiment import write_json


def summarize(root):
    root = Path(root)
    config = json.loads((root / "config.json").read_text())
    traced_object = "JCP2022_107356"
    rows = []
    for fold in config["folds"]:
        name = f"repeat_{fold['repeat']}_fold_{fold['fold']}"
        path = root / "stability_baseline" / name
        if not (path / "policy.json").exists():
            continue
        with np.load(path / "model.npz", allow_pickle=False) as params:
            basis = params["basis"]
            noise = params["residual_var"] + np.sum((basis @ params["within_cov"]) * basis, axis=1)
            raw_noise_sd = np.sqrt(noise) * params["scale"]
            maximum_scale = float(params["scale"].max())
        policy = json.loads((path / "policy.json").read_text())
        rows.append(dict(fold=name,
            traced_object_in_train=traced_object in fold["train_ids"],
            maximum_training_scale=maximum_scale,
            maximum_original_coordinate_noise_sd=float(raw_noise_sd.max()),
            predicted_gamma_mean={m["action"]: m["predicted_mean"] for m in policy["action_metrics"]},
            actual_gamma_mean={m["action"]: m["actual_mean"] for m in policy["action_metrics"]}))
    result = dict(analysis_script=str(Path(__file__).resolve()),
        scope="post-hoc saved-model diagnostic, no refit, intervention, row deletion or threshold change",
        traced_object=traced_object,
        selection_reason="largest previously observed calibration squared-error contributor",
        limitation="other training objects change across folds; association is not a single-factor causal test",
        complete=len(rows) == len(config["folds"]), rows=rows)
    write_json(root / "tail_diagnostics.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    print(json.dumps(summarize(parser.parse_args().root), indent=2))
