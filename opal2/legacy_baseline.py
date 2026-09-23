"""Run the existing full three-head OPAL adapter on the new fixed DEV split.

The old adapter is imported read-only; no original pipeline or results are
changed. Its morphology selection, clipping, 400 trees/head, chemistry hashing
and metadata features are retained, not replaced by the new neural encoder.
"""
from pathlib import Path
import importlib.util
import sys

import joblib
import numpy as np
import pandas as pd

from .source5 import _legacy_loader
from .evaluation import actual_utilities, gain_metrics, write_json


def run_legacy_baseline(dataset, splits, directory, legacy_root, *, seed=20260911, threads=4):
    root = Path(legacy_root).resolve()
    legacy = _legacy_loader(root).load_development(dataset.metadata["space"])
    if not np.array_equal(legacy["ids"], dataset.ids) or not np.array_equal(legacy["Y"], dataset.Y):
        raise ValueError("Legacy baseline and new method must use identical IDs and fixed-space measurements")
    scripts = root / "scripts"
    old_path = list(sys.path)
    old_bytecode_setting = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(scripts))
    try:
        path = scripts / "source5_opal_tunable.py"
        spec = importlib.util.spec_from_file_location("source5_opal_tunable", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = old_bytecode_setting
        sys.path[:] = old_path
    train, evaluation = splits["train"], splits["evaluation"]
    gain_train = actual_utilities(dataset, train).samples[0, :, -1]
    metadata = legacy["initial_metadata"]
    model = module.fit_model(dataset.Y[train, 0], metadata.iloc[train], gain_train,
                             feature_names=dataset.feature_names, seed=seed, n_jobs=threads)
    prediction = module.predict_model(model, dataset.Y[evaluation, 0], metadata.iloc[evaluation],
                                       feature_names=dataset.feature_names)
    actual = actual_utilities(dataset, evaluation).samples[0, :, -1]
    metrics = gain_metrics(actual, prediction["pred_gain"], prediction["p_null"], prediction["p_positive"])
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    joblib.dump(model, directory / "three_heads.joblib")
    pd.DataFrame({"compound_id": dataset.ids[evaluation], "actual_gamma": actual,
                  "predicted_gamma": prediction["pred_gain"], "p_null": prediction["p_null"],
                  "p_positive": prediction["p_positive"]}).to_csv(directory / "predictions.tsv", sep="\t", index=False)
    report = {"evidence_scope": "same previously-opened DEV compound split, not certification",
              "model": "original full source5 OPAL adapter, default C00",
              "train_compounds": len(train), "evaluation_compounds": len(evaluation),
              "gain_metrics": metrics, "adapter_info": prediction["info"],
              "training_ids": dataset.ids[train], "evaluation_ids": dataset.ids[evaluation],
              "note": "Same endpoint and training split; original information bundle, not equal-information neural ablation"}
    write_json(directory / "evaluation.json", report)
    return report
