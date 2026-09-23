"""Development-only nested comparisons of the complete, configured models.

Each inner experiment trains its own encoder, scaler, reference templates and
library bank. Outer predictions are out of fold, not one frozen-rule certificate.
No realized outer outcomes participate in architecture or budget selection.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold

from .training import fit_model, load_model
from .evaluation import predict_utilities, actual_utilities, allocation_observed, gain_metrics, write_json
from .planning import allocate_actions
from .splits import development_split


def _folds(indices, count, seed, groups=None):
    indices = np.asarray(indices, int)
    if groups is None:
        iterator = KFold(count, shuffle=True, random_state=seed).split(indices)
    else:
        iterator = GroupKFold(count).split(indices, groups=np.asarray(groups)[indices])
    for train, test in iterator:
        yield indices[train], indices[test]


def _fit_partition(dataset, pool, heldout, seed, groups=None):
    """Split fitting population before preprocessing; no reuse of heldout IDs."""
    pool = np.asarray(pool, int)
    if groups is None:
        chunks = np.array_split(np.random.default_rng(seed).permutation(pool), 10)
        train, validation, calibration = np.sort(np.concatenate(chunks[:8])), np.sort(chunks[8]), np.sort(chunks[9])
    else:
        g = np.asarray(groups)
        unique = np.unique(g[pool])
        if len(unique) < 3:
            raise ValueError("At least three fitting groups needed for train/validation/calibration")
        ordered = np.random.default_rng(seed).permutation(unique)
        n = len(ordered)
        nt = max(1, min(n-2, int(.8*n)))
        nv = max(1, min(n-nt-1, int(.1*n)))
        train = pool[np.isin(g[pool], ordered[:nt])]
        validation = pool[np.isin(g[pool], ordered[nt:nt+nv])]
        calibration = pool[np.isin(g[pool], ordered[nt+nv:])]
    if min(len(train), len(validation), len(calibration), len(heldout)) < 1:
        raise ValueError("Insufficient data for a genuine four-way inner experiment")
    union = np.sort(np.concatenate([pool, heldout]))
    lookup = {int(old): new for new, old in enumerate(union)}
    split = {name: np.array([lookup[int(x)] for x in ix]) for name, ix in
             (("train", train), ("validation", validation), ("calibration", calibration), ("evaluation", heldout))}
    return dataset.subset(union), split


def _evaluate_frozen(model, scaler, dataset, split, config, fractions):
    ix = split["evaluation"]
    predicted, measurement = predict_utilities(model, scaler, dataset, ix, config)
    actual = actual_utilities(dataset, ix)
    rows, records = [], []
    for fraction in fractions:
        budget = 2 * int(np.ceil(fraction * len(ix)))
        allocation = allocate_actions(predicted, budget)
        rows.append(allocation_observed(actual, allocation.action_indices,
                                       label=f"budget_{fraction:g}", budget=budget))
        for j, i in enumerate(ix):
            action = allocation.action_indices[j]
            records.append({"compound_id": dataset.ids[i], "fraction": fraction,
                            "action": int(action), "actual_net_gain": actual.samples[0,j,action],
                            "predicted_net_gain": predicted.mean[j,action],
                            "actual_add_two": actual.samples[0,j,-1],
                            "predicted_add_two": predicted.mean[j,-1],
                            "p_null_add_two": predicted.p_null[j,-1],
                            "p_positive_add_two": predicted.p_positive[j,-1]})
    return rows, records, measurement


def nested_development_cv(dataset, configurations, directory, *, outer_folds=5, inner_folds=3,
                          fractions=(.05,.1,.25), seed=20260911, groups=None):
    """Select by prespecified mean realized policy value across inner budgets.

    This function trains the full configuration each time. Epoch limits and
    stopping are the supplied scientific configuration, not reduced for CV.
    It never changes a cost, event definition or contract to improve a result.
    """
    if not configurations or len(set(configurations)) != len(configurations):
        raise ValueError("A nonempty named configuration family is required")
    if any(not 0 < f <= 1 for f in fractions):
        raise ValueError("Budget fractions must lie in (0,1]")
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"evidence": "DEVELOPMENT_NESTED_CV_NOT_CERTIFICATION",
                "selection": "mean realized net gain across declared inner budgets; lexical tie break",
                "outer_folds":outer_folds,"inner_folds":inner_folds,"fractions":fractions,
                "configurations":{name:asdict(cfg.validate()) for name,cfg in configurations.items()},
                "ids":dataset.ids.tolist(),"groups":None if groups is None else list(map(str,groups)),
                "seed":seed,"cost_per_well":.01,"original_final_opened":False}
    write_json(output/"fold_manifest.json",manifest)
    all_records, outer_reports = [], []
    all_indices = np.arange(len(dataset))
    for outer, (pool, heldout) in enumerate(_folds(all_indices,outer_folds,seed,groups)):
        scores = {name:[] for name in configurations}
        if len(configurations) > 1:
            for inner, (inner_pool, inner_test) in enumerate(_folds(pool,inner_folds,seed+outer+1,groups)):
                inner_ds, split = _fit_partition(dataset,inner_pool,inner_test,seed+inner+101,groups)
                for name,cfg in configurations.items():
                    path = output/f"outer_{outer}"/f"inner_{inner}"/name
                    fit_model(inner_ds,split,cfg,path)
                    model,scaler,reloaded,_ = load_model(path)
                    rows,_,_ = _evaluate_frozen(model,scaler,inner_ds,split,reloaded,fractions)
                    score = float(np.mean([r["population_mean_net_gain"] for r in rows]))
                    scores[name].append(score)
                    write_json(path/"inner_policy_selection.json",{"score":score,"rows":rows})
            selected = min(configurations,key=lambda name:(-np.mean(scores[name]),name))
        else:
            selected = next(iter(configurations))
        outer_ds, split = _fit_partition(dataset,pool,heldout,seed+1001+outer,groups)
        path = output/f"outer_{outer}"/"selected"
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(output/f"outer_{outer}"/"selection.json",{"chosen":selected,"inner_scores":scores,
                   "outer_ids_not_used":dataset.ids[heldout].tolist()})
        fit_model(outer_ds,split,configurations[selected],path)
        model,scaler,cfg,_ = load_model(path)
        rows,records,measurement = _evaluate_frozen(model,scaler,outer_ds,split,cfg,fractions)
        all_records.extend({**record,"outer_fold":outer,"selected_config":selected} for record in records)
        outer_reports.append({"fold":outer,"selected":selected,"inner_scores":scores,
                              "policy_values":rows,"measurement":measurement})
        pd.DataFrame(all_records).to_csv(output/"cross_validation_predictions.tsv",sep="\t",index=False)
        write_json(output/"cross_validation.json",{"manifest":manifest,"outer_results":outer_reports,
                   "formal_certificate":False,"interpretation":"folds share training data and may share environments"})
    return outer_reports
