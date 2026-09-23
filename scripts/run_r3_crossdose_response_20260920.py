"""Paired same/cross-dose response borrowing, without changing CORE or Gamma."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
import numpy as np
from threadpoolctl import threadpool_limits

from opal2.crossdose_response import (response_candidates, apply_response_correction,
    fit_convex_strength, profile_scores, fit_ridge_response)
from opal2.reference_information_memory import (cosine_relationship,
    morphology_similarity, normalized_topk_weights)
from opal2.biology_random_reference_experiment import amplitude_bins, matched_random_weights
from opal2.rxrx3_r3_biology import load_rxrx3_biology_metadata
from opal2.biology_kernel_evaluation import write_json

ROOT = PROJECT/'runs/r3_crossdose_response_20260920_v1'
REPORT = PROJECT/'reports/r3_crossdose_response_20260920_v1'
METRICS = ('profile_mse', 'cosine_loss', 'lognorm_squared_error')
TASKS = ('SAME', 'CROSS')
FAMILIES = ('TARGET', 'MORPH')
KINDS = ('DIRECT', 'TRANSPORT')
STRENGTHS = ('CAL', 'A025', 'A100')
REPLICATES = 20
SEED = 2026092009


def read_npz(path):
    with np.load(path, allow_pickle=False) as file:
        return {k: file[k] for k in file.files}


def arm_names():
    result = ['RIDGE_RESPONSE', 'TRAIN_MEAN', 'SOURCE_X']
    for family in FAMILIES:
        for rep in range(REPLICATES+1):
            label = family if rep == 0 else family+f'_R{rep:02d}'
            result += [f'{label}_{kind}_{strength}' for kind in KINDS for strength in STRENGTHS]
    return result


def representatives(rows, groups, ids):
    selected = {}
    for row in sorted(rows, key=lambda i: str(ids[i])):
        selected.setdefault(str(groups[row]), int(row))
    return np.asarray(list(selected.values()), int)


def score_array(prediction, target):
    scored = profile_scores(prediction, target)
    return np.stack([scored[k] for k in METRICS], axis=1)


def prepare():
    ROOT.mkdir(parents=True, exist_ok=True)
    spec = dict(dataset='RxRx3 approved development only', tasks=list(TASKS), dose_pairs=7,
        outer_folds=5, coordinate_dimension=951, replicas=REPLICATES, seed=SEED,
        arms=arm_names(), metrics=list(METRICS), primary='TARGET_TRANSPORT_CAL',
        frozen_core_changed=False, protected_measurements_opened=False,
        gamma_endpoint_tested=False, support_rule='positive source-side top16 relation',
        reference_representative='lexicographic source identity per connectivity group',
        strengths=list(STRENGTHS), source=str(PROJECT/'data/rxrx3_r2_20260918/prepared_r2/data.npz'))
    if (ROOT/'run_manifest.json').exists():
        if json.loads((ROOT/'run_manifest.json').read_text()) != spec:
            raise ValueError('Existing manifest differs')
        if (ROOT/'PROTOCOL.md').read_bytes() != (REPORT/'PROTOCOL.md').read_bytes():
            raise ValueError('Protocol changed after start')
    else:
        write_json(ROOT/'run_manifest.json', spec)
        shutil.copy2(REPORT/'PROTOCOL.md', ROOT/'PROTOCOL.md')
        shutil.copy2(__file__, ROOT/Path(__file__).name)
    return spec


def run_unit(number, data, pairing, biology):
    fold, task_index = divmod(number, 7)
    folder = ROOT/f'unit_{number:02d}'
    folder.mkdir(exist_ok=True)
    if (folder/'complete.json').exists():
        return json.loads((folder/'complete.json').read_text())
    started = time.monotonic()
    def progress(stage):
        write_json(folder/'status.json', dict(state='RUNNING', unit=number, stage=stage,
            elapsed_seconds=time.monotonic()-started))
    progress('building paired tasks')
    rows = np.flatnonzero(pairing['task_index'] == task_index)
    source, target = pairing['source_rows'][rows], pairing['target_rows'][rows]
    groups = pairing['groups'][rows]
    ids = pairing['object_ids'][rows]
    roles = pairing['outer_roles'][rows, fold]
    part = {r: np.flatnonzero(roles == r) for r in
        ('TRAIN', 'VALIDATION', 'REF_FIT', 'DIST_CAL', 'DEV_EVAL')}
    for a, left in part.items():
        for b, right in part.items():
            if a != b and set(groups[left]) & set(groups[right]):
                raise ValueError('Chemical-group role overlap')
    part['REF_FIT'] = representatives(part['REF_FIT'], groups, ids)
    t, v, r, c, q = (part[k] for k in ('TRAIN','VALIDATION','REF_FIT','DIST_CAL','DEV_EVAL'))
    x = data['Y'][source, 0]
    y_same = data['Y'][source[:, None], pairing['source_roles'][rows]].mean(1)
    y_cross = data['Y'][target[:, None], pairing['target_roles'][rows]].mean(1)
    ys = (y_same, y_cross)
    # Query outcomes enter only score_array. Their rows never fit a parameter.
    cq = np.r_[c, q]
    nc = len(c)
    amplitude = np.log(np.maximum(np.linalg.norm(x, axis=1), 1e-12))
    donor_bins, bin_edges = amplitude_bins(amplitude[t], amplitude[r])
    legal = groups[cq, None] != groups[None, r]
    source_target = biology['arrays']['target'][source]
    source_mask = biology['arrays']['target_mask'][source]
    target_sim = cosine_relationship(source_target[cq], source_target[r], source_mask[cq], source_mask[r])
    morph_sim = morphology_similarity(x[cq], x[r])
    plans = {}
    plan_arrays = dict(query_ids=ids[q], cal_ids=ids[c], reference_ids=ids[r],
        reference_groups=groups[r], source_pair_rows=rows, cal_rows=c, query_rows=q,
        reference_rows=r, donor_bins=donor_bins, amplitude_bin_edges=bin_edges)
    source_plate = data['plates'][source, 0]
    for family, sim in zip(FAMILIES, (target_sim, morph_sim)):
        retrieved = normalized_topk_weights(sim, donor_ids=ids[r], top_k=16,
            eligible=legal, query_groups=groups[cq], donor_groups=groups[r])
        w, support = retrieved['weights'], retrieved['support']
        w[~support] = 0.
        random_legal = legal & (source_mask[r][None] if family == 'TARGET' else True)
        if np.any((w > 0) & ~random_legal):
            raise ValueError('Positive weights outside random legal pool')
        weights = [w]
        diags = []
        for rep in range(1, REPLICATES+1):
            wr, audit = matched_random_weights(w, random_legal, donor_bins, amplitude[r],
                seed=SEED+number*10000+FAMILIES.index(family)*1000+rep)
            weights.append(wr)
            diags.append(dict(rep=rep,
                retained_weight_mass=float(audit['retained_weight_mass'][support].mean()) if support.any() else 0.,
                log_amplitude_displacement=float(audit['weighted_log_amplitude_displacement'][support].mean()) if support.any() else 0.,
                query_same_plate_weight=float((wr[nc:]*(source_plate[q,None] == source_plate[None,r])).sum(1)[support[nc:]].mean()) if support[nc:].any() else 0.))
        plans[family] = dict(weights=weights, support=support, random_audit=diags,
            cal_supported=int(support[:nc].sum()), query_supported=int(support[nc:].sum()),
            cal_supported_groups=len(np.unique(groups[c][support[:nc]])),
            query_supported_groups=len(np.unique(groups[q][support[nc:]])))
        plan_arrays[family+'_weights'] = np.stack(weights)
        plan_arrays[family+'_support'] = support
        plan_arrays[family+'_ess'] = np.divide(1., np.square(w).sum(1),
            out=np.zeros(len(w)), where=support)
        plan_arrays[family+'_same_plate_weight'] = (w*(source_plate[cq,None] == source_plate[None,r])).sum(1)
    np.savez_compressed(folder/'references.npz', **plan_arrays)
    names = arm_names()
    scores = np.empty((2, len(q), len(names), len(METRICS)))
    saved_predictions = {}
    saved_models = {}
    fitted = {}
    calibration = {}
    for task_no, (task, y) in enumerate(zip(TASKS, ys)):
        progress(task+' ridge fitting')
        model = fit_ridge_response(x[t], y[t], x[v], y[v],
            training_groups=groups[t], validation_groups=groups[v])
        base = model.predict(x[cq])
        fitted[task] = model.report
        for field in ('input_center','input_scale','target_center','coefficient'):
            saved_models[task+'_'+field] = getattr(model, field)
        scores[task_no,:,0] = score_array(base[nc:], y[q])
        scores[task_no,:,1] = score_array(np.broadcast_to(model.target_center, y[q].shape), y[q])
        scores[task_no,:,2] = score_array(x[q], y[q])
        saved_predictions[task+'_RIDGE_RESPONSE'] = base[nc:].astype(np.float32)
        saved_predictions[task+'_actual'] = y[q].astype(np.float32)
        arm_no = 3
        calibration[task] = {}
        for family in FAMILIES:
            support = plans[family]['support']
            for rep, weights in enumerate(plans[family]['weights']):
                progress(task+' '+family+f' reference draw {rep}/{REPLICATES}')
                candidates = response_candidates(x[cq], x[r], y[r], weights, support)
                label = family if rep == 0 else family+f'_R{rep:02d}'
                for kind in KINDS:
                    candidate = candidates[kind]
                    fit = fit_convex_strength(base[:nc], y[c], candidate[:nc], groups[c], support[:nc])
                    calibration[task][label+'_'+kind] = fit
                    for strength, alpha in zip(STRENGTHS, (fit['alpha'], .25, 1.)):
                        prediction = apply_response_correction(base[nc:], candidate[nc:], support[nc:], alpha)
                        np.testing.assert_array_equal(prediction[~support[nc:]], base[nc:][~support[nc:]])
                        if alpha == 0:
                            np.testing.assert_array_equal(prediction, base[nc:])
                        scores[task_no,:,arm_no] = score_array(prediction, y[q])
                        assert names[arm_no] == f'{label}_{kind}_{strength}'
                        if rep == 0 and strength == 'CAL':
                            saved_predictions[task+'_'+label+'_'+kind] = prediction.astype(np.float32)
                        arm_no += 1
        assert arm_no == len(names)
    np.savez_compressed(folder/'query_scores.npz', scores=scores, arm_names=np.asarray(names),
        metric_names=np.asarray(METRICS), task_names=np.asarray(TASKS), pair_rows=rows[q],
        object_ids=ids[q], groups=groups[q], source_ids=pairing['source_ids'][rows[q]],
        target_ids=pairing['target_ids'][rows[q]], source_dose=pairing['source_dose'][rows[q]],
        target_dose=pairing['target_dose'][rows[q]], batch=data['batches'][source[q]],
        source_plate=source_plate[q], layout=data['layout'][source[q]],
        target_support=plans['TARGET']['support'][nc:], morph_support=plans['MORPH']['support'][nc:])
    np.savez_compressed(folder/'predictions.npz', source_x=x[q].astype(np.float32), **saved_predictions)
    np.savez_compressed(folder/'ridge_models.npz', **saved_models)
    write_json(folder/'calibration.json', calibration)
    write_json(folder/'ridge_fitting.json', fitted)
    support_report = {k:{a:b for a,b in p.items() if a not in ('weights','support')} for k,p in plans.items()}
    result = dict(state='COMPLETE', unit=number, fold=fold, dose_pair_index=task_index,
        source_dose=float(pairing['source_dose'][rows[0]]), target_dose=float(pairing['target_dose'][rows[0]]),
        query_n=len(q), query_groups=len(np.unique(groups[q])), counts={k:len(v) for k,v in part.items()},
        supports=support_report, elapsed_seconds=time.monotonic()-started,
        exact_fallback_verified=True, tasks=list(TASKS), scoring_complete=True)
    write_json(folder/'complete.json', result)
    write_json(folder/'status.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--units', nargs='*', type=int)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    start = time.monotonic()
    prepare()
    data = read_npz(PROJECT/'data/rxrx3_r2_20260918/prepared_r2/data.npz')
    pairing = read_npz(REPORT/'qualification.npz')
    biology = load_rxrx3_biology_metadata(data)
    write_json(ROOT/'biology_audit.json', biology['report'])
    units = list(range(35)) if args.units is None else args.units
    try:
        with threadpool_limits(limits=args.threads):
            for number in units:
                result = run_unit(number, data, pairing, biology)
                finished = len(list(ROOT.glob('unit_*/complete.json')))
                write_json(ROOT/'status.json', dict(state='RUNNING', finished_units=finished,total_units=35,
                    last_unit=number, last_unit_seconds=result['elapsed_seconds'],pid=os.getpid(),
                    elapsed_seconds=time.monotonic()-start))
                print(json.dumps(dict(unit=number,seconds=result['elapsed_seconds'],finished=finished)),flush=True)
        complete = len(list(ROOT.glob('unit_*/complete.json'))) == 35
        write_json(ROOT/'status.json', dict(state='NUMERICS_COMPLETE' if complete else 'PARTIAL',
            finished_units=len(list(ROOT.glob('unit_*/complete.json'))), total_units=35,
            elapsed_seconds=time.monotonic()-start, pid=os.getpid()))
    except Exception:
        write_json(ROOT/'status.json', dict(state='FAILED', traceback=traceback.format_exc(),
            elapsed_seconds=time.monotonic()-start))
        raise


if __name__ == '__main__':
    main()
