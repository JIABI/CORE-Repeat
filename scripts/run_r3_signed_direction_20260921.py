"""Reuse all35 paired-dose Ridge fits; test signed and directional borrowing."""
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

from opal2.crossdose_response import fit_convex_strength, profile_scores
from opal2.signed_direction_borrowing import (DIRECTION_GRID, source_cosine,
    cosine_modulated_weights, transport_candidate, effective_support,
    apply_borrowing, fit_direction_strength, matched_signed_random)
from opal2.rxrx3_r3_biology import load_rxrx3_biology_metadata
from opal2.biology_kernel_evaluation import write_json

OLD = PROJECT/'runs/r3_crossdose_response_20260920_v1'
QUALIFICATION = PROJECT/'reports/r3_crossdose_response_20260920_v1/qualification.npz'
ROOT = PROJECT/'runs/r3_signed_program_dose_20260921_v1/borrowing'
REPORT = PROJECT/'reports/r3_signed_program_dose_20260921_v1/borrowing'
DATA = PROJECT/'data/rxrx3_r2_20260918/prepared_r2/data.npz'
FAMILIES = ('TARGET', 'TARGET_POSCOS', 'TARGET_SIGNED', 'MORPH')
MODES = ('FREE', 'DIRECTION')
STRENGTHS = ('CAL', 'A025')
TASKS = ('SAME', 'CROSS')
METRICS = ('profile_mse', 'cosine_loss', 'lognorm_squared_error')
REPLICATES = 20
SEED = 2026092103


def read_npz(path):
    with np.load(path, allow_pickle=False) as f:
        return {k:f[k] for k in f.files}


def arm_names():
    names = ['RIDGE_RESPONSE']
    for family in FAMILIES:
        for rep in range(REPLICATES + 1):
            label = family if rep == 0 else family + f'_R{rep:02d}'
            names.extend(f'{label}_{mode}_{strength}' for mode in MODES for strength in STRENGTHS)
        names.extend(f'{family}_RANDOM_MEAN_{mode}_{strength}' for mode in MODES for strength in STRENGTHS)
    return names


def prepare():
    config = dict(dataset='RxRx3 approved development', units=35, tasks=list(TASKS),
        source_models=str(OLD), qualification=str(QUALIFICATION), source_data=str(DATA),
        newly_fitted_parameters='CAL alpha only; no Ridge/CORE refit', random_replicates=REPLICATES,
        seed=SEED, family_order=list(FAMILIES), mode_order=list(MODES), arms=arm_names(),
        metric_names=list(METRICS), target_cosine='source X only',
        target_poscos='original top16 TARGET weights times max(source cosine,0); positive sum normalization',
        target_signed='original top16 TARGET weights times source cosine; absolute L1 normalization',
        candidate='X_query + W @ (Y_reference - X_reference)',
        free='baseline + alpha*(candidate-baseline)',
        direction='project candidate-baseline tangent to baseline; move; normalize to baseline norm',
        direction_grid=list(DIRECTION_GRID),
        direction_cal='smallest grid alpha within paired supported-group one SE of minimum CAL MSE',
        free_cal='original supported-group continuous quadratic minimum and one-SE admission',
        random_matching='exact count and signed coefficient multiset within TRAIN amplitude bin and exact source-X same-plate relation',
        random_target_pool='annotated references, different chemical group; no plate/bin relaxation',
        exact_fallback='alpha=0, unsupported, or baseline norm <=1e-12',
        primary_comparisons=['TARGET_SIGNED_FREE_CAL versus TARGET/POSCOS/MORPH and matched random',
            'DIRECTION versus its corresponding FREE'],
        evaluation_masks=['original TARGET support', 'full cohort', 'source dose', 'source batch'],
        zero_norm_score_convention='cosine=0 if either norm<=1e-12; log(max(norm,1e-12))',
        unchanged=['SAME/CROSS qualification and outcomes', 'old Ridge weights', 'old real top16 reference plans'],
        gamma_endpoint_tested=False, protected_measurements_opened=False)
    for path in (ROOT, REPORT):
        path.mkdir(parents=True, exist_ok=True)
        destination = path/'METHOD_CONFIG.json'
        if destination.exists():
            if json.loads(destination.read_text()) != config:
                raise ValueError('Method configuration changed after start')
        else:
            write_json(destination, config)
    comparisons=[]
    for family in FAMILIES:
        for strength in STRENGTHS:
            for mode in MODES:
                name=f'{family}_{mode}_{strength}'
                comparisons.extend([[name,'RIDGE_RESPONSE'],
                    [name,f'{family}_RANDOM_MEAN_{mode}_{strength}']])
            comparisons.append([f'{family}_DIRECTION_{strength}',f'{family}_FREE_{strength}'])
    for other in ('TARGET','TARGET_POSCOS','MORPH'):
        for mode in MODES:
            for strength in STRENGTHS:
                comparisons.append([f'TARGET_SIGNED_{mode}_{strength}',f'{other}_{mode}_{strength}'])
    write_json(ROOT/'comparisons.json',comparisons)
    if not (ROOT/Path(__file__).name).exists():
        shutil.copy2(__file__, ROOT/Path(__file__).name)
        shutil.copy2(PROJECT/'opal2/signed_direction_borrowing.py', ROOT/'signed_direction_borrowing.py')


def score_array(prediction, target):
    result = profile_scores(prediction, target)
    return np.stack([result[k] for k in METRICS], axis=1), result['prediction_zero_norm'], result['target_zero_norm']


def run_unit(number, data, pairing, annotated):
    folder, old = ROOT/f'unit_{number:02d}', OLD/f'unit_{number:02d}'
    folder.mkdir(exist_ok=True)
    if (folder/'complete.json').exists():
        return json.loads((folder/'complete.json').read_text())
    started = time.monotonic()
    def progress(stage):
        write_json(folder/'status.json', dict(state='RUNNING', unit=number, stage=stage,
                   elapsed_seconds=time.monotonic()-started))
    progress('reusing qualification and source-only reference plans')
    fold, task_index = divmod(number, 7)
    refs, models, old_scores = (read_npz(old/name) for name in
        ('references.npz', 'ridge_models.npz', 'query_scores.npz'))
    rows = refs['source_pair_rows']
    np.testing.assert_array_equal(rows, np.flatnonzero(pairing['task_index'] == task_index))
    source, target = pairing['source_rows'][rows], pairing['target_rows'][rows]
    groups, ids = pairing['groups'][rows], pairing['object_ids'][rows]
    c, q, r = (refs[k] for k in ('cal_rows', 'query_rows', 'reference_rows'))
    cq, nc = np.r_[c,q], len(c)
    for local, role in ((c,'DIST_CAL'),(q,'DEV_EVAL'),(r,'REF_FIT')):
        assert np.all(pairing['outer_roles'][rows[local],fold] == role)
    np.testing.assert_array_equal(rows[q], old_scores['pair_rows'])
    np.testing.assert_array_equal(ids[q], old_scores['object_ids'])
    x = data['Y'][source,0]
    # Identical metadata-selected endpoints to the completed response experiment.
    outcomes = (
        data['Y'][source[:,None], pairing['source_roles'][rows]].mean(axis=1),
        data['Y'][target[:,None], pairing['target_roles'][rows]].mean(axis=1))
    plate = data['plates'][source,0]
    same_plate = plate[cq,None] == plate[None,r]
    legal = groups[cq,None] != groups[None,r]
    cos = source_cosine(x[cq], x[r])
    real = {family:refs[family+'_weights'][0].copy() for family in ('TARGET','MORPH')}
    supports = {family:refs[family+'_support'].copy() for family in ('TARGET','MORPH')}
    for family, signed in (('TARGET_POSCOS',False),('TARGET_SIGNED',True)):
        real[family], supports[family] = cosine_modulated_weights(real['TARGET'],cos,signed=signed)
    logamp = np.log(np.maximum(np.linalg.norm(x[r],axis=1),1e-12))
    plan_arrays = dict(source_pair_rows=rows, cal_rows=c, query_rows=q, reference_rows=r,
        donor_bins=refs['donor_bins'], amplitude_bin_edges=refs['amplitude_bin_edges'],
        reference_ids=ids[r], reference_groups=groups[r], query_ids=ids[q], cal_ids=ids[c],
        query_source_plate=plate[q], reference_source_plate=plate[r])
    plans, audits = {}, {}
    for family_no, family in enumerate(FAMILIES):
        w, support = real[family], supports[family]
        eligible = legal & (annotated[source[r]][None,:] if family.startswith('TARGET') else True)
        plans[family] = [w]
        audits[family] = []
        for rep in range(1,REPLICATES+1):
            random, audit = matched_signed_random(w, eligible, refs['donor_bins'],
                same_plate, logamp, seed=SEED+number*10000+family_no*1000+rep)
            plans[family].append(random)
            record = dict(rep=rep, amplitude_bin_merges=audit['amplitude_bin_merges'],
                source_plate_relations_relaxed=audit['source_plate_relations_relaxed'],
                maximum_stratum_absolute_mass_error=audit['maximum_stratum_absolute_mass_error'],
                maximum_stratum_signed_mass_error=audit['maximum_stratum_signed_mass_error'])
            for split, slc in (('cal',slice(0,nc)),('query',slice(nc,None))):
                active = support[slc]
                record[split] = {name:float(value[slc][active].mean()) if active.any() else 0.
                    for name,value in audit.items() if isinstance(value,np.ndarray)}
                record[split]['original_same_plate_absolute_mass'] = float(
                    (np.abs(w[slc])*same_plate[slc]).sum(1)[active].mean()) if active.any() else 0.
                record[split]['random_same_plate_absolute_mass'] = float(
                    (np.abs(random[slc])*same_plate[slc]).sum(1)[active].mean()) if active.any() else 0.
            audits[family].append(record)
        plan_arrays[family+'_weights'] = np.stack(plans[family])
        plan_arrays[family+'_support'] = support
        plan_arrays[family+'_absolute_ess'] = np.divide(1.,np.square(w).sum(1),
            out=np.zeros(len(w)),where=support)
        plan_arrays[family+'_same_plate_absolute_mass'] = (np.abs(w)*same_plate).sum(1)
    np.savez_compressed(folder/'references.npz',**plan_arrays)
    write_json(folder/'random_matching_audit.json',audits)
    names = arm_names()
    scores = np.empty((2,len(q),len(names),3),float)
    prediction_zero = np.zeros((2,len(q),len(names)),bool)
    target_zero = np.zeros((2,len(q)),bool)
    predictions, calibration, baseline_checks = {}, {}, {}
    direction_norm_error = 0.
    for task_no,(task,y) in enumerate(zip(TASKS,outcomes)):
        progress(task+' reuse frozen Ridge predictions')
        base = ((x[cq]-models[task+'_input_center'])/models[task+'_input_scale']) @ models[task+'_coefficient'] + models[task+'_target_center']
        scores[task_no,:,0],prediction_zero[task_no,:,0],target_zero[task_no] = score_array(base[nc:],y[q])
        old_base = old_scores['scores'][task_no,:,0]
        error = float(np.max(np.abs(scores[task_no,:,0]-old_base)))
        np.testing.assert_allclose(scores[task_no,:,0],old_base,atol=1e-11,rtol=1e-12)
        baseline_checks[task] = dict(maximum_score_absolute_difference=error,
            scores_bitwise_equal=bool(np.array_equal(scores[task_no,:,0],old_base)),
            reused_coefficients=True, refitted=False)
        predictions[task+'_RIDGE_RESPONSE'] = base[nc:].astype(np.float32)
        predictions[task+'_actual'] = y[q].astype(np.float32)
        calibration[task] = {}
        arm = 1
        for family in FAMILIES:
            random_indices = {mode+'_'+strength:[] for mode in MODES for strength in STRENGTHS}
            support = effective_support(base,supports[family])
            for rep,w in enumerate(plans[family]):
                progress(task+' '+family+f' draw {rep}/{REPLICATES}')
                candidate = transport_candidate(x[cq],x[r],y[r],w,supports[family])
                label = family if rep == 0 else family+f'_R{rep:02d}'
                for mode in MODES:
                    fit = (fit_convex_strength(base[:nc],y[c],candidate[:nc],groups[c],support[:nc])
                        if mode == 'FREE' else
                        fit_direction_strength(base[:nc],y[c],candidate[:nc],groups[c],support[:nc]))
                    calibration[task][label+'_'+mode] = fit
                    for strength,alpha in (('CAL',fit['alpha']),('A025',.25)):
                        prediction = apply_borrowing(base[nc:],candidate[nc:],support[nc:],alpha,mode)
                        np.testing.assert_array_equal(prediction[~support[nc:]],base[nc:][~support[nc:]])
                        if alpha == 0:
                            np.testing.assert_array_equal(prediction,base[nc:])
                        if mode == 'DIRECTION':
                            discrepancy = np.abs(np.linalg.norm(prediction,axis=1)-np.linalg.norm(base[nc:],axis=1))
                            direction_norm_error = max(direction_norm_error,float(discrepancy.max(initial=0)))
                            np.testing.assert_allclose(np.linalg.norm(prediction,axis=1),np.linalg.norm(base[nc:],axis=1),rtol=1e-12,atol=1e-12)
                        scores[task_no,:,arm],prediction_zero[task_no,:,arm],_ = score_array(prediction,y[q])
                        assert names[arm] == f'{label}_{mode}_{strength}'
                        if rep:
                            random_indices[mode+'_'+strength].append(arm)
                        elif strength == 'CAL':
                            predictions[task+'_'+label+'_'+mode+'_CAL'] = prediction.astype(np.float32)
                        arm += 1
            for mode in MODES:
                for strength in STRENGTHS:
                    selected = random_indices[mode+'_'+strength]
                    scores[task_no,:,arm] = np.mean(scores[task_no][:,selected,:],axis=1)
                    prediction_zero[task_no,:,arm] = np.any(prediction_zero[task_no][:,selected],axis=1)
                    assert names[arm] == f'{family}_RANDOM_MEAN_{mode}_{strength}'
                    arm += 1
        assert arm == len(names)
    shared = {k:old_scores[k] for k in ('pair_rows','object_ids','groups','source_ids','target_ids',
        'source_dose','target_dose','batch','source_plate','layout','target_support','morph_support')}
    np.savez_compressed(folder/'query_scores.npz',scores=scores,arm_names=np.asarray(names),
        metric_names=np.asarray(METRICS),task_names=np.asarray(TASKS),
        prediction_zero_norm=prediction_zero,target_zero_norm=target_zero,
        poscos_support=supports['TARGET_POSCOS'][nc:],signed_support=supports['TARGET_SIGNED'][nc:],**shared)
    np.savez_compressed(folder/'predictions.npz',source_x=x[q].astype(np.float32),**predictions)
    write_json(folder/'calibration.json',calibration)
    result = dict(state='COMPLETE',unit=number,fold=fold,dose_pair_index=task_index,
        source_dose=float(pairing['source_dose'][rows[0]]),target_dose=float(pairing['target_dose'][rows[0]]),
        query_n=len(q),query_groups=len(np.unique(groups[q])),baseline_reproduction=baseline_checks,
        maximum_direction_norm_absolute_error=direction_norm_error,
        support_counts={f:dict(cal_rows=int(supports[f][:nc].sum()),query_rows=int(supports[f][nc:].sum()),
            cal_groups=len(np.unique(groups[c][supports[f][:nc]])),query_groups=len(np.unique(groups[q][supports[f][nc:]]))) for f in FAMILIES},
        exact_fallback_verified=True,random_matching_complete=True,random_replicates=REPLICATES,
        zero_target_rows=int(target_zero.sum()),prediction_zero_entries=int(prediction_zero.sum()),
        elapsed_seconds=time.monotonic()-started)
    write_json(folder/'complete.json',result)
    write_json(folder/'status.json',result)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--units',nargs='*',type=int)
    parser.add_argument('--threads',type=int,default=2)
    args=parser.parse_args()
    start=time.monotonic()
    prepare()
    data,pairing=read_npz(DATA),read_npz(QUALIFICATION)
    biology=load_rxrx3_biology_metadata(data)
    annotated=biology['arrays']['target_mask']
    units=range(35) if args.units is None else args.units
    try:
        with threadpool_limits(limits=args.threads):
            for number in units:
                result=run_unit(number,data,pairing,annotated)
                finished=len(list(ROOT.glob('unit_*/complete.json')))
                write_json(ROOT/'status.json',dict(state='RUNNING',finished_units=finished,total_units=35,
                    last_unit=number,last_unit_seconds=result['elapsed_seconds'],pid=os.getpid(),
                    elapsed_seconds=time.monotonic()-start))
                print(json.dumps(dict(unit=number,seconds=result['elapsed_seconds'],finished=finished)),flush=True)
        complete=len(list(ROOT.glob('unit_*/complete.json')))
        write_json(ROOT/'status.json',dict(state='NUMERICS_COMPLETE' if complete==35 else 'PARTIAL',
            finished_units=complete,total_units=35,pid=os.getpid(),elapsed_seconds=time.monotonic()-start))
    except Exception:
        write_json(ROOT/'status.json',dict(state='FAILED',traceback=traceback.format_exc(),
            elapsed_seconds=time.monotonic()-start))
        raise


if __name__=='__main__':
    main()
