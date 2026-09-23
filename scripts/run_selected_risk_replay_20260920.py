"""Reuse frozen predictions for set-risk reporting and CAL-only policy tuning."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
import numpy as np
from threadpoolctl import threadpool_limits

_spec = importlib.util.spec_from_file_location(
    'previous_decision_run', PROJECT/'scripts/run_decision_region_calibration_20260920.py')
previous = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(previous)
from opal2.decision_region_calibration import ARMS, paired_cluster_interval
from opal2.selected_risk_replay import evaluate_cell
from opal2.selected_risk_statistics import summarize_risk_forecasts

INPUT = PROJECT/'runs/decision_region_calibration_20260920_v1'
OUTPUT = PROJECT/'runs/selected_risk_replay_20260920_v1'
REPORT = PROJECT/'reports/selected_risk_replay_20260920_v1'
DATASETS = ('EU', 'JUMP', 'LINCS', 'RxRx3')
POLICIES = ('FIXED_0', 'FIXED_02', 'CAL_TUNED')
SEED = 20260920


def jobs():
    result = []
    for dataset in DATASETS:
        for folder in sorted((INPUT/dataset).iterdir()):
            if folder.is_dir() and (folder/'complete.json').exists():
                result.append((dataset, folder))
    if len(result) != 60:
        raise ValueError('The complete 60-cell input is required')
    return result


def load_cell(folder):
    cell = previous.load_cache(folder)
    if 'cal_layout' not in cell:
        if cell['dataset'] != 'LINCS':
            raise ValueError('Missing saved CAL layout')
        with np.load(PROJECT/'runs/lincs_r2_completion_20260918_v1/CORE_ORIGINAL.npz',allow_pickle=False) as source:
            lookup = dict(zip(source['ids'].astype(str), source['layout'].astype(str)))
        cell['cal_layout'] = np.asarray([lookup[str(v)] for v in cell['cal_ids']])
    return cell


def score_policy(actual, selected, probability):
    actual = np.asarray(actual)
    selected = np.asarray(selected, bool)
    label = actual <= 0
    k = int(selected.sum())
    return dict(n=len(actual), k=k, actual_NULL=int(label[selected].sum()),
                predicted_NULL=float(np.asarray(probability)[selected].sum()),
                mean_selected_Gamma=float(actual[selected].mean()),
                total_Gamma=float(actual[selected].sum()),
                value_per_candidate=float((actual*selected).mean()),
                NULL_per_candidate=float((label*selected).mean()),
                FDP=float(label[selected].mean()))


def risk_cell_record(dataset, cell_name, arm, arrays):
    cs = arrays[arm+'__cal_selected_L02'].astype(bool)
    qs = arrays[arm+'__query_selected_FIXED_02'].astype(bool)
    cy = arrays['cal_actual'] <= 0
    qy = arrays['query_actual'] <= 0
    p = arrays[arm+'__query_p']
    b, c, k = int(cs.sum()), int(cy[cs].sum()), int(qs.sum())
    q, qj = c/b, (c+.5)/(b+1)
    observed = int(qy[qs].sum())
    result = dict(dataset=dataset, cell=cell_name, arm=arm,
                  n_cal=len(cy), cal_selected=b, cal_NULL=c,
                  cal_selected_groups=len(set(arrays['cal_groups'][cs])),
                  cal_selected_layouts=len(set(arrays['cal_layout'][cs])),
                  cal_boundary_events=c in (0,b),
                  query_selected=k, actual_NULL=observed)
    for name, prediction in [('BASE', p[qs]), ('EMPIRICAL', np.full(k,q)),
                             ('JEFFREYS',np.full(k,qj))]:
        forecast = float(prediction.sum())
        gap = observed-forecast
        result[name+'_predicted_NULL'] = forecast
        result[name+'_gap'] = gap
        result[name+'_rate_absolute_error'] = abs(gap)/k
        result[name+'_rate_squared_error'] = (gap/k)**2
        result[name+'_brier'] = float(np.mean((qy[qs]-prediction)**2))
    return result


def risk_stats_input(arrays, arm, name):
    return dict(cell=name, cal_null=(arrays['cal_actual']<=0).astype(float),
                cal_selected=arrays[arm+'__cal_selected_L02'],
                cal_groups=arrays['cal_groups'], cal_layout=arrays['cal_layout'],
                query_null=(arrays['query_actual']<=0).astype(float),
                query_selected=arrays[arm+'__query_selected_FIXED_02'],
                query_probability=arrays[arm+'__query_p'],
                query_groups=arrays['query_groups'], query_layout=arrays['query_layout'])


def analyse_dataset(dataset, cells, metadata):
    # Concatenate QUERY arrays only; CAL records repeat legally across cells.
    common = ('query_ids','query_groups','query_layout','query_actual')
    q = {key:np.concatenate([a[key] for _,a in cells]) for key in common}
    if len(set(q['query_ids'])) != len(q['query_ids']):
        raise ValueError('Duplicated outer query condition')
    if len(q['query_ids']) != previous.EXPECTED_N[dataset]:
        raise ValueError('Dataset is incomplete')
    policy, policy_cells, risk_cells, risk_summary, paired, uncertainty, mc = [],[],[],[],[],[],[]
    for arm in ARMS:
        for suffix in ('query_p','query_mean'):
            q[arm+'__'+suffix] = np.concatenate([a[arm+'__'+suffix] for _,a in cells])
        for mode in POLICIES:
            key=arm+'__query_selected_'+mode
            q[key]=np.concatenate([a[key] for _,a in cells])
            s=q[key].astype(bool)
            baseline=np.concatenate([a[arm+'__query_selected_FIXED_02'] for _,a in cells]).astype(bool)
            summary=score_policy(q['query_actual'],s,q[arm+'__query_p'])
            policy.append(dict(dataset=dataset,arm=arm,policy=mode,**summary,
                               replaced_vs_fixed02=int((s&~baseline).sum()),
                               additional_action_wells=2*summary['k']))
        for cell_name,a in cells:
            risk_cells.append(risk_cell_record(dataset,cell_name,arm,a))
            for mode in POLICIES:
                s=a[arm+'__query_selected_'+mode].astype(bool)
                baseline=a[arm+'__query_selected_FIXED_02'].astype(bool)
                policy_cells.append(dict(dataset=dataset,cell=cell_name,arm=arm,policy=mode,
                    **score_policy(a['query_actual'],s,a[arm+'__query_p']),
                    replaced_vs_fixed02=int((s&~baseline).sum()),
                    applied_lambda=(0. if mode=='FIXED_0' else .2 if mode=='FIXED_02'
                                    else metadata[cell_name]['arms'][arm]['chosen_lambda']),
                    chosen_lambda=metadata[cell_name]['arms'][arm]['chosen_lambda']))
        arm_risk=[r for r in risk_cells if r['arm']==arm]
        total_k=sum(r['query_selected'] for r in arm_risk)
        for estimator in ('BASE','EMPIRICAL','JEFFREYS'):
            predicted=sum(r[estimator+'_predicted_NULL'] for r in arm_risk)
            observed=sum(r['actual_NULL'] for r in arm_risk)
            risk_summary.append(dict(dataset=dataset,arm=arm,estimator=estimator,k=total_k,
                predicted_NULL=predicted,actual_NULL=observed,gap=observed-predicted,
                absolute_pooled_gap=abs(observed-predicted),
                weighted_cell_rate_MAE=sum(r['query_selected']*r[estimator+'_rate_absolute_error'] for r in arm_risk)/total_k,
                weighted_cell_rate_MSE=sum(r['query_selected']*r[estimator+'_rate_squared_error'] for r in arm_risk)/total_k,
                selected_brier=sum(r['query_selected']*r[estimator+'_brier'] for r in arm_risk)/total_k,
                CAL_selected_appearances=sum(r['cal_selected'] for r in arm_risk),
                CAL_selected_NULL_appearances=sum(r['cal_NULL'] for r in arm_risk),
                cells_with_zero_or_all_CAL_events=sum(r['cal_boundary_events'] for r in arm_risk)))
        for block in ('groups','layout'):
            inputs=[risk_stats_input(a,arm,name) for name,a in cells]
            stats=summarize_risk_forecasts(inputs,block=block,replicates=2000,seed=SEED)
            uncertainty.append(dict(dataset=dataset,arm=arm,block=block,statistics=stats))
            for control in ('FIXED_0','FIXED_02'):
                snew=q[arm+'__query_selected_CAL_TUNED'].astype(float)
                sold=q[arm+'__query_selected_'+control].astype(float)
                for metric,values in [('value_per_candidate',q['query_actual']),
                                       ('NULL_per_candidate',(q['query_actual']<=0).astype(float))]:
                    ci=paired_cluster_interval((snew-sold)*values,q['query_'+block],
                                               replicates=2000,seed=SEED)
                    paired.append(dict(dataset=dataset,arm=arm,contrast='CAL_TUNED minus '+control,
                                       metric=metric,block=block,**ci))
    # Both original and new policies are compared to the same strong interfaces.
    for arm in ARMS[1:]:
        for mode in POLICIES:
            delta=q[arm+'__query_selected_'+mode].astype(float)-q['CORE__query_selected_'+mode].astype(float)
            for block in ('groups','layout'):
                for metric,values in [('value_per_candidate',q['query_actual']),
                                       ('NULL_per_candidate',(q['query_actual']<=0).astype(float))]:
                    paired.append(dict(dataset=dataset,arm=arm,contrast=arm+' minus CORE '+mode,
                        metric=metric,block=block,**paired_cluster_interval(delta*values,q['query_'+block],
                                                                 replicates=2000,seed=SEED)))
    # Existing integration seeds only; chosen lambda is not retuned.
    for mode in POLICIES:
        primary=q['CORE__query_selected_'+mode].astype(bool)
        for seed_index in (0,1):
            key='CORE__query_selected_MC'+str(seed_index)+'_'+mode
            if all(key in a for _,a in cells):
                mask=np.concatenate([a[key] for _,a in cells]).astype(bool)
                probabilities=np.concatenate([a['CORE__query_seed_p'][seed_index] for _,a in cells])
                mc.append(dict(dataset=dataset,policy=mode,seed_index=seed_index,
                    **score_policy(q['query_actual'],mask,probabilities),
                    replaced_vs_primary=int((mask&~primary).sum()),
                    note='Existing seed m/p and frozen chosen lambda; no new draw or retuning.'))
    previous.save_npz(OUTPUT/dataset/'all_query_predictions.npz',q)
    return dict(policy=policy,policy_cells=policy_cells,risk_cells=risk_cells,risk=risk_summary,
                paired=paired,risk_uncertainty=uncertainty,mc=mc)


def render_report(data,elapsed):
    lines=['# Selected-set risk reporting and CAL-selected lambda', '',
           'All four development datasets and 60 cells completed. This run reused frozen predictions; no new model fitting, Monte Carlo draws, reference acquisition or confirmation measurements.', '',
           '## Same original CORE list: risk reporting', '',
           '| Dataset | Estimator | Predicted NULL | Actual NULL | Cell-weighted absolute rate error |',
           '|---|---|---:|---:|---:|']
    for r in data['risk']:
        if r['arm']=='CORE':
            lines.append('| %s | %s | %.3f | %d | %.4f |'%(r['dataset'],r['estimator'],r['predicted_NULL'],r['actual_NULL'],r['weighted_cell_rate_MAE']))
    lines += ['', 'The pooled count can look accurate while cell errors cancel. Read the cell-weighted absolute error and its paired resampling alongside it. EMPIRICAL and JEFFREYS forecast the selected set, not each object\'s conditional probability.', '',
              '## Allocation: unchanged m and p, CAL-only lambda choice', '',
              '| Dataset | Interface | Policy | NULL / selected | Mean realized Gamma | Replaced vs 0.2 |',
              '|---|---|---|---:|---:|---:|']
    for r in data['policy']:
        lines.append('| %s | %s | %s | %d / %d | %.8f | %d |'%(r['dataset'],r['arm'],r['policy'],r['actual_NULL'],r['k'],r['mean_selected_Gamma'],r['replaced_vs_fixed02']))
    lines += ['', '## Interpretation', '',
              'Lambda selection maximizes CAL realized Gamma. It is not probability calibration, NULL-count matching or an imposed purity constraint. Fixed 0 and fixed 0.2 are retained even if a data-chosen policy scores better on development queries.', '',
              'Risk resampling recomputes each CAL set-rate with shared chemical-group/layout weights across CAL and QUERY appearances. Policy intervals condition on fixed fitted scores and selected lambda; they do not refit the full model or give top-k finite-sample guarantees. Few layouts and zero-event CAL sets limit what bootstrap intervals can establish.', '',
              'Files retain all five interfaces, per-cell budgets/choices, CAL support, paired risk and policy intervals, lists, and existing MC seed stability. The current source archives have already been used for development; these results do not constitute R4.', '',
              'No automatic change to CORE, BIO or representation activation follows this run. Gamma distribution scores and assay costs remain those of R2. Computation wall-clock elapsed time: %.2f seconds.'%elapsed]
    (REPORT/'REPORT.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports-only',action='store_true',help='Refresh presentation fields from saved results without computation')
    args=parser.parse_args()
    if args.reports_only:
        summary=json.loads((REPORT/'summary.json').read_text())
        data=summary['results']
        for row in data['policy_cells']:
            row['applied_lambda']=(0. if row['policy']=='FIXED_0' else .2 if row['policy']=='FIXED_02'
                                   else row['chosen_lambda'])
        previous.write_csv(REPORT/'policy_cells.csv',data['policy_cells'])
        previous.write_json(REPORT/'summary.json',summary)
        render_report(data,summary['elapsed_seconds'])
        print('Reports refreshed from saved numerical results; no repeated experiment.')
        return
    OUTPUT.mkdir(parents=True,exist_ok=True)
    REPORT.mkdir(parents=True,exist_ok=True)
    if not (REPORT/'PROTOCOL.md').exists():
        raise RuntimeError('Declare the protocol before running')
    start=time.perf_counter()
    manifest=dict(version=1,input=str(INPUT),lambda_grid=[0.,.1,.2,.4],objective='CAL selected mean realized Gamma',
                  risk_estimators=['BASE','EMPIRICAL','JEFFREYS'],datasets=list(DATASETS),
                  new_model_fits=0,new_MC_draws=0,protected_measurements_opened=False)
    previous.write_json(OUTPUT/'manifest.json',manifest)
    all_data={k:[] for k in ('policy','policy_cells','risk_cells','risk','paired','risk_uncertainty','mc')}
    metadata=[]
    with threadpool_limits(limits=1):
        for dataset in DATASETS:
            cells=[];cell_metadata={}
            for _,folder in [j for j in jobs() if j[0]==dataset]:
                cell=load_cell(folder)
                saved=previous.load_npz(folder/'query_predictions.npz')
                arrays,meta=evaluate_cell(cell,saved)
                arrays['cal_layout']=np.asarray(cell['cal_layout']).astype(str)
                destination=OUTPUT/dataset/folder.name
                previous.save_npz(destination/'replay_predictions.npz',arrays)
                previous.write_json(destination/'decisions.json',meta)
                cells.append((folder.name,arrays));cell_metadata[folder.name]=meta
                metadata.append({**meta,'dataset':dataset,'cell':folder.name})
                status=dict(state='RUNNING',stage='REPLAY',dataset=dataset,cell=folder.name,
                            completed_cells=len(metadata),total_cells=60,elapsed=time.perf_counter()-start)
                previous.write_json(OUTPUT/'status.json',status)
            previous.write_json(OUTPUT/'status.json',dict(state='RUNNING',stage='UNCERTAINTY',dataset=dataset,
                                completed_cells=len(metadata),total_cells=60,elapsed=time.perf_counter()-start))
            result=analyse_dataset(dataset,cells,cell_metadata)
            for key,value in result.items():all_data[key].extend(value)
            print(json.dumps(dict(dataset=dataset,state='COMPLETE',cells=len(cells),
                                  elapsed=time.perf_counter()-start)),flush=True)
    for key in ('policy','policy_cells','risk_cells','risk','paired','mc'):
        previous.write_csv(REPORT/(key+'.csv'),all_data[key])
    previous.write_json(REPORT/'risk_uncertainty.json',all_data['risk_uncertainty'])
    previous.write_json(REPORT/'cell_decisions.json',metadata)
    elapsed=time.perf_counter()-start
    previous.write_json(REPORT/'summary.json',dict(**manifest,completed_cells=len(metadata),complete=True,
                                                elapsed_seconds=elapsed,results=all_data))
    render_report(all_data,elapsed)
    status=dict(state='COMPLETE',completed_cells=len(metadata),datasets=list(DATASETS),
                elapsed_seconds=elapsed,report=str(REPORT/'REPORT.md'),updated=previous.now())
    previous.write_json(OUTPUT/'status.json',status)
    print(json.dumps(status),flush=True)


if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        previous.write_json(OUTPUT/'status.json',dict(state='FAILED',error=str(exc),
                            traceback=traceback.format_exc(),updated=previous.now()))
        raise
