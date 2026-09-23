"""Read-only nested gate-development audit; does not run fitting or sampling."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def grouped_summary(values, groups):
    groups = np.asarray(groups, str)
    grouped = np.array([np.asarray(values)[groups==group].mean() for group in np.unique(groups)])
    return dict(group_equal_mean=float(grouped.mean()), group_count=len(grouped),
                descriptive_group_se=float(grouped.std(ddof=1)/np.sqrt(len(grouped))) if len(grouped)>1 else None)


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def audit(root):
    root = Path(root)
    paths = sorted(root.glob('cell_*/gate_development.json'))
    population = None
    if (root/'CORE.npz').exists():
        with np.load(root/'CORE.npz', allow_pickle=False) as core:
            population = dict(zip(core['ids'].astype(str),core['groups'].astype(str)))
    rows, pooled = [], {}
    for path in paths:
        report = json.loads(path.read_text())
        require(not report['query_outcomes_used'], 'Gate declares query outcomes used')
        require(not report['world_model_retrained'], 'Gate declares core refitting')
        cal_ids, cal_groups = set(report['calibration_ids']), set(report['calibration_groups'])
        cell_path = path.parent/'cell.json'
        if cell_path.exists():
            cell = json.loads(cell_path.read_text())
            require(cal_ids == set(cell['calibration_ids']), 'Gate calibration IDs differ from cell')
            require(not cal_ids & set(cell['query_ids']), 'QUERY IDs entered gate fit')
            require(not cal_ids & set(cell['model_fit_ids']), 'Gate CAL IDs entered mean fit')
            if population is not None:
                actual_cal_groups = {population[name] for name in cell['calibration_ids']}
                query_groups = {population[name] for name in cell['query_ids']}
                fit_groups = {population[name] for name in cell['model_fit_ids']}
                require(actual_cal_groups == cal_groups, 'Stored calibration group labels disagree')
                require(not actual_cal_groups & query_groups, 'QUERY chemistry entered gate/reference fit')
                require(not fit_groups & (actual_cal_groups|query_groups), 'Core FIT chemistry overlaps CAL/QUERY')
        for channel, data in report['channels'].items():
            for outer in data['outer_ledger']:
                outer_fit, outer_query = set(outer['fit_ids']), set(outer['query_ids'])
                require(not outer_fit & outer_query, 'Nested outer ID leakage')
                require(outer_fit | outer_query == cal_ids, 'Nested outer population changed')
                require(not set(outer['fit_groups']) & set(outer['query_groups']), 'Nested outer group leakage')
                for inner in outer['inner_oof']:
                    require(set(inner['fit_ids']) | set(inner['query_ids']) == outer_fit,
                            'Inner gate population includes outer query or omits outer fit')
                    require(not set(inner['fit_groups']) & set(inner['query_groups']), 'Inner group leakage')
            final = data['final_training_records']
            nested = data['nested_records']
            require(set(final['ids']) == cal_ids == set(nested['ids']), 'Training or nested population changed')
            support = np.asarray(final['supported'], bool)
            nested_support = np.asarray(nested['supported'], bool)
            groups = np.asarray(final['groups'], str)
            final_delta = np.asarray(final['candidate_deltas'])
            group_mean = np.mean([final_delta[groups==group].mean(0) for group in np.unique(groups)],axis=0)
            require(np.allclose(group_mean,data['final_fixed_candidate_group_mean'],atol=1e-12,rtol=0),
                    'Fixed-alpha candidate means cannot be reproduced')
            require(report['alphas'][np.argmax(group_mean)]==data['final_fixed_alpha'],
                    'Fixed alpha did not follow calibration-only group mean and declared ties')
            for estimator in ('final_ridge','final_boost'):
                require(data[estimator]['supported_rows']==int(support.sum()),'Supported training count mismatch')
                require(data[estimator]['supported_groups']==len(np.unique(groups[support])),
                        'Supported training group count mismatch')
                require((data[estimator]['fallback_reason'] is not None)==(len(np.unique(groups[support]))<4),
                        'Fallback contradicts declared minimum-group condition')
            true_delta = np.asarray(nested['actual_candidate_deltas'])
            require(np.array_equal(true_delta[:, 0], np.zeros(len(true_delta))), 'Alpha-zero delta not zero')
            require(not np.any(true_delta[~nested_support]), 'Unsupported delta nonzero')
            row = dict(cell=path.parent.name, channel=channel, calibration_n=len(cal_ids),
                calibration_groups=len(cal_groups), supported_training_rows=int(support.sum()),
                supported_training_groups=len(np.unique(groups[support])),
                final_fixed_alpha=data['final_fixed_alpha'],
                ridge_fallback=data['final_ridge']['fallback_reason'],
                boost_fallback=data['final_boost']['fallback_reason'],
                outer_supported_training_groups=[part['ridge']['supported_groups'] for part in data['outer_ledger']],
                outer_fallback_folds=sum(part['ridge']['fallback_reason'] is not None for part in data['outer_ledger']),
                final_boost_actual_splits=data['final_boost'].get('total_actual_splits'),
                nested_supported_rows=int(nested_support.sum()), strategies={})
            for strategy, values in nested['strategies'].items():
                predicted = np.asarray(values['predicted_delta'])
                alpha = np.asarray(values['alpha'])
                selected = np.searchsorted(report['alphas'], alpha)
                realized = np.asarray(values['realized_delta'])
                require(np.array_equal(np.asarray(report['alphas'])[np.argmax(predicted, axis=1)], alpha),
                        'Gate alpha does not maximize predicted delta with stable ties')
                require(np.array_equal(realized, true_delta[np.arange(len(alpha)), selected]),
                        'Nested realized delta mismatches fixed-alpha score')
                require(not np.any(alpha[~nested_support]), 'Unsupported nested alpha nonzero')
                supplied = data['nested_diagnostics'][strategy]['logscore_gain']['mean']
                require(abs(float(realized.mean())-supplied) < 1e-12, 'Nested mean cannot be reproduced')
                row['strategies'][strategy] = dict(mean_delta=float(realized.mean()),
                    supported_mean_delta=float(realized[nested_support].mean()) if nested_support.any() else None,
                    predicted_selected_mean=float(predicted[np.arange(len(alpha)), selected].mean()),
                    active_rows=int((alpha>0).sum()),
                    supported_r2=data['nested_diagnostics'][strategy]['prediction']['supported']['r2'])
                pooled.setdefault((channel,strategy),[]).append(dict(ids=final['ids'],groups=groups,
                    delta=realized, predicted=predicted[np.arange(len(alpha)), selected],
                    support=nested_support, alpha=alpha,candidates=true_delta))
            rows.append(row)
    aggregate = {}
    for (channel,strategy), parts in pooled.items():
        ids = [name for part in parts for name in part['ids']]
        delta = np.concatenate([part['delta'] for part in parts])
        prediction = np.concatenate([part['predicted'] for part in parts])
        support = np.concatenate([part['support'] for part in parts])
        alpha = np.concatenate([part['alpha'] for part in parts])
        groups = np.concatenate([part['groups'] for part in parts])
        candidates = np.concatenate([part['candidates'] for part in parts])
        fixed_selected = np.concatenate([part['delta'] for part in pooled[(channel,'FIXED_SELECTED')]])
        aggregate.setdefault(channel,{})[strategy] = dict(records=len(ids),unique_ids=len(set(ids)),
            mean_delta=float(delta.mean()),mean_predicted_selected_delta=float(prediction.mean()),
            supported_n=int(support.sum()),supported_mean_delta=float(delta[support].mean()) if support.any() else None,
            active_n=int((alpha>0).sum()),
            per_cell_mean=[float(part['delta'].mean()) for part in parts],
            vs_core=grouped_summary(delta,groups),
            vs_calibration_selected_fixed=grouped_summary(delta-fixed_selected,groups),
            vs_always_fixed={str(alpha_value):grouped_summary(delta-candidates[:,j],groups)
                             for j,alpha_value in enumerate((0.,.25,.5,1.))})
    return dict(status='PASS' if len(paths)==10 else 'PARTIAL_PASS',completed_gate_cells=len(paths),
                outer_chemical_group_artifact_check=population is not None,
                cells=rows,aggregate=aggregate,
                interpretation='Nested calibration development only; correlated donors/models, no inferential certificate; '
                               'query Gamma/Brier are separate outcomes, not these logscore gains.')


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('run')
    parser.add_argument('--output')
    args=parser.parse_args()
    result=audit(args.run)
    rendered=json.dumps(result,indent=2,allow_nan=False)+'\n'
    if args.output:
        with Path(args.output).open('x') as file:
            file.write(rendered)
    print(rendered)
