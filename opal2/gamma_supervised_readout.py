"""Supplement the frozen main comparison using saved predictions only."""
import argparse
import json
from pathlib import Path

import numpy as np

from .biology_kernel_evaluation import write_json
from .gamma_supervised_summary import ARMS


def analyze(root):
    root = Path(root).resolve()
    manifest = json.loads((root/'run_manifest.json').read_text())
    summary = json.loads((root/'summary.json').read_text())
    if not summary['complete']:
        raise ValueError('Complete the main comparison first')
    result = dict(scope='saved checkpoint/readout analysis, no refitting or selection', models={})
    for arm in ARMS:
        errors_before, errors_after, errors_gamma, covariance_rows, history_rows = [], [], [], [], []
        for record in manifest['folds']:
            folder = root/'folds'/f"fold_{record['fold']}"/'arms'/arm
            with np.load(folder/'evaluation/predictions.npz', allow_pickle=False) as saved:
                eb = saved['predicted_observables'][:,2]-saved['actual_observables'][:,2]
                eg = saved['predicted'][:,2]-saved['actual'][:,2]
                ea = 2*eg+eb
                errors_before.extend(eb); errors_after.extend(ea); errors_gamma.extend(eg)
                n = len(saved['ids'])
            diagnostic = json.loads((folder/'evaluation/u_diagnostics.json').read_text())
            covariance_rows.append((n, diagnostic['coverage']))
            monitor = folder/'gamma_monitoring.jsonl'
            if monitor.is_file():
                history_rows.append([json.loads(line) for line in monitor.read_text().splitlines()])
        eb, ea, eg = map(np.asarray, (errors_before, errors_after, errors_gamma))
        bm, am, cross, gm = (float(np.mean(eb**2)), float(np.mean(ea**2)),
            float(np.mean(eb*ea)), float(np.mean(eg**2)))
        identity_error = gm-.25*(am+bm-2*cross)
        if abs(identity_error) > 1e-12:
            raise ValueError('The original Gamma/cosine-error identity failed')
        coverage = []
        for i, reference in enumerate(covariance_rows[0][1]):
            row = dict(level=reference['level'])
            for key in ('marginal_coverage', 'joint_ellipsoid_coverage'):
                row[key] = sum(n*values[i][key] for n, values in covariance_rows)/639
            coverage.append(row)
        curve = []
        for epoch in (0,5,10,15,20,25,30):
            rows = [next(row for row in history if row['epoch']==epoch) for history in history_rows]
            if not rows:
                continue
            fields = ('fit_u_mse', 'validation_u_mse', 'fit_gamma_crps', 'validation_gamma_crps',
                'fit_normalized_gamma_crps', 'validation_normalized_gamma_crps', 'validation_gamma_mse',
                'geometry_gradient_norm', 'normalized_gamma_gradient_norm', 'gradient_cosine')
            row = dict(epoch=epoch, folds=len(rows), aggregation='equal-fold mean')
            for key in fields:
                row[key] = float(np.mean([r[key] for r in rows])) if all(r[key] is not None for r in rows) else None
            row['per_fold_gradient_cosine'] = [r['gradient_cosine'] for r in rows]
            row['per_fold_gamma_to_geometry_gradient_ratio'] = [r['normalized_gamma_gradient_norm']/r['geometry_gradient_norm']
                if r['geometry_gradient_norm'] > 0 else None for r in rows]
            curve.append(row)
        result['models'][arm] = dict(before_cosine_mse=bm, after_cosine_mse=am,
            prediction_error_cross_moment=cross, gamma_mse=gm, identity_error=identity_error,
            gamma_mean_error=float(np.mean(eg)),
            gamma_squared_mean_error=float(np.mean(eg)**2),
            gamma_centered_error_mse=float(np.mean((eg-np.mean(eg))**2)),
            coverage=coverage, checkpoint_curve=curve,
            error_cross_moment_is_not_physical_covariance=True)
    j, k = (result['models'][name] for name in ARMS[1:])
    result['K_minus_J_gamma_mse'] = dict(total=k['gamma_mse']-j['gamma_mse'],
        squared_mean_error_contribution=k['gamma_squared_mean_error']-j['gamma_squared_mean_error'],
        centered_error_contribution=k['gamma_centered_error_mse']-j['gamma_centered_error_mse'],
        separate_cosine_mse_contribution=.25*(k['before_cosine_mse']+k['after_cosine_mse']
            -j['before_cosine_mse']-j['after_cosine_mse']),
        paired_error_contribution=-.5*(k['prediction_error_cross_moment']-j['prediction_error_cross_moment']))
    result['gamma_scale_by_fold'] = [json.loads((root/'folds'/f"fold_{r['fold']}"/'gamma_objective.json').read_text())['gamma_scale']
        for r in manifest['folds']]
    write_json(root/'saved_readout_diagnostics.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root')
    print(json.dumps(analyze(parser.parse_args().root), ensure_ascii=False, allow_nan=False))
