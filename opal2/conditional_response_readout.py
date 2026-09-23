"""Read-only analysis of saved means, functional errors and joint coverage.

No additional fitting or checkpoint selection. This post-training analysis
supplements, and never overwrites, the predeclared main summary.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .biology_kernel_evaluation import write_json
from .conditional_response_summary import ARMS, HISTORICAL_ARMS


def analyze(root):
    root = Path(root).resolve()
    manifest = json.loads((root/'run_manifest.json').read_text())
    summary = json.loads((root/'summary.json').read_text())
    if not summary['complete']:
        raise ValueError('Complete the declared evaluation first')
    result = dict(scope='saved predictions only; no training, object deletion or checkpoint selection', models={})
    for arm in (*ARMS, *HISTORICAL_ARMS):
        before_errors, after_errors, gain_errors, coverages, trajectories = [], [], [], [], []
        for record in manifest['folds']:
            if arm in ARMS:
                folder = root/'folds'/f"fold_{record['fold']}"/'arms'/arm
                evaluation = folder/'evaluation'
            else:
                folder = Path(manifest['historical_reference_run'])/'folds'/f"fold_{record['fold']}"/'arms'/arm
                evaluation = folder/'test'
            with np.load(evaluation/'predictions.npz', allow_pickle=False) as saved:
                before = saved['predicted_observables'][:, 2]
                actual_before = saved['actual_observables'][:, 2]
                predicted_gain = saved['predicted'][:, 2]
                actual_gain = saved['actual'][:, 2]
                # E[cos(after,V)] = 2(E[Gamma]+cost) + E[cos(X,V)].
                after = 2*(predicted_gain+.02)+before
                actual_after = 2*(actual_gain+.02)+actual_before
                before_errors.extend(before-actual_before)
                after_errors.extend(after-actual_after)
                gain_errors.extend(predicted_gain-actual_gain)
                n = len(saved['ids'])
            d = json.loads((evaluation/'u_diagnostics.json').read_text())
            coverages.append((n, d['coverage']))
            path = folder/'history.jsonl'
            if path.is_file():
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                trajectories.append([row for row in rows if row['epoch'] in (0,5,10,15,20,25,30)])
        eb, ea, eg = map(np.asarray, (before_errors, after_errors, gain_errors))
        mse_b, mse_a, cross, mse_g = (float(np.mean(eb**2)), float(np.mean(ea**2)),
                                     float(np.mean(eb*ea)), float(np.mean(eg**2)))
        if abs(mse_g-.25*(mse_a+mse_b-2*cross)) > 1e-12:
            raise ValueError('The original gain and paired cosine error identity disagree')
        cov = []
        for k in range(len(coverages[0][1])):
            row = dict(level=coverages[0][1][k]['level'])
            for key in ('marginal_coverage', 'joint_ellipsoid_coverage', 'mean_marginal_width'):
                row[key] = sum(n*values[k][key] for n, values in coverages)/sum(n for n, _ in coverages)
            row['marginal_per_coordinate'] = (sum(n*np.asarray(values[k]['marginal_per_coordinate'])
                for n, values in coverages)/sum(n for n, _ in coverages)).tolist()
            cov.append(row)
        curve = []
        if trajectories:
            for epoch in (0,5,10,15,20,25,30):
                rows = [next((row for row in history if row['epoch']==epoch), None) for history in trajectories]
                rows = [row for row in rows if row is not None]
                if rows:
                    curve.append(dict(epoch=epoch, folds_available=len(rows),
                        fit_u_mse=float(np.mean([row['fit_u_mse'] for row in rows])),
                        validation_u_mse=float(np.mean([row['validation_u_mse'] for row in rows]))))
        result['models'][arm] = dict(before_cosine_mse=mse_b, after_cosine_mse=mse_a,
            prediction_error_cross_moment=cross, gamma_mse=mse_g,
            identity_error=mse_g-.25*(mse_a+mse_b-2*cross), coverage=cov,
            training_trajectory_equal_fold_average=curve,
            error_cross_moment_is_not_physical_covariance=True)
    baseline = result['models']['A_HR']
    for arm in ARMS[1:]:
        row = result['models'][arm]
        row['gamma_mse_change_from_hr'] = row['gamma_mse']-baseline['gamma_mse']
        row['marginal_cosine_error_contribution'] = .25*(row['before_cosine_mse']+row['after_cosine_mse']
            -baseline['before_cosine_mse']-baseline['after_cosine_mse'])
        row['cross_error_contribution'] = -.5*(row['prediction_error_cross_moment']
            -baseline['prediction_error_cross_moment'])
    write_json(root/'saved_readout_diagnostics.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root')
    print(json.dumps(analyze(parser.parse_args().root), ensure_ascii=False, allow_nan=False))
