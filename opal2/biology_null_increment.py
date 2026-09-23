"""Reference-only biological NULL residual borrowing on opened LINCS DEV.

The original STATE50 predictor is frozen. Each auxiliary fit uses only the
opposite reference half of its own original outer test fold. Probabilities
from other outer folds are never used for auxiliary training, because those
models can have trained on the current query labels.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
from scipy.special import logit
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .biology_kernel_evaluation import write_json
from .lincs_biology_experiment import load_data
from .reference_information_memory import cosine_relationship

PROJECT = Path(__file__).resolve().parents[1]
SEED = 20260916
ALPHAS = (0., .25, .5, 1.)
BIO_ARMS = ('TARGET', 'MOA', 'BOTH')
ARMS = ('STATE50', 'GENERIC', *BIO_ARMS)
EPS = .001


def bound_probability(p):
    return np.clip(np.asarray(p, float), EPS, 1-EPS)


def fit_generic(direction, descriptors, labels, fit, predict):
    """Fit all preprocessing and a fixed regularized probability readout."""
    fit, predict = np.asarray(fit), np.asarray(predict)
    pca = PCA(n_components=4, svd_solver='full').fit(direction[fit])
    xfit = np.c_[descriptors[fit], pca.transform(direction[fit])]
    xpred = np.c_[descriptors[predict], pca.transform(direction[predict])]
    scaler = StandardScaler().fit(xfit)
    if len(np.unique(labels[fit])) < 2:
        value = (labels[fit].sum()+.5)/(len(fit)+1)
        return np.full(len(predict), value)
    model = LogisticRegression(C=1., solver='lbfgs', max_iter=2000)
    model.fit(scaler.transform(xfit), labels[fit])
    if model.n_iter_[0] >= model.max_iter:
        raise RuntimeError('Probability readout did not converge')
    return bound_probability(model.predict_proba(scaler.transform(xpred))[:, 1])


def borrow(similarity, query_groups, donor_groups, donor_residual, shrinkage=8.):
    """Reference errors, not query outcomes, determine the correction."""
    rel = np.asarray(similarity, float).copy()
    if np.any(rel < 0) or not np.isfinite(rel).all():
        raise ValueError('Invalid relationship')
    rel[np.asarray(query_groups)[:, None] == np.asarray(donor_groups)[None]] = 0
    total = rel.sum(1)
    support = total > 0
    w = np.divide(rel, total[:, None], out=np.zeros_like(rel), where=total[:, None] > 0)
    sq = np.square(w).sum(1)
    neff = np.divide(1., sq, out=np.zeros(len(w)), where=sq > 0)
    correction = (w@donor_residual)*neff/(neff+shrinkage)
    return correction, support, neff


def choose_alpha(base_cal, correction_cal, cal_labels):
    losses = [float(np.mean((bound_probability(base_cal+a*correction_cal)-cal_labels)**2))
              for a in ALPHAS]
    j = min(range(len(ALPHAS)), key=lambda j: (losses[j], ALPHAS[j]))
    return ALPHAS[j], losses


def combine_corrections(corrections, supports):
    count = sum(s.astype(int) for s in supports)
    return np.divide(sum(corrections), count, out=np.zeros(len(count)), where=count > 0)


def biological_prediction(cell, matrices, permutation, groups, labels):
    fit, cal, query = [cell[k] for k in ('fit', 'cal', 'query')]
    rows = np.r_[cal, query]
    corrections, supports, effective = [], [], []
    for matrix in matrices:
        corr, support, neff = borrow(matrix[np.ix_(permutation[rows], permutation[fit])],
            groups[rows], groups[fit], cell['fit_residual'])
        corrections.append(corr); supports.append(support); effective.append(neff)
    all_corrections = [*corrections, combine_corrections(corrections, supports)]
    all_supports = [*supports, supports[0] | supports[1]]
    predictions, choices = {}, {}
    for arm, corr, support in zip(BIO_ARMS, all_corrections, all_supports):
        alpha, losses = choose_alpha(cell['cal_probability'], corr[:len(cal)], labels[cal])
        qprob = bound_probability(cell['query_probability']+alpha*corr[len(cal):])
        # No relationship support must preserve the entire generic route.
        np.testing.assert_array_equal(qprob[~support[len(cal):]],
            cell['query_probability'][~support[len(cal):]])
        predictions[arm] = qprob
        choices[arm] = dict(alpha=alpha, calibration_brier=losses,
            query_support=int(support[len(cal):].sum()),
            effective_query_references=(effective[BIO_ARMS.index(arm)][len(cal):].tolist()
                if arm != 'BOTH' else None))
    return predictions, choices


def make_permutation_blocks(data, metadata, folds, logamplitude):
    """Chemical groups are atomic. Multi-object groups stay fixed."""
    groups, ids = data['groups'], data['ids']
    unique, count = np.unique(groups, return_counts=True)
    size = dict(zip(unique, count))
    bins = np.zeros(len(ids), int)
    for fold in np.unique(folds):
        take = folds == fold
        edges = np.quantile(logamplitude[take], [1/3, 2/3])
        bins[take] = np.searchsorted(edges, logamplitude[take], side='right')
    blocks = {}
    for i, unit in enumerate(metadata['units']):
        if size[groups[i]] != 1:
            continue
        key = (int(folds[i]), str(unit['roles']['X']['batch_number']),
            str(unit['cell_line']), str(unit['actual_dose_uM']), int(bins[i]),
            bool(data['target_mask'][i]), bool(data['moa_mask'][i]))
        blocks.setdefault(key, []).append(i)
    return [np.asarray(v, int) for v in blocks.values() if len(v) > 1]


def permute_blocks(n, blocks, rng):
    result = np.arange(n)
    for block in blocks:
        result[block] = rng.permutation(block)
    return result


def fixed_budget(ids, probability, cells):
    chosen = np.zeros(len(ids), bool)
    for c in cells:
        q = c['query']
        order = np.lexsort((ids[q], probability[q]))
        chosen[q[order[:c['budget']]]] = True
    return chosen


def grouped_intervals(a, b, y, actual, sel_a, sel_b, labels, seed, draws=2000):
    """Fixed-prediction paired resampling, not a post-selection certificate."""
    names, inverse = np.unique(labels, return_inverse=True)
    rng = np.random.default_rng(seed)
    brier_difference = (a-y)**2-(b-y)**2
    samples = []
    for _ in range(draws):
        wgroup = np.bincount(rng.integers(len(names), size=len(names)), minlength=len(names))
        w = wgroup[inverse]
        if w[y == 0].sum() == 0 or w[y == 1].sum() == 0:
            continue
        ma, mb = w*sel_a, w*sel_b
        fdp_difference = (np.sum(ma*y)/ma.sum()-np.sum(mb*y)/mb.sum()
            if ma.sum() and mb.sum() else np.nan)
        samples.append([np.average(brier_difference, weights=w),
            roc_auc_score(y, a, sample_weight=w)-roc_auc_score(y, b, sample_weight=w),
            np.average((sel_a-sel_b.astype(int))*actual, weights=w), fdp_difference])
    rows = np.asarray(samples)
    return {key: dict(ci95=np.nanquantile(rows[:, j], [.025, .975]).tolist())
        for j, key in enumerate(('brier_difference', 'auc_difference', 'all_candidate_value_difference',
                                  'fdp_difference'))}


def run(source, output, permutations=199):
    started = time.monotonic()
    source, root = Path(source).resolve(), Path(output).resolve()
    if root.exists():
        raise FileExistsError(root)
    summary = json.loads((source/'summary.json').read_text())
    reference = Path(summary['reference_run'])
    manifest = json.loads((reference/'run_manifest.json').read_text())
    data, metadata = load_data(manifest['data_directory'])
    ids, groups = data['ids'], data['groups']; n = len(ids)
    if n != 1188 or ids.tolist() != manifest['ids']:
        raise ValueError('Opened universe changed')
    root.mkdir(parents=True)
    shutil.copy2(__file__, root/'biology_null_increment.py')
    write_json(root/'status.json', dict(state='RUNNING', phase='reference_fits'))
    raw_p, raw_m, actual, folds = [np.full(n, np.nan) for _ in range(4)]
    for record in manifest['folds']:
        rows = np.asarray(record['test'])
        folder = reference/'folds'/f"fold_{record['fold']}"/'arms/STATE50/evaluation'
        with np.load(folder/'predictions.npz', allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], ids[rows])
            raw_p[rows] = z['p_null'][:, 2]
            raw_m[rows] = z['predicted'][:, 2]
            actual[rows] = z['actual'][:, 2]
            folds[rows] = record['fold']
    with np.load(source/'GAUSSIAN.npz', allow_pickle=False) as z:
        np.testing.assert_array_equal(z['ids'], ids)
        np.testing.assert_allclose(actual, z['actual'], atol=1e-12, rtol=1e-12)
    if not np.isfinite(np.c_[raw_p, raw_m, actual, folds]).all():
        raise ValueError('Incomplete frozen model predictions')
    y = (actual <= 0).astype(int)
    norm = np.linalg.norm(data['Y'][:, 0], axis=1)
    if np.any(norm <= 0):
        raise ValueError('Zero first-well norm requires explicit original completion')
    direction = data['Y'][:, 0]/norm[:, None]
    logamp = np.log(norm)
    counts = np.asarray([u['roles']['X']['cell_count'] for u in metadata['units']], float)
    descriptors = np.c_[logit(bound_probability(raw_p)), logamp, np.log1p(counts),
                        data['target_mask'], data['moa_mask']]
    lookup = {v: i for i, v in enumerate(ids)}
    probabilities = {a: np.empty(n) for a in ARMS}
    probabilities['STATE50'] = bound_probability(raw_p)
    cells, seen = [], np.zeros(n, int)
    records = {r['fold']: r for r in manifest['folds']}
    for c in summary['cells']:
        fit, cal, query = [np.asarray([lookup[v] for v in c[k]])
            for k in ('fit_ids', 'calibration_ids', 'query_ids')]
        record = records[c['fold']]
        old_train = np.r_[record['fit'], record['inner_validation']]
        parts = [set(groups[v]) for v in (old_train, fit, cal, query)]
        if any(parts[i] & parts[j] for i in range(4) for j in range(i+1, 4)):
            raise ValueError('Model/reference/query chemical group leakage')
        if set(np.r_[fit, cal, query]) != set(record['test']):
            raise ValueError('Reference split changed')
        fprob = np.full(len(fit), np.nan)
        splits = []
        for train, test in GroupKFold(n_splits=3).split(fit, groups=groups[fit]):
            fprob[test] = fit_generic(direction, descriptors, y, fit[train], fit[test])
            splits.append(dict(train_ids=ids[fit[train]].tolist(), test_ids=ids[fit[test]].tolist()))
        pred = fit_generic(direction, descriptors, y, fit, np.r_[cal, query])
        probabilities['GENERIC'][query] = pred[len(cal):]
        cell = dict(fold=c['fold'], half=c['half'], fit=fit, cal=cal, query=query,
            budget=c['budget'], fit_residual=y[fit]-fprob, cal_probability=pred[:len(cal)],
            query_probability=pred[len(cal):], inner_reference_splits=splits)
        cells.append(cell); seen[query] += 1
        print(f"BIO generic reference fit {c['fold']}/{c['half']} done", flush=True)
    if not np.all(seen == 1):
        raise ValueError('Every query must be evaluated exactly once')
    matrices = [cosine_relationship(data[key], data[key], data[key+'_mask'], data[key+'_mask'])
                for key in ('target', 'moa')]
    choices = []
    identity = np.arange(n)
    for cell in cells:
        predicted, selected = biological_prediction(cell, matrices, identity, groups, y)
        for arm in BIO_ARMS:
            probabilities[arm][cell['query']] = predicted[arm]
        choices.append(dict(fold=cell['fold'], half=cell['half'], choices=selected,
            fit_ids=ids[cell['fit']], cal_ids=ids[cell['cal']], query_ids=ids[cell['query']],
            inner_reference_splits=cell['inner_reference_splits']))
    layout = np.asarray([u['layout_block'] for u in metadata['units']])
    masks = {a: fixed_budget(ids, probabilities[a], cells) for a in ARMS}
    metrics = {}
    for arm, p in probabilities.items():
        chosen = masks[arm]
        metrics[arm] = dict(brier=float(np.mean((p-y)**2)), auc=float(roc_auc_score(y, p)),
            predicted_null_mean=float(p.mean()), selected_n=int(chosen.sum()),
            selected_null=int(y[chosen].sum()), selected_positive=int((actual[chosen] >= .005).sum()),
            selected_actual_mean=float(actual[chosen].mean()),
            all_candidate_value=float(np.mean(chosen*actual)),
            selected_mean_predicted_probability=float(p[chosen].mean()),
            selected_symmetric_difference_from_generic=int(np.count_nonzero(chosen != masks['GENERIC'])))
        np.savez_compressed(root/(arm+'.npz'), ids=ids, groups=groups, layout=layout, fold=folds,
            actual=actual, predicted=raw_m, p_null=p, selected=chosen,
            brier=(p-y)**2, policy_value=chosen*actual)
    write_json(root/'reference_choices.json', choices)
    write_json(root/'metrics.json', metrics)
    comparisons = {}
    for arm, base in [('GENERIC', 'STATE50'), *[(a, 'GENERIC') for a in BIO_ARMS]]:
        comparisons[arm+' minus '+base] = {name: grouped_intervals(probabilities[arm], probabilities[base],
            y, actual, masks[arm], masks[base], labels, SEED+91)
            for name, labels in [('chemistry', groups), ('layout', layout)]}
    write_json(root/'comparisons.json', comparisons)
    blocks = make_permutation_blocks(data, metadata, folds, logamp)
    rng = np.random.default_rng(SEED+318)
    observed_gain = {a: metrics['GENERIC']['brier']-metrics[a]['brier'] for a in BIO_ARMS}
    permuted_gain = np.empty((permutations, len(BIO_ARMS)))
    changed = np.empty(permutations, int)
    base_brier = metrics['GENERIC']['brier']
    for step in range(permutations):
        perm = permute_blocks(n, blocks, rng)
        changed[step] = np.count_nonzero(perm != identity)
        pp = {a: np.empty(n) for a in BIO_ARMS}
        for cell in cells:
            pred, _ = biological_prediction(cell, matrices, perm, groups, y)
            for arm in BIO_ARMS:
                pp[arm][cell['query']] = pred[arm]
        permuted_gain[step] = [base_brier-float(np.mean((pp[a]-y)**2)) for a in BIO_ARMS]
        if (step+1) % 20 == 0 or step == permutations-1:
            write_json(root/'status.json', dict(state='RUNNING', phase='annotation_permutation',
                complete=step+1, total=permutations, elapsed_seconds=time.monotonic()-started))
            print(f'BIO permutation {step+1}/{permutations}', flush=True)
    permutation_result = {a: dict(observed_brier_gain=observed_gain[a],
        null_quantiles=np.quantile(permuted_gain[:, j], [.025, .5, .975]),
        exploratory_p=(1+int(np.count_nonzero(permuted_gain[:, j] >= observed_gain[a])))/(permutations+1))
        for j, a in enumerate(BIO_ARMS)}
    np.savez_compressed(root/'permutation.npz', arms=np.asarray(BIO_ARMS),
        brier_gain=permuted_gain, changed_rows=changed)
    result = dict(n=n, total_null=int(y.sum()), metrics=metrics, comparisons=comparisons,
        annotation_permutation=permutation_result, permutation_count=permutations,
        movable_annotation_rows=sum(map(len, blocks)), annotation_rows_changed_quantiles=np.quantile(changed, [0,.5,1]),
        source_frozen_model=str(reference), primary_comparison='TARGET/MOA/BOTH versus GENERIC',
        generic_inputs=['STATE50 logit PNULL', 'log norm X', 'log1p observed X cell count',
                        'four FIT-only PCA components of normalized X', 'target known', 'MoA known'],
        nll_or_joint_model_changed=False, original_mean_prediction_changed=False,
        p_null_not_derived_from_a_new_joint_distribution=True,
        scope='Opened DEV, original five outer folds and ten reference/query units; auxiliary fitting and CAL selection only within each reference half',
        permutation_interpretation='Stratified annotation-association sensitivity, not an exact conditional test given full morphology',
        ci_scope='Fixed-prediction chemical/layout paired bootstrap, no full pipeline refitting or repeated-development uncertainty',
        reference_cost_included=False, final_opened=False, elapsed_seconds=time.monotonic()-started)
    write_json(root/'summary.json', result)
    write_json(root/'status.json', dict(state='COMPLETE', elapsed_seconds=result['elapsed_seconds']))
    print(json.dumps(metrics, indent=2), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=PROJECT/'runs/lincs_empirical_radial_20260916_v1')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--permutations', type=int, default=199)
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        run(args.source, args.output, args.permutations)
