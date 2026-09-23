"""Fixed-cell, descriptive residual-coherence audit of saved RxRx3 DEV.

Reconstructs the existing biological weights with the original random seed;
does not train, select models, or change the R3 predictive pipeline.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from threadpoolctl import threadpool_limits

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from opal2.dual_branch_features import FIELDS
from opal2.eu_r3_biology import build_eu_biology_features
from opal2.joint_contrast_scale import contrast_projector
from opal2.rxrx3_r3_cache import load_cell
from scripts.run_r3_rxrx3_modules_20260920 import enriched_scope, SEED

SOURCE = PROJECT / 'runs/r3_rxrx3_modules_20260920_v1'
OUTPUT = PROJECT / 'runs/r3_biology_support_pilot_20260920_v1'
CELLS = (0, 3, 5, 7)
DOSES = (0.0025, 0.1, 1.0, 10.0)
SPACES = ('full', 'P3', 'P6')
RELATIONS = ('target', 'moa')
ASSIGNMENTS = ('real', 'matched_random')
DEGREES = np.array([9.0, 3.0, 6.0])
SHRINKAGES = (0.25, 0.5, 1.0)


def write_json(path, value):
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        raise TypeError(type(obj).__name__)
    Path(path).write_text(json.dumps(value, indent=2, default=convert, allow_nan=False) + '\n')


def paired_interval(real, random, groups, valid, *, replicates=2000):
    valid = np.asarray(valid, bool)
    values = np.asarray(real)[valid] - np.asarray(random)[valid]
    labels = np.asarray(groups)[valid]
    if not len(values):
        return dict(n=0, chemical_groups=0, real_mean=None, random_mean=None,
                    difference=None, ci95=None)
    unique, inverse = np.unique(labels, return_inverse=True)
    sizes = np.bincount(inverse)
    sums = np.bincount(inverse, weights=values)
    rng = np.random.default_rng(SEED + 47000)
    draws = np.empty(replicates)
    for start in range(0, replicates, 64):
        stop = min(replicates, start + 64)
        indices = rng.integers(len(unique), size=(stop-start, len(unique)))
        draws[start:stop] = sums[indices].sum(1) / sizes[indices].sum(1)
    return dict(n=len(values), chemical_groups=len(unique),
                real_mean=float(np.asarray(real)[valid].mean()),
                random_mean=float(np.asarray(random)[valid].mean()),
                difference=float(values.mean()),
                ci95=None if len(unique) < 2 else np.quantile(draws, [.025, .975]).tolist())


def audit_cell(index, expected_dose, scope):
    cell = load_cell(f'fold_{index}', scope=scope)
    data, part, stats = (cell[k] for k in ('data', 'part', 'stats'))
    arrays = cell['oldarrays']
    if int(cell['cell_info']['outer_fold']) != 0 or float(cell['cell_info']['dose_uM']) != expected_dose:
        raise ValueError('Fixed pilot cell/dose allocation changed')
    meta = dict(scope['biology_metadata'])
    meta['units'] = [meta['units'][int(i)] for i in cell['global_rows']]
    with np.load(SOURCE / f'fold_{index}' / 'module_predictions.npz', allow_pickle=False) as cache:
        q_all = part['DEV_EVAL']
        np.testing.assert_array_equal(cache['query_ids'], data['ids'][q_all])
        support = cache['query_support'].copy()
        cached = cache['query_biology'][support].copy()
    q, r, t = q_all[support], part['REF_FIT'], part['TRAIN']
    scale, center = np.asarray(stats['u_scale']), np.asarray(stats['u_center'])
    rawmean = cell['means'] * scale + center
    covariance = (arrays['query_scatter_u'][support]
                  * arrays['radial_variance_multiplier'][support, None, None]
                  * scale[None, :, None] * scale[None, None, :])
    donor_residual = cell['raw'][r] - rawmean[r]
    logamp = np.log(np.linalg.norm(data['Y'][:, 0], axis=1))
    edges = np.quantile(logamp[t], [.2, .4, .6, .8])
    random_seed = SEED + 100 * index + 7000
    randomized = build_eu_biology_features(data, meta, q, r, rawmean[q], covariance,
        donor_residual, random_seed=random_seed, amplitude_edges=edges)
    diagnostics = randomized['audit']['relation_diagnostics']
    decomposition = contrast_projector(rawmean[q], np.ones(9), covariance)
    # QUERY outcomes enter only the following descriptive audit, never weights.
    query_residual = cell['raw'][q] - rawmean[q]
    count = np.zeros((len(q), 2), int)
    ess = np.zeros((len(q), 2))
    available = np.zeros((len(q), 2), bool)
    offdiag_defined = np.zeros((len(q), 2), bool)
    weights = np.zeros((len(q), 2, 2, len(r)))
    shape = (len(q), 2, 2, 3)
    names = ('meanE', 'Emean', 'jensen_gap', 'coherence', 'coherent_fraction',
             'query_alignment', 'query_energy', 'mean_cosine')
    metrics = {name: np.full(shape, np.nan) for name in names}
    for shrinkage in SHRINKAGES:
        metrics[f'shrink_mse_{shrinkage:g}'] = np.full(shape, np.nan)
        metrics[f'shrink_mse_gain_{shrinkage:g}'] = np.full(shape, np.nan)
    donor_means = np.full((*shape, 9), np.nan)
    for i in range(len(q)):
        whitened = np.linalg.solve(decomposition['factor'][i], donor_residual.T).T
        white_query = np.linalg.solve(decomposition['factor'][i], query_residual[i])
        pair = whitened @ decomposition['projector'][i]
        query_pair = white_query @ decomposition['projector'][i]
        frames = (whitened, pair, whitened-pair)
        query_frames = (white_query, query_pair, white_query-query_pair)
        for relation, diagnostic in enumerate(diagnostics):
            original = diagnostic['original_similarity'][i]
            count[i, relation] = np.count_nonzero(original > 0)
            if count[i, relation] == 0:
                continue
            available[i, relation] = True
            for assignment, label in enumerate(('original_similarity', 'randomized_similarity')):
                similarities = diagnostic[label][i]
                w = similarities / similarities.sum()
                weights[i, relation, assignment] = w
                w2 = float(np.square(w).sum())
                ess[i, relation] = 1.0 / w2
                offdiag_defined[i, relation] = 1.0-w2 > 1e-12
                for space, (frame, qr, degrees) in enumerate(zip(frames, query_frames, DEGREES, strict=True)):
                    position = (i, relation, assignment, space)
                    energy = np.square(frame).sum(1) / degrees
                    mean = w @ frame
                    mean_energy = float(w @ energy)
                    squared_mean = float(mean @ mean / degrees)
                    query_energy = float(qr @ qr / degrees)
                    alignment = float(qr @ mean / degrees)
                    metrics['meanE'][position] = mean_energy
                    metrics['Emean'][position] = squared_mean
                    metrics['jensen_gap'][position] = mean_energy-squared_mean
                    metrics['query_alignment'][position] = alignment
                    metrics['query_energy'][position] = query_energy
                    if mean_energy > 0:
                        metrics['coherent_fraction'][position] = squared_mean / mean_energy
                    norm_product = np.linalg.norm(qr) * np.linalg.norm(mean)
                    if norm_product > 0:
                        metrics['mean_cosine'][position] = float(qr @ mean / norm_product)
                    if offdiag_defined[i, relation]:
                        metrics['coherence'][position] = (squared_mean - float(np.square(w) @ energy)) / (1.0-w2)
                    for shrinkage in SHRINKAGES:
                        error = qr-shrinkage*mean
                        mse = float(error @ error / degrees)
                        metrics[f'shrink_mse_{shrinkage:g}'][position] = mse
                        metrics[f'shrink_mse_gain_{shrinkage:g}'][position] = query_energy-mse
                    donor_means[position] = mean
    energy_error = 0.0
    random_error = 0.0
    for relation in range(2):
        take = available[:, relation]
        for space, field in ((1, 'pair_log_energy'), (2, 'remainder_log_energy')):
            column = relation*len(FIELDS) + FIELDS.index(field)
            np.testing.assert_allclose(cached[take, column],
                np.log1p(metrics['meanE'][take, relation, 0, space]), atol=1e-11, rtol=1e-11)
            np.testing.assert_allclose(randomized['values'][take, column],
                np.log1p(metrics['meanE'][take, relation, 1, space]), atol=1e-11, rtol=1e-11)
            if take.any():
                energy_error = max(energy_error, float(np.max(np.abs(
                    np.expm1(cached[take, column])-metrics['meanE'][take, relation, 0, space]))))
                random_error = max(random_error, float(np.max(np.abs(
                    np.expm1(randomized['values'][take, column])-metrics['meanE'][take, relation, 1, space]))))
    np.testing.assert_array_equal(available, randomized['support_by_relation'])
    if np.nanmin(metrics['jensen_gap']) < -1e-10:
        raise ValueError('Jensen inequality failed')
    expected_finite = np.broadcast_to(available[:, :, None, None], shape)
    for name in ('meanE', 'Emean', 'jensen_gap', 'query_alignment', 'query_energy'):
        if not np.isfinite(metrics[name][expected_finite]).all():
            raise ValueError(f'Unexpected nonfinite {name}')
        np.testing.assert_allclose(metrics[name][..., 0],
            (3*metrics[name][..., 1]+6*metrics[name][..., 2])/9,
            atol=1e-11, rtol=1e-11, equal_nan=True)
    identity = dict(ids=data['ids'][q], groups=data['groups'][q],
                    cell=np.full(len(q), index), dose=np.full(len(q), expected_dose),
                    available=available, count=count, ess=ess,
                    offdiag_defined=offdiag_defined)
    np.savez_compressed(OUTPUT / f'coherence_cell_{index:02d}.npz', **identity, **metrics,
        donor_ids=data['ids'][r], donor_groups=data['groups'][r],
        donor_raw_residual=donor_residual, query_raw_residual=query_residual,
        query_raw_mean=rawmean[q], query_raw_covariance=covariance,
        donor_mean_whitened=donor_means, weights=weights,
        cached_query_biology=cached, random_seed=random_seed,
        spaces=np.array(SPACES), relations=np.array(RELATIONS), assignments=np.array(ASSIGNMENTS))
    record = dict(cell=index, dose=expected_dose, query_n=len(q_all), supported_query_n=len(q),
        supported_chemical_groups=len(np.unique(data['groups'][q])), donor_n=len(r),
        relation_counts=available.sum(0), singleton_counts=((count == 1)&available).sum(0),
        random_seed=random_seed, cached_energy_max_absolute_error=energy_error,
        randomized_energy_max_absolute_error=random_error,
        cached_energy_matches_meanE=True,
        cached_energy_aggregation='weighted mean of squared residual norms; can equal squared mean for singleton support')
    print(json.dumps(record, default=lambda obj: obj.tolist() if isinstance(obj, np.ndarray) else obj), flush=True)
    return {**identity, **metrics}, record


def summarize(arrays, mask):
    result = {}
    for relation, label in enumerate(RELATIONS):
        result[label] = {}
        for space, label_space in enumerate(SPACES):
            row = {}
            for metric in ('meanE', 'Emean', 'jensen_gap', 'coherence', 'coherent_fraction',
                           'query_alignment', 'mean_cosine',
                           *(f'shrink_mse_gain_{a:g}' for a in SHRINKAGES)):
                real = arrays[metric][:, relation, 0, space]
                random = arrays[metric][:, relation, 1, space]
                valid = mask & arrays['available'][:, relation] & np.isfinite(real) & np.isfinite(random)
                row[metric] = paired_interval(real, random, arrays['groups'], valid)
            result[label][label_space] = row
    return result


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    scope = enriched_scope()
    records, pieces = [], []
    with threadpool_limits(limits=1):
        for index, dose in zip(CELLS, DOSES, strict=True):
            piece, record = audit_cell(index, dose, scope)
            records.append(record)
            pieces.append(piece)
        arrays = {key: np.concatenate([piece[key] for piece in pieces]) for key in pieces[0]}
        np.savez_compressed(OUTPUT/'coherence_per_query.npz', **arrays,
            spaces=np.array(SPACES), relations=np.array(RELATIONS), assignments=np.array(ASSIGNMENTS))
        report = dict(complete=True, cells=records, n_supported_conditions=len(arrays['ids']),
            n_supported_chemical_groups=len(np.unique(arrays['groups'])),
            pooled=summarize(arrays, np.ones(len(arrays['ids']), bool)),
            by_dose={str(dose): summarize(arrays, arrays['dose'] == dose) for dose in DOSES},
            inference='Descriptive DEV, fixed models and one matched random assignment; chemical-group paired bootstrap keeps doses together; 2000 replicates, no multiplicity adjustment.',
            fitting=False, outcome_based_cell_selection=False,
            query_outcomes_used_for_weights=False, query_outcomes_used_for_descriptive_alignment=True,
            raw_images_opened=False, confirmation_opened=False,
            metric_axes=['query', 'relation', 'assignment', 'space'],
            definitions={
                'meanE': 'sum_j w_j ||e_j||^2 / d',
                'Emean': '||sum_j w_j e_j||^2 / d',
                'jensen_gap': 'meanE-Emean; nonnegative weighted within-donor dispersion',
                'coherence': '(Emean-sum_j w_j^2 ||e_j||^2/d)/(1-sum_j w_j^2); off-diagonal weighted dot product per dimension, not cosine; singleton undefined',
                'coherent_fraction': 'Emean/meanE; singleton equals 1 and is not independent evidence of coherence',
                'query_alignment': 'query whitened residual dot weighted donor mean / d',
                'shrink_mse_gain': '||query residual||^2/d - ||query residual-alpha*donor mean||^2/d; alpha fixed at .25,.5,1, never fitted or selected',
                'spaces': 'full d=9; P3 pair-observable Jacobian subspace d=3; P6 orthogonal remainder d=6, not an identified biological/shared-noise decomposition',
                'undefined_values': 'NPZ NaN only for unsupported relations, singleton off-diagonal coherence, or zero-norm ratios; JSON aggregates use null when no eligible observations',
                'difference': 'real minus matched_random, condition-weighted point estimates'},
            aggregate_energy_cache_max_absolute_error=max(c['cached_energy_max_absolute_error'] for c in records))
        write_json(OUTPUT/'coherence_summary.json', report)
    print(json.dumps(dict(complete=True, output=str(OUTPUT),
        n_supported_conditions=report['n_supported_conditions'],
        n_supported_chemical_groups=report['n_supported_chemical_groups'])), flush=True)


if __name__ == '__main__':
    main()
