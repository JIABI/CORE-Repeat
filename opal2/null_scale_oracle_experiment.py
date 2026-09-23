"""Frozen CORE self-simulation reference for realized-outcome scale search.

Each synthetic truth is drawn in the existing nine-dimensional geometry and
then passed through the original Gamma. An independent second geometry is
scored using the parameter selected from the first. Candidate Monte Carlo
draws are reused across truths as an integration device, never as truths.
This reference is conditional on CORE, not a correction that estimates a
learnability ceiling.
"""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from .biology_borrowing_experiment import read_json, read_npz
from .biology_kernel_evaluation import write_json
from .conditional_joint_error_experiment import observable_forward
from .empirical_radial import draw_radial, fit_radial, radial_ppf
from .joint_contrast_scale import contrast_projector
from .variance_headroom_math import (adapter_grid, gamma_only, sample_components,
                                    residual_at_eta)


PROJECT = Path(__file__).resolve().parents[1]
RADIAL = PROJECT/'runs/lincs_empirical_radial_20260916_v1'
DUAL = PROJECT/'runs/dual_branch_biology_20260917_v2'
SEED = 17091729
REPLICATES = 20
SEARCH_SAMPLES = 4096
EVAL_SAMPLES = 100000
MC_BLOCKS = 20
ARMS = ('CORE', 'H1', 'H2')


def stage_rng(stage, cell, query=0):
    """Disjoint SeedSequence namespaces: truth=1, replication=2, search=3, eval=4."""
    if stage not in (1, 2, 3, 4):
        raise ValueError('Unknown random-stream purpose')
    return np.random.default_rng(np.random.SeedSequence([SEED, stage, int(cell), int(query)]))


def scores_from_sorted(ordered, targets):
    """Fair scalar CRPS for many targets using one sorted predictive ensemble."""
    values, y = np.asarray(ordered, float), np.atleast_1d(np.asarray(targets, float))
    if (values.ndim != 1 or len(values) < 2 or y.ndim != 1
            or not np.isfinite(values).all() or not np.isfinite(y).all()
            or np.any(values[1:] < values[:-1])):
        raise ValueError('Finite sorted draws and finite scalar targets required')
    n = len(values)
    prefix = np.r_[0., np.cumsum(values)]
    count = np.searchsorted(values, y, side='right')
    first = ((2*count-n)*y+prefix[-1]-2*prefix[count])/n
    pair_half = np.dot(2*np.arange(n)-n+1, values)/(n*(n-1))
    return first-pair_half


def _quantiles_sorted(ordered, levels=(.025, .975)):
    indices = np.asarray(levels)*(len(ordered)-1)
    lower, upper = np.floor(indices).astype(int), np.ceil(indices).astype(int)
    return ordered[lower]+(indices-lower)*(ordered[upper]-ordered[lower])


def synthetic_geometries(mean, scatter, stats, law, weights, cell, replicates=REPLICATES):
    """First and independent second realizations, never Gaussian Gamma draws."""
    n, d = np.asarray(mean).shape
    if d != 9 or replicates < 1:
        raise ValueError('Nine-dimensional conditional geometry required')
    output = {}
    for stage, label in ((1, 'first'), (2, 'second')):
        rng = stage_rng(stage, cell)
        error = draw_radial(law, weights, scatter, rng.normal(size=(replicates, n, d)),
                            rng.random((replicates, n)), rng.random((replicates, n)))
        geometry = np.asarray(mean)[None]+error
        gamma = observable_forward(geometry*np.asarray(stats['u_scale'])+np.asarray(stats['u_center']))[0]
        output[label+'_geometry'] = geometry.transpose(1, 0, 2)
        output[label+'_gamma'] = gamma.T
    return output


def search_targets(mean, scatter, stats, law, weights, gamma_targets, support,
                   cell, samples=SEARCH_SAMPLES, block_size=8):
    """One fixed 81-candidate integration grid serves real plus synthetic truths."""
    mean, targets, support = np.asarray(mean, float), np.asarray(gamma_targets, float), np.asarray(support, bool)
    n = len(mean)
    if targets.ndim != 2 or len(targets) != n or support.shape != (n,) or not np.isfinite(targets).all():
        raise ValueError('Targets must be finite [N,T] and support [N]')
    scale, center = np.asarray(stats['u_scale']), np.asarray(stats['u_center'])
    dec = contrast_projector(mean*scale+center, scale, scatter)
    grid = adapter_grid(9)
    scores = np.empty((n, len(grid), targets.shape[1]))
    for lo in range(0, n, block_size):
        hi = min(lo+block_size, n); ix = slice(lo, hi); m = hi-lo
        rng = stage_rng(3, cell, lo)
        normal, mix, kernel = rng.normal(size=(samples, m, 9)), rng.random((samples, m)), rng.random((samples, m))
        bdec = {k: v[ix] for k, v in dec.items()}
        base, contrast = sample_components(scatter[ix], bdec, law, weights[ix], normal, mix, kernel)
        for k, eta in enumerate(grid):
            gamma = gamma_only((mean[None, ix]+residual_at_eta(base, contrast, eta))*scale+center)
            ordered = np.sort(gamma, axis=0)
            for j in range(m):
                scores[lo+j, k] = scores_from_sorted(ordered[:, j], targets[lo+j])
    scalar = np.flatnonzero(grid[:, 0] == grid[:, 1])
    indices = {'CORE': np.zeros_like(targets, dtype=int),
               'H1': scalar[np.argmin(scores[:, scalar], axis=1)],
               'H2': np.argmin(scores, axis=1)}
    for index in indices.values():
        index[~support] = 0
    return dict(grid_eta=grid, search_crps=scores, **{arm+'_indices': index for arm, index in indices.items()})


def _energies(mean, geometry, decomposition):
    residual = np.asarray(geometry)-np.asarray(mean)[:, None]
    white = np.linalg.solve(decomposition['factor'], residual.swapaxes(1, 2)).swapaxes(1, 2)
    pair = np.einsum('nij,ntj->nti', decomposition['projector'], white)
    return np.stack((np.square(pair).sum(-1), np.square(white-pair).sum(-1)), axis=-1)


def evaluate_selected_union(mean, scatter, stats, law, weights, targets, geometries,
                            second_targets, second_geometries, search, cell,
                            samples=EVAL_SAMPLES, mc_blocks=MC_BLOCKS):
    """Independent MC; evaluate each selected candidate once for all its uses.

Column zero is the observed real outcome. The remaining columns are first
synthetic outcomes. Second outcomes have no real-data column: their scores
use the candidate chosen by the corresponding *first* synthetic outcome.
"""
    mean, targets, second_targets = map(lambda a: np.asarray(a, float), (mean, targets, second_targets))
    n, t = targets.shape
    if (second_targets.shape != (n, t-1) or samples < 4 or samples % mc_blocks
            or mc_blocks < 2 or samples//mc_blocks < 2):
        raise ValueError('Aligned second targets and independent MC blocks required')
    scale, center = np.asarray(stats['u_scale']), np.asarray(stats['u_center'])
    dec = contrast_projector(mean*scale+center, scale, scatter)
    energy = _energies(mean, geometries, dec)
    second_energy = _energies(mean, second_geometries, dec)
    radius95 = radial_ppf(law, weights, [.95])[:, 0]**2
    grid = search['grid_eta']
    out = {}
    for arm in ARMS:
        z = {}
        for key in ('crps', 'gamma95', 'joint95', 'predicted', 'p_null'):
            z[key] = np.empty((n, t)); z['second_'+key] = np.empty((n, t-1))
        z['crps_blocks'] = np.empty((n, t, mc_blocks))
        z['second_crps_blocks'] = np.empty((n, t-1, mc_blocks))
        eta = grid[search[arm+'_indices']]
        z['joint95'] = (np.sum(energy*np.exp(-eta), axis=-1) <= radius95[:, None]).astype(float)
        z['second_joint95'] = (np.sum(second_energy*np.exp(-eta[:, 1:]), axis=-1) <= radius95[:, None]).astype(float)
        out[arm] = z
    unions = np.empty(n, int)
    for j in range(n):
        rng = stage_rng(4, cell, j)
        normal, mix, kernel = rng.normal(size=(samples, 1, 9)), rng.random((samples, 1)), rng.random((samples, 1))
        bdec = {k: v[j:j+1] for k, v in dec.items()}
        base, contrast = sample_components(scatter[j:j+1], bdec, law, weights[j:j+1], normal, mix, kernel)
        union = np.unique(np.concatenate([search[a+'_indices'][j] for a in ARMS]))
        unions[j] = len(union)
        for k in union:
            gamma = gamma_only((mean[None, j:j+1]+residual_at_eta(base, contrast, grid[k]))*scale+center)[:, 0]
            ordered = np.sort(gamma)
            combined_targets = np.r_[targets[j], second_targets[j]]
            crps = scores_from_sorted(ordered, combined_targets)
            lower, upper = _quantiles_sorted(ordered)
            coverage = ((combined_targets >= lower)&(combined_targets <= upper)).astype(float)
            blocks = np.stack([scores_from_sorted(np.sort(block), combined_targets)
                               for block in np.split(gamma, mc_blocks)], axis=1)
            mean_gamma, null = float(gamma.mean()), float(np.mean(gamma <= 0))
            for arm in ARMS:
                take = search[arm+'_indices'][j] == k
                out[arm]['crps'][j, take] = crps[:t][take]
                out[arm]['gamma95'][j, take] = coverage[:t][take]
                out[arm]['crps_blocks'][j, take] = blocks[:t][take]
                out[arm]['predicted'][j, take] = mean_gamma
                out[arm]['p_null'][j, take] = null
                second_take = take[1:]
                out[arm]['second_crps'][j, second_take] = crps[t:][second_take]
                out[arm]['second_gamma95'][j, second_take] = coverage[t:][second_take]
                out[arm]['second_crps_blocks'][j, second_take] = blocks[t:][second_take]
                out[arm]['second_predicted'][j, second_take] = mean_gamma
                out[arm]['second_p_null'][j, second_take] = null
    for arm in ARMS:
        for prefix in ('', 'second_'):
            delta = out[arm][prefix+'crps_blocks']-out['CORE'][prefix+'crps_blocks']
            out[arm][prefix+'paired_crps_mc_se'] = delta.std(-1, ddof=1)/np.sqrt(mc_blocks)
        if not all(np.isfinite(v).all() for v in out[arm].values()):
            raise ValueError('Nonfinite or unfilled score arrays')
    return out, unions


def _summary(root, ids, support, cells, elapsed):
    lookup = {v: i for i, v in enumerate(ids)}; stores = {}; seen = np.zeros(len(ids), int)
    for arm in ARMS:
        values = []
        for cell in cells:
            file = read_npz(root/f'cell_{cell["fold"]}_{cell["half"]}'/(arm+'.npz'))
            q = np.asarray([lookup[v] for v in file['ids']]); values.append((q, file))
            if arm == 'CORE': np.add.at(seen, q, 1)
        keys = set(values[0][1])-{'ids'}
        out = {k: np.empty((len(ids), *values[0][1][k].shape[1:]), values[0][1][k].dtype) for k in keys}
        for q, value in values:
            assert set(value)-{'ids'} == keys
            for k in keys: out[k][q] = value[k]
        stores[arm] = out
        np.savez_compressed(root/(arm+'.npz'), ids=ids, support=support, **out)
    np.testing.assert_array_equal(seen, np.ones(len(ids), int))
    metrics = {}
    for subset, take in [('supported', support), ('full', np.ones(len(ids), bool))]:
        metrics[subset] = {}
        core = stores['CORE']['crps'][take]; core2 = stores['CORE']['second_crps'][take]
        for arm in ARMS:
            value = stores[arm]; score = value['crps'][take]; second = value['second_crps'][take]
            gain = (core[:, 1:]-score[:, 1:]).mean(0)
            gain2 = (core2-second).mean(0)
            metrics[subset][arm] = dict(
                n=int(take.sum()), real_core_crps=float(core[:, 0].mean()), real_crps=float(score[:, 0].mean()),
                real_absolute_gain=float((core[:, 0]-score[:, 0]).mean()),
                real_relative_gain=float(1-score[:, 0].mean()/core[:, 0].mean()),
                null_absolute_gains=gain.tolist(), null_relative_gains=(gain/core[:, 1:].mean(0)).tolist(),
                null_gain_mean=float(gain.mean()), null_gain_sd=float(gain.std(ddof=1)),
                null_gain_range=[float(gain.min()), float(gain.max())],
                second_absolute_gains=gain2.tolist(), second_relative_gains=(gain2/core2.mean(0)).tolist(),
                second_gain_mean=float(gain2.mean()), second_gain_sd=float(gain2.std(ddof=1)),
                real_joint95=float(value['joint95'][take, 0].mean()),
                null_joint95=value['joint95'][take, 1:].mean(0).tolist(),
                second_joint95=value['second_joint95'][take].mean(0).tolist(),
                real_gamma95=float(value['gamma95'][take, 0].mean()),
                null_gamma95=value['gamma95'][take, 1:].mean(0).tolist(),
                second_gamma95=value['second_gamma95'][take].mean(0).tolist(),
                paired_mc_se=(np.sqrt(np.square(value['paired_crps_mc_se'][take]).sum(0))/take.sum()).tolist(),
                second_paired_mc_se=(np.sqrt(np.square(value['second_paired_crps_mc_se'][take]).sum(0))/take.sum()).tolist())
    result = dict(state='COMPLETE', n=len(ids), supported_n=int(support.sum()), replicates=REPLICATES,
        search_samples=SEARCH_SAMPLES, evaluation_samples=EVAL_SAMPLES, metrics=metrics,
        elapsed_seconds=elapsed, grid_size=9, mean_changed=False, radial_law_changed=False,
        truth_space='Original nine-dimensional conditional geometry, then original Gamma',
        formal_core_goodness_of_fit_test=False, subtraction_is_learnability_estimator=False,
        interpretation='Conditional model-self-simulation reference for hindsight adaptation; no biological information ceiling.',
        numerical_integration='Candidate ensembles shared across synthetic truths; 20 paired MC blocks reported separately.',
        second_realization='Independent new geometry from CORE; parameters frozen from first pseudo realization.',
        new_model_training=False, protected_data_opened=False, cells=cells)
    write_json(root/'summary.json', result)
    lines = ['# Frozen CORE self-simulation reference', '', result['interpretation'], '',
             '| Arm | Real CRPS | Real gain | Mean self-simulation gain | Mean second-realization gain |',
             '|---|---:|---:|---:|---:|']
    for arm, v in metrics['supported'].items():
        lines.append(f'| {arm} | {v["real_crps"]:.8f} | {v["real_absolute_gain"]:.8f} | '
                     f'{v["null_gain_mean"]:.8f} | {v["second_gain_mean"]:.8f} |')
    lines += ['', 'Twenty pseudo datasets vary truths with CORE fixed. This is not a null-corrected estimate of learnable headroom.',
              'The real-data and same-realization oracles use their outcomes to choose parameters; neither is a deployment model.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return result


def run(output, protocol):
    start = time.monotonic(); root = Path(output).resolve(); root.mkdir(parents=True, exist_ok=True)
    protocol = Path(protocol).resolve()
    if not protocol.is_file(): raise ValueError('A frozen protocol file is required')
    for source in (protocol, Path(__file__), PROJECT/'opal2/variance_headroom_math.py'):
        destination = root/('PROTOCOL.md' if source == protocol else source.name)
        if destination.exists() and destination.read_bytes() != source.read_bytes():
            raise ValueError('Implementation/protocol changed; use a new output directory')
        shutil.copy2(source, destination)
    old = read_json(RADIAL/'summary.json')
    manifest = read_json(Path(old['reference_run'])/'run_manifest.json')
    prior = read_npz(RADIAL/'AMP_EMP_LOCAL.npz'); real = read_npz(DUAL/'CORE.npz')
    ids = prior['ids']; lookup = {v: i for i, v in enumerate(ids)}
    if len(ids) != 1188 or ids.tolist() != manifest['ids']: raise ValueError('Unexpected development objects')
    np.testing.assert_array_equal(real['ids'], ids)
    support = read_npz(DUAL/'GELU.npz')['resource_support'].astype(bool)
    spec = dict(seed=SEED, replicates=REPLICATES, search_samples=SEARCH_SAMPLES,
                evaluation_samples=EVAL_SAMPLES, grid_size=9, mc_blocks=MC_BLOCKS)
    if (root/'spec.json').exists() and read_json(root/'spec.json') != spec:
        raise ValueError('Run specification changed')
    write_json(root/'spec.json', spec)
    for cell_number, cell in enumerate(old['cells']):
        fold, half = cell['fold'], cell['half']; folder = root/f'cell_{fold}_{half}'
        folder.mkdir(exist_ok=True)
        if (folder/'complete.json').exists(): continue
        q = np.asarray([lookup[v] for v in cell['query_ids']])
        ref = read_npz(RADIAL/f'cell_{fold}_{half}_radial.npz')
        np.testing.assert_array_equal(ref['query_ids'], ids[q])
        stats = read_json(Path(old['reference_run'])/'folds'/f'fold_{fold}'/'preprocessing.json')
        mean, scatter = prior['mean_u'][q], prior['scatter_u'][q]
        law = fit_radial(ref['amplitude_radii']); weights = ref['local_weights']
        status = dict(state='RUNNING', cell=cell_number+1, cells=10, elapsed_seconds=time.monotonic()-start)
        write_json(root/'status.json', dict(status, stage='self_simulation_and_search'))
        synthetic = synthetic_geometries(mean, scatter, stats, law, weights, cell_number)
        targets = np.column_stack((real['actual'][q], synthetic['first_gamma']))
        geometries = np.concatenate((prior['actual_u'][q, None], synthetic['first_geometry']), axis=1)
        search = search_targets(mean, scatter, stats, law, weights, targets, support[q], cell_number)
        np.savez_compressed(folder/'search.npz', ids=ids[q], support=support[q], actual_gamma=real['actual'][q],
                            actual_geometry=prior['actual_u'][q], **synthetic, **search)
        write_json(root/'status.json', dict(status, stage='independent_candidate_union_evaluation'))
        print(f'null cell {cell_number+1}/10: independent candidate-union sampling', flush=True)
        out, union = evaluate_selected_union(mean, scatter, stats, law, weights, targets, geometries,
            synthetic['second_gamma'], synthetic['second_geometry'], search, cell_number)
        for arm, values in out.items():
            np.savez_compressed(folder/(arm+'.npz'), ids=ids[q], **values)
        record = dict(state='COMPLETE', query_n=len(q), supported_n=int(support[q].sum()),
                      union_mean=float(union.mean()), union_max=int(union.max()),
                      elapsed_seconds=time.monotonic()-start)
        write_json(folder/'complete.json', record)
        print(f'null cell {cell_number+1}/10 complete: {record}', flush=True)
    result = _summary(root, ids, support, old['cells'], time.monotonic()-start)
    write_json(root/'status.json', dict(state='COMPLETE', elapsed_seconds=time.monotonic()-start))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', required=True)
    parser.add_argument('--protocol', required=True); args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.output, args.protocol)
