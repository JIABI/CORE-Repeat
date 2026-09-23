"""Prespecified MEDINA/USC morphology-neighbourhood endpoint for frozen R4 lists.

Only DEV profiles define the common anchor set. Cosines never cross sites:
cross-site agreement compares rankings over matching development identities.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from .r4_confirmatory_metrics import campaign_budget
from .r4_evaluation import load_selections, select

SITES = ('MEDINA', 'USC')
ANCHOR_SCHEMA = 'opal-r4-external-anchor-space-v1'


def _json(path):
    return json.loads(Path(path).read_text())


def _write(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n')


def _load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def _align(data, ids):
    own = np.asarray(data['ids'], str)
    if own.ndim != 1 or len(set(own)) != len(own) or set(own) != set(ids):
        raise ValueError('Unique exact population identities required at every site')
    lookup = {v: i for i, v in enumerate(own)}
    ix = np.asarray([lookup[v] for v in ids])
    return {k: (v[ix] if isinstance(v, np.ndarray) and v.ndim and len(v) == len(own)
                and k != 'role_order' else v) for k, v in data.items()}


def _valid_mean(data):
    y = np.asarray(data['Y'], float)
    if y.ndim != 3 or y.shape[1] != 4 or y.shape[0] != len(data['ids']):
        raise ValueError('All four designed roles are required')
    if 'role_order' in data and list(data['role_order']) != ['X', 'Z1', 'Z2', 'V']:
        raise ValueError('External role ordering changed')
    valid = np.isfinite(y).all(axis=(1, 2))
    for key in ('valid', 'present'):
        if key in data:
            if np.asarray(data[key]).shape != y.shape[:2]:
                raise ValueError('Role validity mask differs')
            valid &= np.asarray(data[key], bool).all(axis=1)
    means = y.mean(axis=1)
    valid &= np.linalg.norm(means, axis=1) > 0
    means[~valid] = np.nan
    return means, valid


def relation_vectors(query, anchors):
    """Within-site cosine vectors; invalid query rows stay entirely missing."""
    q, a = np.asarray(query, float), np.asarray(anchors, float)
    if q.ndim != 2 or a.ndim != 2 or q.shape[1] != a.shape[1]:
        raise ValueError('Query and anchors must share a site-specific feature space')
    an = np.linalg.norm(a, axis=1)
    if not np.isfinite(a).all() or np.any(an == 0):
        raise ValueError('Frozen anchors must be finite and nonzero')
    qn = np.linalg.norm(q, axis=1)
    valid = np.isfinite(q).all(axis=1) & (qn > 0)
    out = np.full((len(q), len(a)), np.nan)
    out[valid] = (q[valid]/qn[valid, None]) @ (a/an[:, None]).T
    return np.clip(out, -1, 1)


def spearman_rows(left, right):
    """Average ranks, no pairwise deletion and no zero-imputation of constants."""
    left, right = np.asarray(left, float), np.asarray(right, float)
    if left.ndim != 2 or left.shape != right.shape or left.shape[1] < 3:
        raise ValueError('At least three aligned anchor coordinates required')
    valid = np.isfinite(left).all(axis=1) & np.isfinite(right).all(axis=1)
    out = np.full(len(left), np.nan)
    if valid.any():
        x, y = rankdata(left[valid], axis=1), rankdata(right[valid], axis=1)
        x -= x.mean(axis=1, keepdims=True)
        y -= y.mean(axis=1, keepdims=True)
        denom = np.linalg.norm(x, axis=1)*np.linalg.norm(y, axis=1)
        nonconstant = denom > 0
        rows = np.flatnonzero(valid)[nonconstant]
        out[rows] = (x[nonconstant]*y[nonconstant]).sum(axis=1)/denom[nonconstant]
    return np.clip(out, -1, 1)


def common_anchor_arrays(fmp, external):
    """Freeze complete technical-valid DEV intersections, never correlation-filter."""
    if set(external) != set(SITES):
        raise ValueError('Both prespecified external sites are mandatory')
    ids = np.asarray(sorted(fmp['ids'].astype(str)), str)
    data = {'FMP': _align(fmp, ids), **{s: _align(external[s], ids) for s in SITES}}
    groups = data['FMP']['groups'].astype(str)
    if len(set(groups)) != len(groups):
        raise ValueError('Distinct development connectivity identities required')
    means, masks = {}, {}
    for site, d in data.items():
        if not np.array_equal(d['groups'].astype(str), groups):
            raise ValueError('Cross-site development connectivity differs')
        means[site], masks[site] = _valid_mean(d)
    common = np.logical_and.reduce(list(masks.values()))
    arrays = dict(ids=ids[common], groups=groups[common])
    arrays.update({site: values[common] for site, values in means.items()})
    record = dict(development_n=len(ids), common_anchor_n=int(common.sum()),
        site_valid_n={s: int(v.sum()) for s, v in masks.items()},
        excluded_development_ids=ids[~common].tolist(), anchor_ids=ids[common].tolist(),
        anchor_rule='All four roles technically valid at all three sites; finite nonzero '
                    'four-well means; common identity intersection, no performance filtering')
    return arrays, record


def _distribution(values):
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    return dict(observed_n=len(x), mean=float(x.mean()) if len(x) else None,
                q25=float(np.quantile(x, .25)) if len(x) else None,
                median=float(np.median(x)) if len(x) else None,
                q75=float(np.quantile(x, .75)) if len(x) else None)


def freeze_anchor_space(fmp_file, external_directories, output, *, confirmation_query):
    """Use only completed DEV exports, and refuse if confirmation X already exists."""
    output = Path(output)
    if output.exists():
        raise FileExistsError('Preserve the existing external anchor space')
    if Path(confirmation_query).exists():
        raise PermissionError('External anchors must be fixed before confirmation X')
    external = {}
    for site in SITES:
        directory = Path(external_directories[site])
        record = _json(directory/'complete.json')
        if not (record.get('complete') is True and record.get('stage') == 'anchors'
                and record.get('site') == site):
            raise PermissionError('A completed DEV-only anchor export is required')
        external[site] = _load(directory/'anchors.npz')
    arrays, record = common_anchor_arrays(_load(fmp_file), external)
    diagnostic = {}
    n = len(arrays['ids'])
    if n >= 4:
        fmp_relations = relation_vectors(arrays['FMP'], arrays['FMP'])
        others = {s: relation_vectors(arrays[s], arrays[s]) for s in SITES}
        for site in SITES:
            correlations = np.full(n, np.nan)
            for i in range(n):
                keep = np.arange(n) != i
                correlations[i] = spearman_rows(fmp_relations[i:i+1, keep],
                                                 others[site][i:i+1, keep])[0]
            arrays[site+'_development_leave_self_out_spearman'] = correlations
            diagnostic[site] = _distribution(correlations)
    record.update(schema=ANCHOR_SCHEMA, status='FROZEN',
        available=n >= 3, sites=list(SITES), created_utc=datetime.now(timezone.utc).isoformat(),
        source_fmp=str(Path(fmp_file).resolve()),
        source_external={s: str(Path(external_directories[s]).resolve()) for s in SITES},
        station_dimensions={s: arrays[s].shape[1] for s in ('FMP',)+SITES},
        development_diagnostic_leave_self_out=diagnostic,
        minimum_anchor_count=3, diagnostic_used_to_select_anchors=False,
        constant_relation_rule='missing; never correlation zero',
        missing_site_rule='both sites required for d; otherwise d missing in [-2,2]',
        feature_spaces='FMP unchanged DEV space; each external site DMSO-only space',
        confirmation_measurements_used=False)
    output.mkdir(parents=True)
    np.savez_compressed(output/'anchors.npz', **arrays)
    _write(output/'ANCHORS_FROZEN.json', record)
    return record


def external_endpoint(ids, x, z1_z2, external, anchors):
    """Return before/after agreement at BOTH sites and their fixed mean increment."""
    ids = np.asarray(ids, str)
    if set(external) != set(SITES) or set(ids) & set(anchors['ids']):
        raise ValueError('Both external sites and DEV-disjoint queries required')
    x, z = np.asarray(x, float), np.asarray(z1_z2, float)
    if z.shape != (len(ids), 2, x.shape[1]) or x.shape[0] != len(ids):
        raise ValueError('Only FMP X and the two action roles define query representations')
    before = relation_vectors(x, anchors['FMP'])
    after = relation_vectors((x+z.sum(axis=1))/3, anchors['FMP'])
    result = dict(ids=ids)
    deltas = []
    for site in SITES:
        d = _align(external[site], ids)
        mean, valid = _valid_mean(d)
        relation = relation_vectors(mean, anchors[site])
        a0, a1 = spearman_rows(before, relation), spearman_rows(after, relation)
        delta = a1-a0
        result.update({site+'_before': a0, site+'_after': a1, site+'_delta': delta,
                       site+'_four_well_valid': valid})
        deltas.append(delta)
    result['delta'] = np.mean(deltas, axis=0)
    result['observed'] = np.isfinite(result['delta'])
    return result


def bounded_contribution(delta, weights):
    """Paired missing bounds: a shared missing action cancels before bounding."""
    d, w = np.asarray(delta, float), np.asarray(weights, float)
    if d.ndim != 1 or w.shape != d.shape or not len(d) or not np.isfinite(w).all():
        raise ValueError('Aligned finite contribution weights required')
    known = np.isfinite(d)
    if np.isinf(d).any() or np.any(np.abs(d[known]) > 2+1e-12):
        raise ValueError('Neighbourhood increment must be missing or in [-2,2]')
    value = float(np.dot(w[known], d[known]))
    missing_extent = 2*float(np.abs(w[~known]).sum())
    return dict(lower=(value-missing_extent)/len(d), upper=(value+missing_extent)/len(d),
        point=value/len(d) if missing_extent == 0 else None,
        unresolved_contributing_n=int(np.count_nonzero(w[~known])), population_n=len(d))


def summarize_endpoint(result, policies, eligible):
    summary = {}
    for outcome in ('delta', 'MEDINA_delta', 'USC_delta'):
        d = result[outcome]
        policy_stats = {}
        for name, policy in policies.items():
            a = np.asarray(policy['selected'], bool)
            policy_stats[name] = dict(**bounded_contribution(d, a.astype(float)),
                selected_n=int(a.sum()), observed_selected_n=int((a & np.isfinite(d)).sum()),
                selected_mean_increment=float(d[a].mean()) if a.any() and np.isfinite(d[a]).all() else None)
        difference = bounded_contribution(d, policies['CORE']['selected'].astype(float)
                                          - policies['HISTGB_CAL']['selected'].astype(float))
        k = campaign_budget(len(d), int(np.sum(eligible)))['activations']
        random_weights = np.asarray(eligible, float)*(k/np.sum(eligible) if np.sum(eligible) else 0)
        summary[outcome] = dict(observed_n=int(np.isfinite(d).sum()), policies=policy_stats,
            primary_difference=difference, exact_random_expectation=bounded_contribution(d, random_weights))
    return summary


def external_resampling(ids, delta, eligible, policies, labels, *, repeats=10000, seed=20260923):
    """Same prespecified dependency sensitivity, with external [-2,2] bounds."""
    ids, labels, eligible = np.asarray(ids, str), np.asarray(labels, str), np.asarray(eligible, bool)
    _, inverse = np.unique(labels, return_inverse=True)
    blocks = [np.flatnonzero(inverse == i) for i in range(int(inverse.max())+1)]
    rng = np.random.default_rng(seed)
    draws = {mode: np.empty((repeats, 2)) for mode in ('fixed_list', 'new_campaign_topk')}
    for b in range(repeats):
        ix = np.concatenate([blocks[j] for j in rng.integers(0, len(blocks), len(blocks))])
        k = campaign_budget(len(ix), int(eligible[ix].sum()))['activations']
        for mode in draws:
            masks = {name: (policies[name]['selected'][ix] if mode == 'fixed_list' else
                select(ids[ix], policies[name]['score'][ix], eligible[ix], k))
                for name in ('CORE', 'HISTGB_CAL')}
            bounds = bounded_contribution(delta[ix], masks['CORE'].astype(float)-masks['HISTGB_CAL'].astype(float))
            draws[mode][b] = bounds['lower'], bounds['upper']
    return dict(blocks=len(blocks), repeats=repeats, seed=seed,
        interval_kind='approximate fixed-model dependence sensitivity, not a certificate',
        **{mode: dict(lower=float(np.quantile(d[:, 0], .025)),
                      upper=float(np.quantile(d[:, 1], .975)),
                      degenerate=bool(np.ptp(d) == 0)) for mode, d in draws.items()})


def evaluate_external_files(anchor_directory, query_file, future_file, external_files,
                            selections_directory, protocol_freeze, output):
    """Only accept already-authorized exports after all immutable list gates."""
    anchor_directory, output = Path(anchor_directory), Path(output)
    anchor_record = _json(anchor_directory/'ANCHORS_FROZEN.json')
    selection_record = _json(Path(selections_directory)/'SELECTIONS_FROZEN.json')
    freeze = _json(protocol_freeze)
    if (anchor_record.get('schema') != ANCHOR_SCHEMA or anchor_record.get('status') != 'FROZEN'
            or not anchor_record.get('available') or selection_record.get('status') != 'FROZEN'
            or freeze.get('status') != 'FROZEN'):
        raise PermissionError('Protocol, common anchor space and lists must be frozen')
    if output.exists():
        raise FileExistsError('Preserve existing external endpoint evaluation')
    ids, eligible, policies = load_selections(selections_directory)
    if list(ids) != selection_record['ids']:
        raise ValueError('Saved selections differ from their frozen identity list')
    for path, site, stage in [(query_file, 'FMP', 'x'), (future_file, 'FMP', 'outcomes')]+[
            (external_files[s], s, 'external_outcomes') for s in SITES]:
        completion = _json(Path(path).parent/'complete.json')
        if not (completion.get('complete') is True and completion.get('site') == site
                and completion.get('stage') == stage and completion.get('data_file') == Path(path).name):
            raise PermissionError('Only completed stage-authorized exports may enter evaluation')
    query, future = _align(_load(query_file), ids), _align(_load(future_file), ids)
    if list(future['role_order']) != ['Z1', 'Z2', 'V']:
        raise ValueError('FMP future role ordering changed')
    result = external_endpoint(ids, query['X'], future['future'][:, :2],
        {s: _load(external_files[s]) for s in SITES}, _load(anchor_directory/'anchors.npz'))
    result.update(groups=query['groups'], layout=query['layout'])
    summary = summarize_endpoint(result, policies, eligible)
    summary['resampling'] = {label: external_resampling(ids, result['delta'], eligible, policies,
        query[label]) for label in ('groups', 'layout')}
    summary['leave_one_layout'] = []
    for label in sorted(set(query['layout'])):
        keep = query['layout'] != label
        weights = policies['CORE']['selected'].astype(float)-policies['HISTGB_CAL']['selected'].astype(float)
        summary['leave_one_layout'].append(dict(omitted_layout=str(label),
            analysis='fixed-list descriptive sensitivity', **bounded_contribution(result['delta'][keep], weights[keep])))
    summary.update(complete=True, primary_comparison='CORE minus HISTGB_CAL',
        endpoint='mean MEDINA/USC morphology-neighbourhood Spearman improvement',
        sites=list(SITES), population_n=len(ids), anchor_n=anchor_record['common_anchor_n'],
        action_cost_deducted=False, primary_gamma_unchanged=True,
        biological_interpretation='cross-station morphology reproducibility, not target or clinical validation')
    output.mkdir(parents=True)
    np.savez_compressed(output/'endpoint_table.npz', **result)
    with (output/'endpoint_table.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result))
        writer.writeheader()
        writer.writerows({k: v[i] for k, v in result.items()} for i in range(len(ids)))
    _write(output/'summary.json', summary)
    return summary
